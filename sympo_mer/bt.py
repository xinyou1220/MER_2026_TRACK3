"""Out-of-fold Bradley-Terry text ranker that provides the dense bt_a / bt_b utilities (paper Eq. 9).

P(a1 preferred) = sigmoid(U(a1) - U(a2)), with U linear in hashed unigram + bigram features of the
candidate text, trained by SGD logistic regression on feature differences. Utility differences are
rescaled to unit standard deviation. Every labeled training row is scored by a ranker that did not
see it (stratified K-fold).
"""

from __future__ import annotations

import math
import random
import re
import zlib
from typing import Any, Dict, List, Optional, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def first_wins(row: Dict[str, Any]) -> Optional[int]:
    label = str(row.get("preference", "")).strip().lower()
    return {"a1": 1, "a2": 0}.get(label)


def candidate_texts(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("a1") or ""), str(row.get("a2") or "")


class HashTextBTRanker:
    def __init__(self, seed: int, hash_features: int = 65536, epochs: int = 5, lr: float = 0.2,
                 l2: float = 1e-6, target_diff_std: float = 1.0, max_tokens: int = 512):
        self.seed = seed
        self.hash_features = hash_features
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.target_diff_std = target_diff_std
        self.max_tokens = max_tokens
        self.weights: Dict[int, float] = {}
        self.utility_shift = 0.0
        self.utility_scale = 1.0

    def _index(self, token: str) -> int:
        return zlib.crc32(token.encode("utf-8")) % self.hash_features

    def _features(self, text: str) -> Dict[int, float]:
        tokens = TOKEN_RE.findall(text.lower())[: self.max_tokens]
        counts: Dict[int, float] = {}
        prev = None
        for token in tokens:
            idx = self._index(token)
            counts[idx] = counts.get(idx, 0.0) + 1.0
            if prev is not None:
                idx = self._index(f"{prev} {token}")
                counts[idx] = counts.get(idx, 0.0) + 1.0
            prev = token
        if not counts:
            return {}
        values = {idx: 1.0 + math.log(value) for idx, value in counts.items()}
        norm = math.sqrt(sum(v * v for v in values.values())) or 1.0
        return {idx: v / norm for idx, v in values.items()}

    @staticmethod
    def _subtract(a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
        out = dict(a)
        for idx, value in b.items():
            new_value = out.get(idx, 0.0) - value
            if abs(new_value) < 1e-12:
                out.pop(idx, None)
            else:
                out[idx] = new_value
        return out

    def _dot(self, features: Dict[int, float]) -> float:
        return sum(self.weights.get(idx, 0.0) * value for idx, value in features.items())

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        examples = []
        for row in rows:
            label = first_wins(row)
            a_text, b_text = candidate_texts(row)
            if label is None or not a_text or not b_text:
                continue
            a, b = self._features(a_text), self._features(b_text)
            examples.append((self._subtract(a, b), label, a, b))

        rng = random.Random(self.seed)
        for epoch in range(self.epochs):
            rng.shuffle(examples)
            step_lr = self.lr / math.sqrt(epoch + 1.0)
            for features, label, _, _ in examples:
                score = max(-40.0, min(40.0, self._dot(features)))
                grad = 1.0 / (1.0 + math.exp(-score)) - float(label)
                for idx, value in features.items():
                    weight = self.weights.get(idx, 0.0)
                    self.weights[idx] = weight - step_lr * (grad * value + self.l2 * weight)

        diffs, utilities = [], []
        for _, _, a, b in examples:
            ua, ub = self._dot(a), self._dot(b)
            diffs.append(ua - ub)
            utilities.extend([ua, ub])
        mean_diff = sum(diffs) / len(diffs)
        std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / max(1, len(diffs)))
        if self.target_diff_std > 0 and std_diff > 1e-12:
            self.utility_scale = self.target_diff_std / std_diff
        scaled = [u * self.utility_scale for u in utilities]
        self.utility_shift = sum(scaled) / len(scaled)

    def utility(self, text: str) -> float:
        return self._dot(self._features(text)) * self.utility_scale - self.utility_shift


def oof_utilities(rows: List[Dict[str, Any]], folds: int = 5, seed: int = 42) -> Tuple[List[float], List[float]]:
    """Stratified K-fold out-of-fold utilities for every labeled row."""
    by_label: Dict[int, List[int]] = {0: [], 1: []}
    for idx, row in enumerate(rows):
        label = first_wins(row)
        if label is not None and all(candidate_texts(row)):
            by_label[label].append(idx)

    n_folds = min(folds, len(by_label[0]) + len(by_label[1]))
    rng = random.Random(seed)
    fold_rows: List[List[int]] = [[] for _ in range(n_folds)]
    for label in (0, 1):
        indices = list(by_label[label])
        rng.shuffle(indices)
        for pos, idx in enumerate(indices):
            fold_rows[pos % n_folds].append(idx)
    fold_rows = [fold for fold in fold_rows if fold]

    bt_a: List[Optional[float]] = [None] * len(rows)
    bt_b: List[Optional[float]] = [None] * len(rows)
    for fold_idx, heldout in enumerate(fold_rows):
        heldout_set = set(heldout)
        ranker = HashTextBTRanker(seed=seed + fold_idx)
        ranker.fit([row for idx, row in enumerate(rows) if idx not in heldout_set])
        for idx in heldout:
            a_text, b_text = candidate_texts(rows[idx])
            bt_a[idx], bt_b[idx] = ranker.utility(a_text), ranker.utility(b_text)

    # Rows without a usable label are scored by a ranker fitted on every labeled row.
    missing = [idx for idx, value in enumerate(bt_a) if value is None]
    if missing:
        ranker = HashTextBTRanker(seed=seed + len(fold_rows))
        ranker.fit(rows)
        for idx in missing:
            a_text, b_text = candidate_texts(rows[idx])
            bt_a[idx], bt_b[idx] = ranker.utility(a_text), ranker.utility(b_text)
    return bt_a, bt_b
