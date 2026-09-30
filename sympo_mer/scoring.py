"""Constrained A/B scoring in both orders, the antisymmetrized decision and the evaluation metrics."""

from __future__ import annotations

import hashlib
import json

import torch
from tqdm.auto import tqdm

from .metrics import waf2
from .rollout import make_slot_batch


def canonical_tie_prediction(pair) -> int:
    """Tie-break that picks the same candidate in either presentation order."""
    signatures = []
    for index in (0, 1):
        # "qbin" is constant; it is kept so that tie-breaks match the original runs.
        payload = {"text": getattr(pair, f"cand{index}_text").strip(), "qbin": 0,
                   "length": float(getattr(pair, f"len{index}"))}
        signatures.append(hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).digest())
    return 0 if signatures[0] <= signatures[1] else 1


@torch.no_grad()
def score_pairs(policy, pairs, device: str, both_orders: bool = True) -> list[dict]:
    """Per pair, the margin log P(a1) - log P(a2) in the original order and, optionally, swapped."""
    records = []
    for pair in tqdm(pairs, dynamic_ncols=True, desc="score"):
        slot_batch, _ = make_slot_batch([pair], orientations=[0, 1] if both_orders else [0], group_k=1, device=device)
        score_a, score_b = (s.float().cpu().tolist() for s in policy.score_ab_logprobs(slot_batch))
        tie = canonical_tie_prediction(pair)
        row = {"name": pair.sample_id, "gold": pair.gold, "margin_ori0": score_a[0] - score_b[0],
               "margin_ori1": None, "tie_break_pred": tie}
        if both_orders:
            # Swapped order: slot B holds a1.
            row["margin_ori1"] = score_b[1] - score_a[1]
        records.append(row)
    return records


def _sign(margin: float, tie: int) -> int:
    return 0 if margin > 0.0 else (1 if margin < 0.0 else tie)


def decide(row: dict, inference_mode: str) -> tuple[int, float]:
    """Return (prediction, margin); prediction 0 = a1 preferred, 1 = a2 preferred."""
    if inference_mode == "antisym":
        margin = 0.5 * (row["margin_ori0"] + row["margin_ori1"])
    else:
        margin = row["margin_ori0"]
    return _sign(margin, row["tie_break_pred"]), margin


def evaluation_metrics(records: list[dict], predictions: list[int]) -> dict:
    """WAF-2 and accuracy of the deployed predictions, plus the per-order symmetry diagnostics."""
    gold = [r["gold"] for r in records]
    pred0 = [_sign(r["margin_ori0"], r["tie_break_pred"]) for r in records]
    pred1 = [_sign(r["margin_ori1"], r["tie_break_pred"]) for r in records]
    slot1_rate = (sum(p == 0 for p in pred0) + sum(p == 1 for p in pred1)) / (2 * len(records))
    return {
        "waf2": waf2(torch.tensor(predictions), torch.tensor(gold)),
        "accuracy": sum(p == g for p, g in zip(predictions, gold)) / len(gold),
        "swap_consistency": sum(a == b for a, b in zip(pred0, pred1)) / len(records),
        "position_bias": abs(slot1_rate - 0.5),
        # Mean |symmetric part| of the slot log-odds, which the antisymmetric average discards.
        "positional_leakage": sum(abs(0.5 * (r["margin_ori0"] - r["margin_ori1"])) for r in records) / len(records),
    }
