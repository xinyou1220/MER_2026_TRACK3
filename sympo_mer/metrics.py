"""WAF-2 and the swap-averaged score used for checkpoint selection."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from .rollout import make_slot_batch


def waf2(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """Support-weighted average of the two per-class F1 scores."""
    total = 0.0
    for c in (0, 1):
        tp = ((pred == c) & (gold == c)).sum().item()
        fp = ((pred == c) & (gold != c)).sum().item()
        fn = ((pred != c) & (gold == c)).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        total += f1 * (gold == c).sum().item()
    return total / gold.numel()


@torch.no_grad()
def evaluate_waf2_swap(policy, pairs, device="cpu", batch_size=1):
    """WAF-2 of predictions from P(a2 preferred) averaged over both presentation orders."""
    preds, golds = [], []
    for start in tqdm(range(0, len(pairs), batch_size), dynamic_ncols=True, desc="eval"):
        chunk = pairs[start:start + batch_size]
        slot_batch, meta = make_slot_batch(chunk, orientations=[0, 1], group_k=1, device=device)
        score_a, score_b = policy.score_ab_logprobs(slot_batch)
        slot_b = torch.softmax(torch.stack([score_a.float(), score_b.float()], dim=1), dim=1)[:, 1]
        a2_prob = torch.where(meta["orientation"] == 0, slot_b, 1.0 - slot_b)
        for pair in chunk:
            preds.append(1 if a2_prob[meta["pair_id"] == pair.pair_id].mean().item() > 0.5 else 0)
            golds.append(pair.gold)
    return waf2(torch.tensor(preds), torch.tensor(golds))
