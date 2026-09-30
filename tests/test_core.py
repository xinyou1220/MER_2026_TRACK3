"""CPU checks for the reward, advantage, scoring and metric code (no model needed).

    python tests/test_core.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sympo_mer.advantage import advantage_reinforce_baseline
from sympo_mer.config import RewardConfig
from sympo_mer.data import Pair
from sympo_mer.metrics import waf2
from sympo_mer.reward import compute_rewards
from sympo_mer.rollout import make_slot_batch
from sympo_mer.scoring import canonical_tie_prediction, decide


def _pair(i, gold=0):
    return Pair(pair_id=i, gold=gold, len0=10.0, len1=20.0, bt0=0.5, bt1=-0.5,
                sample_id=f"s{i}", cand0_text=f"first {i}", cand1_text=f"second {i}")


def _rewards(decision, orientation, gold, cfg=None, bt=(0.0, 0.0)):
    n = len(decision)
    decision = torch.tensor(decision)
    return compute_rewards(
        decision=decision, valid=decision >= 0, pair_id=torch.zeros(n, dtype=torch.long),
        orientation=torch.tensor(orientation), gold=torch.tensor(gold),
        len_slot1=torch.full((n,), 10.0), len_slot2=torch.full((n,), 10.0),
        bt_slot1=torch.full((n,), bt[0]), bt_slot2=torch.full((n,), bt[1]), cfg=cfg or RewardConfig(),
    )


def test_balanced_swap_layout():
    slot_batch, meta = make_slot_batch([_pair(0), _pair(1)], [0, 1], group_k=2)
    assert meta["group_id"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert meta["orientation"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert slot_batch["candidate_slot1"][:4] == ["first 0", "first 0", "second 0", "second 0"]
    assert meta["bt_slot1"][2].item() == -0.5


def test_positional_policy_earns_zero_correctness():
    cfg = RewardConfig(w_swap=0.0, w_bt=0.0, w_len=0.0)
    for slot in (0, 1):
        for gold in (0, 1):
            info = _rewards([slot, slot], [0, 1], [gold, gold], cfg)
            assert info.reward.sum().item() == 0.0


def test_bounded_envelope_keeps_correct_above_wrong():
    info = _rewards([0, 1], [0, 0], [0, 0], bt=(-50.0, 50.0))
    assert info.correct.tolist() == [True, False]
    assert info.reward[0] - info.reward[1] >= 2 * (1.0 - 0.2) - 1e-6


def test_invalid_floor():
    info = _rewards([-1, 0], [0, 0], [0, 0])
    assert abs(info.reward[0].item() + 1.7) < 1e-6
    assert info.reward[0] < info.reward[1]


def test_majority_consistency():
    # Slot A, then slot B after the swap: the same candidate both times.
    assert _rewards([0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]).consistency.tolist() == [1.0] * 4
    # Always slot A: a different candidate in each order.
    assert _rewards([0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]).consistency.tolist() == [0.0] * 4


def test_group_mean_and_global_norm():
    scores = torch.tensor([1.0, 0.0, 2.0, 2.0])
    group = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(4, 3)
    adv, centered, degenerate = advantage_reinforce_baseline(scores, group, mask, 1e-6)
    assert centered.tolist() == [0.5, -0.5, 0.0, 0.0]
    assert abs(adv.mean().item()) < 1e-6 and abs(adv.pow(2).mean().sqrt().item() - 1.0) < 1e-4
    _, _, degenerate = advantage_reinforce_baseline(scores, group, mask, 1e-6, normalization_mode="per_group")
    assert degenerate == 0.5


def test_antisymmetric_decision_is_order_invariant():
    row = {"margin_ori0": 0.25, "margin_ori1": -1.0, "tie_break_pred": 0}
    swapped = {"margin_ori0": 1.0, "margin_ori1": -0.25, "tie_break_pred": 1}
    pred, margin = decide(row, "antisym")
    pred_s, margin_s = decide(swapped, "antisym")
    assert pred == 1 and pred_s == 0 and abs(margin + margin_s) < 1e-12


def test_tie_break_follows_candidate_identity():
    pair = _pair(0)
    swapped = Pair(pair_id=0, gold=0, len0=20.0, len1=10.0, bt0=0.0, bt1=0.0,
                   cand0_text=pair.cand1_text, cand1_text=pair.cand0_text)
    assert canonical_tie_prediction(pair) == 1 - canonical_tie_prediction(swapped)


def test_waf2():
    assert waf2(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1])) == 1.0
    assert abs(waf2(torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 1, 1])) - 1 / 3) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
