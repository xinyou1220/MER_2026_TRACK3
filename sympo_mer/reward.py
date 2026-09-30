"""SymPO composite reward (paper Sec. 4.3).

For a sample with slot decision d and orientation o, the chosen candidate is c = d XOR o.

  r_corr = +1 / -1 on labeled rows, 0 otherwise                 (Eq. 6)
  r_swap = +1 if c matches the majority of the opposite order    (Sec. 4.3a)
  r_bt   = tanh((u+ - u-) / 2)       out-of-fold BT margin        (Eq. 9)
  r_len  = tanh(3 e)                 relative length excess       (Eq. 10)

bounded:   R = w_c r_corr + tau * tanh(w_s r_swap + w_b r_bt - w_l r_len)       (Eq. 11)
unbounded: R = w_c r_corr + w_s r_swap + w_b (u+ - u-) - w_l e                  (ablation)

Invalid outputs:
  invalid_floor: R = -(w_c + tau_labeled + m)                                  (Eq. 12)
  valid_bonus:   valid rows get +w_format, invalid rows get -(1 + w_format)    (naive RL)
  additive:      every row gets +/- w_format and keeps its other channels       (ablation)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardInfo:
    reward: torch.Tensor
    correct: torch.Tensor
    has_label: torch.Tensor
    consistency: torch.Tensor
    slot1_rate: float


def _majority_consistency(chosen, pair_id, orientation, valid):
    consistency = torch.zeros_like(chosen, dtype=torch.float)
    chosen_f = chosen.float()
    for pid in torch.unique(pair_id):
        pair_mask = pair_id == pid
        for o in (0, 1):
            own = pair_mask & (orientation == o) & valid
            other = pair_mask & (orientation == 1 - o) & valid
            if own.any() and other.any():
                consensus = (chosen_f[other].mean() >= 0.5).long()
                consistency[own] = (chosen[own] == consensus).float()
    return consistency


def _continuous_consistency(chosen, pair_id, orientation, valid):
    consistency = torch.zeros_like(chosen, dtype=torch.float)
    chosen_f = chosen.float()
    for pid in torch.unique(pair_id):
        pair_mask = pair_id == pid
        mask0 = pair_mask & (orientation == 0) & valid
        mask1 = pair_mask & (orientation == 1) & valid
        if mask0.any() and mask1.any():
            consistency[pair_mask & valid] = 1.0 - torch.abs(chosen_f[mask0].mean() - chosen_f[mask1].mean())
    return consistency


def compute_rewards(decision, valid, pair_id, orientation, gold,
                    len_slot1, len_slot2, bt_slot1, bt_slot2, cfg) -> RewardInfo:
    d = decision.clamp_min(0)
    chosen = torch.where(orientation == 0, d, 1 - d)

    if cfg.swap_consistency_mode == "majority":
        consistency = _majority_consistency(chosen, pair_id, orientation, valid)
    elif cfg.swap_consistency_mode == "continuous":
        consistency = _continuous_consistency(chosen, pair_id, orientation, valid)
    else:
        raise ValueError(f"Unsupported swap_consistency_mode: {cfg.swap_consistency_mode!r}")
    r_swap = consistency * 2.0 - 1.0

    has_label = gold >= 0
    correct = (chosen == gold) & has_label
    r_corr = torch.where(correct, 1.0, -1.0) * has_label.float()

    bt_margin = torch.where(d == 0, bt_slot1, bt_slot2) - torch.where(d == 0, bt_slot2, bt_slot1)
    r_bt = (torch.sigmoid(bt_margin) - 0.5) * 2.0
    len_chosen = torch.where(d == 0, len_slot1, len_slot2)
    len_other = torch.where(d == 0, len_slot2, len_slot1)
    len_excess = (len_chosen - len_other) / (len_chosen + len_other).clamp_min(1.0)
    r_len = torch.tanh(len_excess * 3.0)

    if cfg.shaping_mode == "bounded":
        z = cfg.w_bt * r_bt - cfg.w_len * r_len + cfg.w_swap * r_swap
        tau = torch.where(has_label, cfg.tau_labeled, cfg.tau_unlabeled)
        aux_reward = tau * torch.tanh(z)
    elif cfg.shaping_mode == "unbounded":
        aux_reward = cfg.w_bt * bt_margin - cfg.w_len * len_excess + cfg.w_swap * r_swap
    else:
        raise ValueError(f"Unsupported shaping_mode: {cfg.shaping_mode!r}")
    reward = cfg.w_correct * r_corr + aux_reward

    r_fmt = torch.where(valid, 1.0, -1.0)
    if cfg.format_mode == "invalid_floor":
        floor = -(cfg.w_correct + cfg.tau_labeled + cfg.invalid_margin)
        reward = torch.where(valid, reward, torch.full_like(reward, floor))
    elif cfg.format_mode == "valid_bonus":
        reward = reward + cfg.w_format * r_fmt
        reward = torch.where(valid, reward, torch.full_like(reward, -cfg.w_format - 1.0))
    elif cfg.format_mode == "additive":
        reward = reward + cfg.w_format * r_fmt
    else:
        raise ValueError(f"Unsupported format_mode: {cfg.format_mode!r}")

    slot1_rate = (d[valid] == 0).float().mean().item() if valid.any() else 0.0
    return RewardInfo(reward=reward, correct=correct, has_label=has_label, consistency=consistency,
                      slot1_rate=slot1_rate)
