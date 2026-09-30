"""Balanced-swap rollouts with group sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .reward import RewardInfo, compute_rewards


@dataclass
class Rollout:
    seq_ids: torch.Tensor
    attn: torch.Tensor
    action_mask: torch.Tensor
    old_logp: torch.Tensor
    aux: dict = field(default_factory=dict)


@dataclass
class RolloutData:
    rollout: Rollout
    reward: torch.Tensor
    group_id: torch.Tensor
    info: RewardInfo


def make_slot_batch(pairs, orientations, group_k, device="cpu"):
    """Lay out each pair in each orientation, group_k times; each (pair, orientation) is one group.

    Orientation 0 shows (a1, a2) as (A, B); orientation 1 shows (a2, a1).
    """
    slot_keys = ("candidate_slot1", "candidate_slot2", "video_path", "audio_path", "asr_text")
    meta_keys = ("group_id", "pair_id", "orientation", "gold", "len_slot1", "len_slot2", "bt_slot1", "bt_slot2")
    cols = {key: [] for key in slot_keys + meta_keys}
    group = 0
    for p in pairs:
        for o in orientations:
            first, second = ((p.len0, p.bt0, p.cand0_text), (p.len1, p.bt1, p.cand1_text))[:: 1 if o == 0 else -1]
            for _ in range(group_k):
                for key, value in zip(meta_keys, (group, p.pair_id, o, p.gold, first[0], second[0], first[1], second[1])):
                    cols[key].append(value)
                for key, value in zip(slot_keys, (first[2], second[2], p.video_path, p.audio_path, p.asr_text)):
                    cols[key].append(value)
            group += 1
    slot_batch = {key: cols[key] for key in slot_keys}
    meta = {key: torch.tensor(cols[key], device=device, dtype=torch.long if i < 4 else torch.float)
            for i, key in enumerate(meta_keys)}
    return slot_batch, meta


def build_rollout(policy, pairs, cfg, device="cpu") -> RolloutData:
    orientations = [0, 1] if cfg.rollout.balanced_swap else [0]
    slot_batch, meta = make_slot_batch(pairs, orientations, cfg.rollout.group_k, device=device)
    rollout = policy.generate(slot_batch)
    decision, valid = policy.parse_decision(rollout)
    info = compute_rewards(decision, valid, meta["pair_id"], meta["orientation"], meta["gold"],
                           meta["len_slot1"], meta["len_slot2"], meta["bt_slot1"], meta["bt_slot2"], cfg.reward)
    return RolloutData(rollout=rollout, reward=info.reward, group_id=meta["group_id"], info=info)
