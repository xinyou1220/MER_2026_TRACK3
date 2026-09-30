"""Group-mean baseline with global normalization (REINFORCE++-baseline, paper Eq. 13)."""
import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def masked_std(x: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    var = ((x - mean).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
    return var.clamp_min(0.0).sqrt()


def global_normalize(adv: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    m = masked_mean(adv, mask)
    s = masked_std(adv, mask, m)
    return ((adv - m) / (s + eps)) * mask


def per_group_normalize(scores: torch.Tensor, group_id: torch.Tensor, eps: float) -> tuple[torch.Tensor, float]:
    """GRPO-style per-group z-score (ablation). Groups with zero spread get advantage 0."""
    normalized = torch.zeros_like(scores)
    groups = torch.unique(group_id)
    degenerate = 0
    for gid in groups:
        mask = group_id == gid
        values = scores[mask]
        mean = values.mean()
        std = (values - mean).pow(2).mean().clamp_min(0.0).sqrt()
        if std <= eps:
            degenerate += 1
            continue
        normalized[mask] = (values - mean) / (std + eps)
    return normalized, degenerate / max(int(groups.numel()), 1)


def advantage_reinforce_baseline(
    scores: torch.Tensor,
    group_id: torch.Tensor,
    action_mask: torch.Tensor,
    eps: float,
    normalization_mode: str = "global",
):
    """Returns (token advantages, centered rewards, degenerate-group ratio)."""
    centered = scores.clone()
    for gid in torch.unique(group_id):
        m = group_id == gid
        centered[m] = scores[m] - scores[m].mean()
    degenerate_ratio = 0.0
    if normalization_mode == "per_group":
        normalized, degenerate_ratio = per_group_normalize(centered, group_id, eps)
        adv = normalized.unsqueeze(1) * action_mask
    elif normalization_mode == "global":
        adv = global_normalize(centered.unsqueeze(1) * action_mask, action_mask, eps)
    else:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode!r}")
    return adv, centered, degenerate_ratio
