"""Experiments reported in the paper. Each entry overrides the SymPO defaults.

SymPO defaults: balanced swap, 4 pairs x 2 orders x K=4 = 32 responses per step, bounded reward
(w_c, w_s, w_b, w_l) = (1, 0.5, 0.3, 0.2), invalid floor, global normalization, k2 KL (beta=0.01),
selection on the distribution-aligned validation half, antisymmetrized inference.
"""

EXPERIMENTS = {
    # Main method
    "sympo": {},
    # Baselines (Table 1)
    "zero_shot_single": {"training": "none", "inference_mode": "single_order"},
    "zero_shot_swap": {"training": "none"},
    "sft_plain": {"estimator": "sft", "balanced_swap": False, "kl_loss_coef": 0.0, "inference_mode": "single_order"},
    "sft_swap": {"estimator": "sft", "kl_loss_coef": 0.0},
    "naive_rl": {
        "balanced_swap": False, "prompts_per_batch": 8,
        "w_swap": 0.0, "w_bt": 0.0, "w_len": 0.0, "shaping_mode": "unbounded", "format_mode": "valid_bonus",
    },
    "rl_swap_correctness": {
        "w_swap": 0.0, "w_bt": 0.0, "w_len": 0.0, "shaping_mode": "unbounded", "format_mode": "valid_bonus",
    },
    "sympo_additive": {"format_mode": "additive"},
    # Ablations (Table 2)
    "no_balanced_swap": {"balanced_swap": False},
    "per_group_norm": {"normalization_mode": "per_group"},
    "no_consistency_reward": {"w_swap": 0.0},
    "unbounded_aux": {"shaping_mode": "unbounded"},
    "continuous_consistency": {"swap_consistency_mode": "continuous"},
    "no_bt": {"w_bt": 0.0},
    "no_length": {"w_len": 0.0},
    "gold_selection": {"selection_split": "gold"},
}

TRAIN_OVERRIDE_KEYS = {
    "estimator", "balanced_swap", "prompts_per_batch", "group_k", "kl_loss_coef", "normalization_mode",
    "w_swap", "w_bt", "w_len", "swap_consistency_mode", "shaping_mode", "format_mode",
}
