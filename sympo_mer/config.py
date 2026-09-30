from dataclasses import dataclass, field


@dataclass
class AdvantageConfig:
    # "reinforce_baseline": sampled REINFORCE++ with a group-mean baseline (SymPO, naive RL).
    # "sft": cross-entropy on the constrained A/B probabilities (SFT baselines).
    estimator: str = "reinforce_baseline"
    normalization_mode: str = "global"  # global | per_group
    norm_eps: float = 1e-6


@dataclass
class RewardConfig:
    w_correct: float = 1.0
    w_swap: float = 0.5
    w_bt: float = 0.3
    w_len: float = 0.2
    w_format: float = 0.1
    swap_consistency_mode: str = "majority"  # majority | continuous
    shaping_mode: str = "bounded"  # bounded | unbounded
    format_mode: str = "invalid_floor"  # invalid_floor | valid_bonus | additive
    tau_labeled: float = 0.2
    tau_unlabeled: float = 0.5
    invalid_margin: float = 0.5


@dataclass
class RolloutConfig:
    prompts_per_batch: int = 4
    balanced_swap: bool = True
    group_k: int = 4


@dataclass
class TrainConfig:
    steps: int = 100
    clip_eps: float = 0.2
    lr: float = 5e-5
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 10
    eval_every: int = 25
    eval_batch_size: int = 1
    topk_checkpoints: int = 3
    output_dir: str = "runs/sympo"


@dataclass
class Config:
    adv: AdvantageConfig = field(default_factory=AdvantageConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    kl_loss_coef: float = 0.01
    device: str = "cuda"
    run_metadata: dict = field(default_factory=dict)
