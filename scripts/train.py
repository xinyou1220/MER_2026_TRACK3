"""Train one model and keep the checkpoint with the best swap-averaged WAF-2 on --eval."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sympo_mer.config import Config
from sympo_mer.data import load_pairs
from sympo_mer.omni_policy import OmniPolicy
from sympo_mer.trainer import Trainer

REWARD_ARGS = ("w_swap", "w_bt", "w_len", "swap_consistency_mode", "shaping_mode", "format_mode")
TRAIN_ARGS = ("steps", "eval_every", "log_every", "eval_batch_size", "topk_checkpoints", "seed", "lr", "output_dir")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--video_min_pixels", type=int, default=3136)
    parser.add_argument("--video_max_pixels", type=int, default=50176)
    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=1, help="Conversations per forward pass.")


def model_kwargs(args) -> dict:
    return {key: getattr(args, key) for key in
            ("device", "video_fps", "video_min_pixels", "video_max_pixels", "video_max_frames", "chunk_size")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output_dir", required=True)
    add_model_args(parser)
    parser.add_argument("--estimator", choices=["reinforce_baseline", "sft"], default="reinforce_baseline")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--topk_checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--prompts_per_batch", type=int, default=4)
    parser.add_argument("--group_k", type=int, default=4)
    parser.add_argument("--single_order_train", dest="balanced_swap", action="store_false")
    parser.add_argument("--normalization_mode", choices=["global", "per_group"], default="global")
    parser.add_argument("--kl_loss_coef", type=float, default=0.01)
    parser.add_argument("--w_swap", type=float, default=0.5)
    parser.add_argument("--w_bt", type=float, default=0.3)
    parser.add_argument("--w_len", type=float, default=0.2)
    parser.add_argument("--swap_consistency_mode", choices=["majority", "continuous"], default="majority")
    parser.add_argument("--shaping_mode", choices=["bounded", "unbounded"], default="bounded")
    parser.add_argument("--format_mode", choices=["invalid_floor", "valid_bonus", "additive"], default="invalid_floor")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = Config(device=args.device)
    cfg.adv.estimator = args.estimator
    cfg.adv.normalization_mode = args.normalization_mode
    cfg.kl_loss_coef = args.kl_loss_coef
    cfg.rollout.prompts_per_batch = args.prompts_per_batch
    cfg.rollout.group_k = 1 if args.estimator == "sft" else args.group_k
    cfg.rollout.balanced_swap = args.balanced_swap
    for key in REWARD_ARGS:
        setattr(cfg.reward, key, getattr(args, key))
    for key in TRAIN_ARGS:
        setattr(cfg.train, key, getattr(args, key))
    cfg.run_metadata = {"model": args.model, "train": args.train, "eval": args.eval}

    train_pairs, eval_pairs = load_pairs(args.train), load_pairs(args.eval)
    policy = OmniPolicy(model_name=args.model, **model_kwargs(args))
    best = Trainer(policy, train_pairs, eval_pairs, cfg).train()
    print(f"best WAF2_swap = {best:.4f}")


if __name__ == "__main__":
    main()
