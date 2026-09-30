"""Run one paper experiment end to end: train (if needed), then evaluate on the held-out test split.

    python scripts/run_experiment.py --experiment sympo --seed 42
    python scripts/run_experiment.py --list

Checkpoints go to runs/<experiment>/seed<seed>/, metrics to results/<experiment>/seed<seed>/.
Arguments not recognized here (for example --steps) are passed on to scripts/train.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sympo_mer.experiments import EXPERIMENTS, TRAIN_OVERRIDE_KEYS


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--list", action="store_true")
    args, train_extra = parser.parse_known_args()
    if args.list:
        print(json.dumps(EXPERIMENTS, indent=2))
        return
    if not args.experiment:
        parser.error("--experiment is required")

    overrides = EXPERIMENTS[args.experiment]
    splits = Path(args.splits)
    run_dir = Path("runs") / args.experiment / f"seed{args.seed}"
    result_dir = Path("results") / args.experiment / f"seed{args.seed}"

    evaluate = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--data", str(splits / "test.csv"),
                "--output_dir", str(result_dir), "--model", args.model,
                "--inference_mode", overrides.get("inference_mode", "antisym")]

    if overrides.get("training") != "none":
        selection = "gold_selection.csv" if overrides.get("selection_split") == "gold" else "validation.csv"
        train = [sys.executable, str(ROOT / "scripts" / "train.py"), "--train", str(splits / "train.jsonl"),
                 "--eval", str(splits / selection), "--output_dir", str(run_dir),
                 "--seed", str(args.seed), "--model", args.model]
        for key, value in overrides.items():
            if key == "balanced_swap":
                if not value:
                    train.append("--single_order_train")
            elif key in TRAIN_OVERRIDE_KEYS:
                train += [f"--{key}", str(value)]
        run(train + train_extra)
        evaluate += ["--checkpoint", str(run_dir / "best_trainable.pt")]

    run(evaluate)


if __name__ == "__main__":
    main()
