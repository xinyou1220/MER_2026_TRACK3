"""Evaluate a checkpoint (or the zero-shot base model) on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sympo_mer.data import load_pairs
from sympo_mer.omni_policy import OmniPolicy
from sympo_mer.scoring import decide, evaluation_metrics, score_pairs
from train import add_model_args, model_kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="splits/test.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--inference_mode", choices=["antisym", "single_order"], default="antisym")
    parser.add_argument("--output_dir", required=True)
    add_model_args(parser)
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    policy = OmniPolicy(model_name=args.model, lora=args.checkpoint is not None, do_sample=False,
                        checkpoint_path=args.checkpoint, gradient_checkpointing=False, **model_kwargs(args))
    policy.model.eval()
    records = score_pairs(policy, pairs, args.device)

    for row in records:
        row["prediction"], row["margin"] = decide(row, args.inference_mode)
    metrics = evaluation_metrics(records, [row["prediction"] for row in records])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (out / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
