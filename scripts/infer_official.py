"""Predict the official test set and write a name,preference submission file.

Use --one_shot_source instead of --checkpoint for the one-shot prompting baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sympo_mer.data import load_pairs
from sympo_mer.omni_policy import OmniPolicy
from sympo_mer.scoring import decide, score_pairs
from train import add_model_args, model_kwargs


def one_shot_example(source: str, exclude: set[str], seed: int) -> dict:
    """The labeled gold pair with the smallest sha256(seed:name) that is not in the test set."""
    candidates = [p for p in load_pairs(source) if p.gold in (0, 1) and p.sample_id not in exclude
                  and p.cand0_text.strip() and p.cand1_text.strip() and p.asr_text.strip()]
    pair = min(candidates, key=lambda p: hashlib.sha256(f"{seed}:{p.sample_id}".encode("utf-8")).digest())
    return {"asr_text": pair.asr_text, "candidate_a": pair.cand0_text, "candidate_b": pair.cand1_text,
            "answer": "A" if pair.gold == 0 else "B"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data_test/track3_candidate.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--one_shot_source", default=None)
    parser.add_argument("--inference_mode", choices=["antisym", "single_order"], default="antisym")
    parser.add_argument("--out", default="outputs/submission.csv")
    add_model_args(parser)
    args = parser.parse_args()

    pairs = load_pairs(args.test)
    example = one_shot_example(args.one_shot_source, {p.sample_id for p in pairs}, seed=42) if args.one_shot_source else None
    policy = OmniPolicy(model_name=args.model, lora=args.checkpoint is not None, do_sample=False,
                        checkpoint_path=args.checkpoint, gradient_checkpointing=False,
                        one_shot_example=example, **model_kwargs(args))
    policy.model.eval()
    records = score_pairs(policy, pairs, args.device, both_orders=args.inference_mode == "antisym")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "preference"])
        for row in records:
            prediction, _ = decide(row, args.inference_mode)
            writer.writerow([row["name"], "a1" if prediction == 0 else "a2"])
    print(f"wrote {len(records)} predictions to {out}")


if __name__ == "__main__":
    main()
