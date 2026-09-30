"""Step 1: split the weakly labeled EmoPrefer-V2 list and build the combined training manifest.

Writes, next to the weak CSV, <name>_train.csv (1/3, used for training) and <name>_val.csv
(2/3, later split into validation and held-out test), stratified by preference with seed 42.
The combined training manifest holds every gold row plus the weak training rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(rows, train_ratio: float, seed: int):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[str(row.get("preference", "")).strip().lower()].append((index, row))
    train, val = [], []
    for label in sorted(buckets):
        bucket = buckets[label]
        rng.shuffle(bucket)
        split_at = round(len(bucket) * train_ratio)
        if len(bucket) > 1:
            split_at = min(max(split_at, 1), len(bucket) - 1)
        train.extend(bucket[:split_at])
        val.extend(bucket[split_at:])
    train.sort(key=lambda item: item[0])
    val.sort(key=lambda item: item[0])
    return [row for _, row in train], [row for _, row in val]


def with_media(source: Path, rows, out_dir: Path):
    source_dir = source.resolve().parent
    for row in rows:
        name = str(row.get("name", "")).strip()
        enriched = dict(row)
        if name:
            for key, folder, suffix in (("video_path", "video", ".mp4"), ("audio_path", "audio", ".wav"),
                                        ("transcript_path", "transcripts", ".json")):
                enriched.setdefault(key, os.path.relpath(source_dir / folder / f"{name}{suffix}", out_dir).replace("\\", "/"))
        yield enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/track3_emoprefer.csv")
    parser.add_argument("--weak", default="data_weak/track3_emopreferv2.csv")
    parser.add_argument("--weak_train_ratio", type=float, default=1 / 3)
    parser.add_argument("--combined_train_out", default="data_mixed/train_gold_weak.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gold, weak = Path(args.gold), Path(args.weak)
    fieldnames, weak_rows = read_csv(weak)
    weak_train, weak_val = stratified_split(weak_rows, args.weak_train_ratio, args.seed)
    stem = weak.with_suffix("").name
    write_csv(weak.with_name(f"{stem}_train.csv"), fieldnames, weak_train)
    write_csv(weak.with_name(f"{stem}_val.csv"), fieldnames, weak_val)

    out = Path(args.combined_train_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out.resolve().parent
    _, gold_rows = read_csv(gold)
    with out.open("w", encoding="utf-8") as f:
        for source, rows in ((gold, gold_rows), (weak, weak_train)):
            for row in with_media(source, rows, out_dir):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"weak: {len(weak_train)} train / {len(weak_val)} val; combined train: {len(gold_rows) + len(weak_train)} -> {out}")


if __name__ == "__main__":
    main()
