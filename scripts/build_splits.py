"""Step 2: build the binary train / validation / held-out test splits and the OOF-BT train manifest.

- validation / test: the weak validation list without `same` rows, split 50/50 stratified (seed 42).
  Validation is used only for checkpoint selection; test only for the final internal evaluation.
- Sample IDs that appear in both the gold list and the weak validation list are removed everywhere.
- train: gold + weak-train rows, binary only, without any validation/test ID, deduplicated,
  then annotated with 5-fold out-of-fold BT utilities.
- gold_selection: the gold rows of train, used only by the gold-selection ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sympo_mer.bt import oof_utilities

BINARY = {"a1", "a2"}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def name(row) -> str:
    return str(row.get("name", "")).strip()


def label(row) -> str:
    return str(row.get("preference", "")).strip().lower()


def relative(path: Path, out_dir: Path) -> str:
    try:
        return os.path.relpath(path, out_dir).replace("\\", "/")
    except ValueError:
        return path.resolve().as_posix()


def attach_media(rows, source_dir: Path, out_dir: Path):
    out = []
    for row in rows:
        row = dict(row)
        for key, folder, suffix in (("video_path", "video", ".mp4"), ("audio_path", "audio", ".wav"),
                                    ("transcript_path", "transcripts", ".json")):
            row[key] = relative(source_dir / folder / f"{name(row)}{suffix}", out_dir)
        out.append(row)
    return out


def rebase_media(rows, source_dir: Path, out_dir: Path):
    out = []
    for row in rows:
        row = dict(row)
        for key in ("video_path", "audio_path", "transcript_path"):
            raw = str(row.get(key, "")).strip()
            if raw:
                path = Path(raw)
                row[key] = relative(path if path.is_absolute() else (source_dir / path).resolve(), out_dir)
        out.append(row)
    return out


def stratified_half(rows, seed: int):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[label(row)].append((index, row))
    first, second = [], []
    for key in sorted(buckets):
        bucket = buckets[key]
        rng.shuffle(bucket)
        split_at = round(len(bucket) * 0.5)
        if len(bucket) > 1:
            split_at = min(max(split_at, 1), len(bucket) - 1)
        first.extend(bucket[:split_at])
        second.extend(bucket[split_at:])
    first.sort(key=lambda item: item[0])
    second.sort(key=lambda item: item[0])
    return [row for _, row in first], [row for _, row in second]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/track3_emoprefer.csv")
    parser.add_argument("--weak_val", default="data_weak/track3_emopreferv2_val.csv")
    parser.add_argument("--train_source", default="data_mixed/train_gold_weak.jsonl")
    parser.add_argument("--out_dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gold_path, weak_val_path, train_source = Path(args.gold), Path(args.weak_val), Path(args.train_source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_abs = out_dir.resolve()

    val_fields, weak_val_rows = read_csv(weak_val_path)
    gold_fields, gold_rows = read_csv(gold_path)
    binary_val = [row for row in weak_val_rows if label(row) in BINARY]
    binary_gold = [row for row in gold_rows if label(row) in BINARY]
    overlap = {name(r) for r in binary_gold} & {name(r) for r in binary_val}
    binary_val = [row for row in binary_val if name(row) not in overlap]
    validation, test = stratified_half(binary_val, args.seed)
    held_out_names = {name(row) for row in binary_val}

    train, seen = [], set()
    for row in read_jsonl(train_source):
        n = name(row)
        if label(row) not in BINARY or n in overlap or n in held_out_names or n in seen:
            continue
        seen.add(n)
        row = dict(row)
        row.pop("bt_a", None)
        row.pop("bt_b", None)
        train.append(row)
    gold_selection = [row for row in binary_gold if name(row) in seen and name(row) not in overlap]

    media = ["video_path", "audio_path", "transcript_path"]
    weak_dir, gold_dir = weak_val_path.resolve().parent, gold_path.resolve().parent
    write_csv(out_dir / "validation.csv", val_fields + media, attach_media(validation, weak_dir, out_abs))
    write_csv(out_dir / "test.csv", val_fields + media, attach_media(test, weak_dir, out_abs))
    write_csv(out_dir / "gold_selection.csv", gold_fields + media, attach_media(gold_selection, gold_dir, out_abs))

    train = rebase_media(train, train_source.resolve().parent, out_abs)
    bt_a, bt_b = oof_utilities(train, folds=5, seed=args.seed)
    for row, u_a, u_b in zip(train, bt_a, bt_b):
        row["bt_a"] = f"{u_a:.8f}"
        row["bt_b"] = f"{u_b:.8f}"
    write_jsonl(out_dir / "train.jsonl", train)
    print(f"train {len(train)}, validation {len(validation)}, test {len(test)}, gold_selection {len(gold_selection)}")

if __name__ == "__main__":
    main()
