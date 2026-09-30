"""Collect results/<experiment>/seed*/test_metrics.json into a mean +/- s.d. table."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

COLUMNS = [("waf2", "WAF-2", 100), ("swap_consistency", "Swap cons.", 100),
           ("position_bias", "Position bias", 100), ("positional_leakage", "Leakage", 1)]


def fmt(values: list[float], scale: float) -> str:
    values = [v * scale for v in values]
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.fmean(values):.2f} ± {statistics.stdev(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/summary.md")
    args = parser.parse_args()

    lines = ["| Experiment | Seeds | " + " | ".join(label for _, label, _ in COLUMNS) + " |",
             "|---|---|" + "---|" * len(COLUMNS)]
    for exp_dir in sorted(p for p in Path(args.results).iterdir() if p.is_dir()):
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(exp_dir.glob("seed*/test_metrics.json"))]
        if runs:
            cells = [fmt([r[key] for r in runs], scale) for key, _, scale in COLUMNS]
            lines.append(f"| {exp_dir.name} | {len(runs)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    Path(args.out).write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
