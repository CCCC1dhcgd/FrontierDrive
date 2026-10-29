#!/usr/bin/env python
"""
Summarize NAVSIM evaluation CSV scores.

The report is intentionally small: it records the score distribution and the
sub-metric distributions used by the TBAT boundary filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


SUBMETRIC_COLUMNS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
]


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower)


def summarize_values(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": ordered[0],
        "p20": percentile(ordered, 0.20),
        "p50": percentile(ordered, 0.50),
        "p80": percentile(ordered, 0.80),
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an existing NAVSIM evaluation CSV for TBAT training.")
    parser.add_argument("--eval-csv", type=Path, required=True, help="Evaluation CSV from run_pdm_score.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for eval_summary.json/txt.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores: List[float] = []
    submetrics: Dict[str, List[float]] = {name: [] for name in SUBMETRIC_COLUMNS}
    total_rows = 0
    valid_rows = 0

    with args.eval_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["token", "valid", "score"] + SUBMETRIC_COLUMNS
        missing = [name for name in required if reader.fieldnames is None or name not in reader.fieldnames]
        if missing:
            raise KeyError(f"{args.eval_csv} is missing columns: {missing}")

        for row in reader:
            if row["token"] == "average_all_frames":
                continue
            total_rows += 1
            if row["valid"] != "True":
                continue
            valid_rows += 1
            scores.append(float(row["score"]))
            for name in SUBMETRIC_COLUMNS:
                submetrics[name].append(float(row[name]))

    if not scores:
        raise RuntimeError(f"{args.eval_csv} contains no valid scene scores")

    summary = {
        "eval_csv": str(args.eval_csv),
        "total_scene_rows": total_rows,
        "valid_scene_rows": valid_rows,
        "valid_fraction": valid_rows / total_rows if total_rows else 0.0,
        "score": summarize_values(scores),
        "submetrics": {name: summarize_values(values) for name, values in submetrics.items()},
    }

    lines = ["NAVSIM evaluation summary"]
    lines.append(f"eval_csv: {args.eval_csv}")
    lines.append(f"valid scenes: {valid_rows}/{total_rows} ({summary['valid_fraction']:.4f})")
    score = summary["score"]
    lines.append(
        "score: "
        f"mean={score['mean']:.6f} std={score['std']:.6f} "
        f"p20={score['p20']:.6f} p50={score['p50']:.6f} p80={score['p80']:.6f}"
    )
    lines.append("")
    lines.append("submetrics:")
    for name in SUBMETRIC_COLUMNS:
        stats = summary["submetrics"][name]
        lines.append(
            f"  {name:<32} mean={stats['mean']:.6f} "
            f"p20={stats['p20']:.6f} p50={stats['p50']:.6f} p80={stats['p80']:.6f}"
        )

    report_text = "\n".join(lines)
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "eval_summary.txt").write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
