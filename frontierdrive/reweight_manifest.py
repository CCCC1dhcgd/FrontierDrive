#!/usr/bin/env python
"""Copy a TBAT manifest while changing selected/replay sampling mass."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def is_selected(row: Dict[str, str]) -> bool:
    return row.get("selected", "") in {"1", "True", "true"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reweight a TBAT train_manifest.csv.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-mass", type=float, required=True)
    args = parser.parse_args()

    if not 0.0 < args.boundary_mass < 1.0:
        raise ValueError(f"--boundary-mass must be in (0, 1), got {args.boundary_mass}")

    with args.input.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{args.input} has no header")
        required = {"token", "selected", "train_weight"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise KeyError(f"{args.input} is missing columns: {sorted(missing)}")
        fieldnames = list(reader.fieldnames)
        rows: List[Dict[str, str]] = [dict(row) for row in reader]

    selected_rows = [row for row in rows if is_selected(row)]
    replay_rows = [row for row in rows if not is_selected(row)]
    if not selected_rows:
        raise RuntimeError(f"{args.input} has no selected rows")
    if not replay_rows:
        raise RuntimeError(f"{args.input} has no replay rows")

    selected_sum = sum(float(row["train_weight"]) for row in selected_rows)
    replay_sum = sum(float(row["train_weight"]) for row in replay_rows)
    if selected_sum <= 0.0:
        raise RuntimeError(f"{args.input} selected rows have zero total train_weight")
    if replay_sum <= 0.0:
        raise RuntimeError(f"{args.input} replay rows have zero total train_weight")

    replay_mass = 1.0 - args.boundary_mass
    for row in selected_rows:
        row["train_weight"] = f"{args.boundary_mass * float(row['train_weight']) / selected_sum:.12f}"
    for row in replay_rows:
        row["train_weight"] = f"{replay_mass * float(row['train_weight']) / replay_sum:.12f}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Reweighted TBAT manifest: "
        f"input={args.input} output={args.output} "
        f"boundary_mass={args.boundary_mass:.6f} replay_mass={replay_mass:.6f}"
    )


if __name__ == "__main__":
    main()
