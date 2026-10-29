#!/usr/bin/env python
"""Read one shell-friendly value from a TBAT training proposal JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a value from agent/training_proposal.json.")
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument(
        "--field",
        choices=[
            "max_epochs",
            "train_manifest",
            "rotation_fraction",
            "epoch_size",
            "sampler_seed",
            "source_checkpoint",
        ],
        required=True,
    )
    args = parser.parse_args()

    with args.proposal.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    if payload.get("schema_version") != "training_proposal.v1":
        raise ValueError(f"{args.proposal} is not a training_proposal.v1 file")

    inputs = payload.get("training_script_inputs", {})
    mapping = {
        "max_epochs": payload.get("max_epochs"),
        "train_manifest": inputs.get("TBAT_MANIFEST") or payload.get("selected_scene_ids_or_sampling_file", {}).get("train_manifest"),
        "rotation_fraction": inputs.get("TBAT_ROTATION_FRACTION"),
        "epoch_size": inputs.get("TBAT_EPOCH_SIZE"),
        "sampler_seed": inputs.get("TBAT_SAMPLER_SEED"),
        "source_checkpoint": payload.get("source_checkpoint"),
    }
    value = mapping[args.field]
    if value is None or str(value) == "":
        raise KeyError(f"{args.field} is missing from {args.proposal}")
    print(value)


if __name__ == "__main__":
    main()
