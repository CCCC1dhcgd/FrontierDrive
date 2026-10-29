#!/usr/bin/env python
"""TBAT training sampler."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Sampler


class TBATSampler(Sampler[int]):
    """Weighted TBAT sampler with per-epoch rotation of selected boundary scenes."""

    def __init__(
        self,
        tokens: List[str],
        manifest_path: str,
        rotation_fraction: float = 0.85,
        epoch_size: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if not 0.0 < rotation_fraction <= 1.0:
            raise ValueError(f"rotation_fraction must be in (0, 1], got {rotation_fraction}")
        if epoch_size is not None and epoch_size <= 0:
            raise ValueError(f"epoch_size must be positive when provided, got {epoch_size}")

        weights_by_token = {}
        selected_by_token = {}
        with Path(manifest_path).open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            required = ["token", "selected", "train_weight"]
            missing = [name for name in required if reader.fieldnames is None or name not in reader.fieldnames]
            if missing:
                raise KeyError(f"{manifest_path} is missing columns: {missing}")
            for row in reader:
                weights_by_token[row["token"]] = float(row["train_weight"])
                selected_by_token[row["token"]] = int(row["selected"]) == 1

        missing_tokens = [token for token in tokens if token not in weights_by_token]
        if missing_tokens:
            examples = ", ".join(missing_tokens[:5])
            raise KeyError(f"TBAT manifest is missing {len(missing_tokens)} train tokens; examples: {examples}")

        weights = [weights_by_token[token] for token in tokens]
        if any(weight < 0.0 for weight in weights):
            raise ValueError("TBAT manifest contains negative train_weight values")

        self._weights = torch.as_tensor(weights, dtype=torch.double)
        self._selected_mask = torch.as_tensor(
            [selected_by_token[token] for token in tokens],
            dtype=torch.bool,
        )
        self._selected_indices = torch.as_tensor(
            [index for index, token in enumerate(tokens) if selected_by_token[token]],
            dtype=torch.long,
        )
        self._rotation_fraction = rotation_fraction
        self._seed = int(seed)
        self._epoch = 0

        if self._selected_indices.numel() == 0:
            raise ValueError("TBAT manifest does not mark any training token as selected")
        if not torch.any(~self._selected_mask):
            raise ValueError("TBAT manifest does not contain replay tokens")
        selected_count = int(self._selected_indices.numel())
        self._num_samples = int(epoch_size) if epoch_size is not None else len(tokens)

    def __iter__(self):
        generator = torch.Generator()
        current_epoch = self._epoch
        generator.manual_seed(self._seed + current_epoch)
        self._epoch += 1

        weights = torch.zeros_like(self._weights)
        selected_count = int(self._selected_indices.numel())
        active_count = max(1, int(round(self._rotation_fraction * selected_count)))
        active_count = min(active_count, selected_count)

        selected_order = torch.randperm(selected_count, generator=generator)
        active_selected_indices = self._selected_indices[selected_order[:active_count]]
        replay_mask = ~self._selected_mask

        selected_mass = torch.sum(self._weights[self._selected_mask])
        replay_mass = torch.sum(self._weights[replay_mask])
        active_selected_weight = self._weights[active_selected_indices]
        replay_weight = self._weights[replay_mask]

        if current_epoch == 0:
            print(
                "[TBATSampler] "
                f"selected={selected_count}, replay={int(torch.sum(replay_mask))}, "
                f"active_selected={active_count}, "
                f"epoch_draws={self._num_samples}, fixed_budget=1, "
                f"boundary_mass={float(selected_mass):.4f}, replay_mass={float(replay_mass):.4f}, "
                f"expected_draws_per_selected={self._num_samples * float(selected_mass) / max(selected_count, 1):.1f}"
            )

        weights[active_selected_indices] = selected_mass * active_selected_weight / torch.sum(active_selected_weight)
        weights[replay_mask] = replay_mass * replay_weight / torch.sum(replay_weight)

        if float(torch.sum(weights)) <= 0.0:
            raise RuntimeError("TBAT sampler has zero total weight after token rotation")

        sampled_indices = torch.multinomial(weights, self._num_samples, replacement=True, generator=generator)
        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        return self._num_samples


def build_curriculum_sampler(
    train_data,
    manifest_path: str,
    rotation_fraction: float,
    epoch_size: Optional[int],
    seed: int,
) -> TBATSampler:
    """Build a TBAT sampler from a train manifest."""
    return TBATSampler(
        tokens=train_data.tokens,
        manifest_path=manifest_path,
        rotation_fraction=rotation_fraction,
        epoch_size=epoch_size,
        seed=seed,
    )
