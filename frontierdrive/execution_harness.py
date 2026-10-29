#!/usr/bin/env python
"""Execution-harness boundary for verified FrontierDrive curricula.

This module is deliberately outside the agent controller.  It translates a
verified plan into the legacy manifest/sampler/training interface and records
the configuration that the platform actually applies.  Scheduler-specific job
submission remains in the existing stage shell scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    from .learning_memory import CrossStageLearningMemory, utc_now
    from .schemas import AppliedCurriculum, CandidateCurriculumPlan, ExecutionResult, VerifiedCurriculumPlan
except ImportError:  # Direct script/import from the train_boundary directory.
    from learning_memory import CrossStageLearningMemory, utc_now
    from schemas import AppliedCurriculum, CandidateCurriculumPlan, ExecutionResult, VerifiedCurriculumPlan


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


class CurriculumExecutionHarness:
    """Convert verified decisions to platform inputs and persist observations."""

    def materialize_verified_plan(self, verified: VerifiedCurriculumPlan, agent_dir: Path) -> Dict[str, Path]:
        candidate = verified.candidate_plan
        legacy = candidate.to_legacy_training_proposal()
        curriculum_manifest = agent_dir.parent / "retrieval" / "curriculum_manifest.csv"
        self._write_curriculum_manifest(candidate, curriculum_manifest)
        legacy["selected_scene_ids_or_sampling_file"]["source_retrieval_manifest"] = candidate.support_manifest
        legacy["selected_scene_ids_or_sampling_file"]["train_manifest"] = str(curriculum_manifest)
        legacy["training_script_inputs"]["TBAT_MANIFEST"] = str(curriculum_manifest)
        paths = {
            "candidate": agent_dir / "candidate_curriculum_plan.json",
            "verified": agent_dir / "verified_curriculum_plan.json",
            "curriculum": agent_dir / "curriculum_plan.json",
            "legacy_proposal": agent_dir / "training_proposal.json",
            "legacy_yaml": agent_dir / "training_proposal.yaml",
            "legacy_env": agent_dir / "training_proposal.env",
            "curriculum_manifest": curriculum_manifest,
        }
        _write_json(paths["candidate"], candidate.to_dict())
        _write_json(paths["verified"], verified.to_dict())
        _write_json(paths["curriculum"], verified.to_dict())
        _write_json(paths["legacy_proposal"], legacy)
        # JSON is valid YAML 1.2 and avoids adding an HPC dependency.
        _write_json(paths["legacy_yaml"], legacy)
        env_lines = ["# Harness-generated inputs. Pass as sbatch --export values; do not source globally."]
        for key, value in legacy["training_script_inputs"].items():
            escaped = str(value).replace("'", "'\"'\"'")
            env_lines.append(f"{key}='{escaped}'")
        paths["legacy_env"].write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        return paths

    def _write_curriculum_manifest(self, candidate: CandidateCurriculumPlan, output_path: Path) -> None:
        """Apply verified region allocation without changing retrieved support."""
        source_path = Path(candidate.support_manifest)
        with source_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            required = {"token", "selected", "nearest_cluster", "train_weight"}
            if not required.issubset(fieldnames):
                raise ValueError(f"{source_path} is missing curriculum manifest columns: {sorted(required - set(fieldnames))}")
            rows = list(reader)

        allocation_by_region = {item.region_id: item.weight for item in candidate.region_allocations}
        selected_by_region: Dict[int, list[int]] = {}
        replay_indices: list[int] = []
        base_weights: list[float] = []
        for index, row in enumerate(rows):
            weight = float(row["train_weight"])
            if weight < 0.0:
                raise ValueError("Source retrieval manifest contains negative train_weight")
            base_weights.append(weight)
            if row["selected"] == "1":
                region_id = int(row["nearest_cluster"])
                selected_by_region.setdefault(region_id, []).append(index)
            else:
                replay_indices.append(index)

        if set(selected_by_region) != set(allocation_by_region):
            raise ValueError(
                "Verified region allocation does not match retrieved support regions: "
                f"plan={sorted(allocation_by_region)}, manifest={sorted(selected_by_region)}"
            )
        if not replay_indices:
            raise ValueError("Curriculum manifest requires a non-empty Retention Pool")

        new_weights = [0.0] * len(rows)
        for region_id, indices in selected_by_region.items():
            region_base = sum(base_weights[index] for index in indices)
            if region_base <= 0.0:
                # The deterministic proximity weights are the preferred
                # within-region distribution; uniform is a fail-safe only.
                for index in indices:
                    new_weights[index] = candidate.frontier_allocation * allocation_by_region[region_id] / len(indices)
            else:
                region_mass = candidate.frontier_allocation * allocation_by_region[region_id]
                for index in indices:
                    new_weights[index] = region_mass * base_weights[index] / region_base

        replay_base = sum(base_weights[index] for index in replay_indices)
        if replay_base <= 0.0:
            raise ValueError("Retention Pool has zero source sampling mass")
        for index in replay_indices:
            new_weights[index] = candidate.retention_replay_allocation * base_weights[index] / replay_base

        if abs(sum(new_weights) - 1.0) > 1e-8:
            raise ValueError("Harness-generated curriculum weights do not sum to one")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, weight in zip(rows, new_weights):
                updated = dict(row)
                updated["train_weight"] = f"{weight:.12f}"
                writer.writerow(updated)

    def record_applied_curriculum(
        self,
        *,
        memory: CrossStageLearningMemory,
        verified: VerifiedCurriculumPlan,
        verified_plan_path: Path,
        actual_manifest_path: Path,
        actual_training_configuration: Mapping[str, Any],
        status: str,
        job_id: Optional[str] = None,
    ) -> AppliedCurriculum:
        candidate = verified.candidate_plan
        applied = AppliedCurriculum(
            stage=candidate.stage,
            verified_plan_path=str(verified_plan_path),
            manifest_path=str(actual_manifest_path),
            sampler_configuration={
                "frontier_allocation": candidate.frontier_allocation,
                "retention_replay_allocation": candidate.retention_replay_allocation,
                "rotation_fraction": candidate.budget.rotation_fraction,
                "epoch_size": candidate.budget.epoch_size,
                "sampler_seed": candidate.budget.sampler_seed,
            },
            training_configuration=dict(actual_training_configuration),
            status=status,
            job_id=job_id,
            applied_at=utc_now(),
        )
        memory.record_applied(candidate.stage, applied.to_dict())
        return applied

    def record_execution_result(
        self,
        *,
        memory: CrossStageLearningMemory,
        stage: int,
        status: str,
        applied_configuration: Mapping[str, Any],
        metrics: Mapping[str, Any],
        artifacts: Mapping[str, str],
        errors: list[str],
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> ExecutionResult:
        result = ExecutionResult(
            stage=stage,
            status=status,
            applied_configuration=dict(applied_configuration),
            metrics=dict(metrics),
            artifacts=dict(artifacts),
            errors=list(errors),
            started_at=started_at,
            completed_at=completed_at or utc_now(),
        )
        memory.record_execution(stage, result.to_dict())
        if errors:
            memory.record_error(stage=stage, source="CurriculumExecutionHarness", message="; ".join(errors))
        return result


def record_platform_observation(
    *,
    memory_path: Path,
    run_name: str,
    stage: int,
    proposal_path: Path,
    manifest_path: Path,
    status: str,
    checkpoint_path: Optional[Path],
    job_id: Optional[str],
    error: Optional[str],
    frontier_allocation: Optional[str] = None,
    max_epochs: Optional[str] = None,
    rotation_fraction: Optional[str] = None,
    epoch_size: Optional[str] = None,
    sampler_seed: Optional[str] = None,
) -> None:
    """Record an observation from an existing shell/scheduler execution path."""
    with proposal_path.open("r", encoding="utf-8") as handle:
        proposal = json.load(handle)
    memory = CrossStageLearningMemory.load(memory_path, run_name)
    training_configuration = dict(proposal.get("training_script_inputs", {}))
    actual_frontier_allocation = proposal.get("boundary_weight")
    if frontier_allocation not in {None, ""}:
        actual_frontier_allocation = float(frontier_allocation)
    actual_replay_allocation = 1.0 - float(actual_frontier_allocation)
    for key, value in {
        "MAX_EPOCHS": max_epochs,
        "TBAT_ROTATION_FRACTION": rotation_fraction,
        "TBAT_EPOCH_SIZE": epoch_size,
        "TBAT_SAMPLER_SEED": sampler_seed,
        "TBAT_MANIFEST": str(manifest_path),
    }.items():
        if value not in {None, ""}:
            training_configuration[key] = str(value)
    applied = {
        "schema_version": "applied_curriculum.v1",
        "stage": stage,
        "verified_plan_path": str(proposal_path.parent / "verified_curriculum_plan.json"),
        "manifest_path": str(manifest_path),
        "sampler_configuration": {
            "frontier_allocation": actual_frontier_allocation,
            "retention_replay_allocation": actual_replay_allocation,
            "rotation_fraction": training_configuration.get("TBAT_ROTATION_FRACTION"),
            "epoch_size": training_configuration.get("TBAT_EPOCH_SIZE"),
            "sampler_seed": training_configuration.get("TBAT_SAMPLER_SEED"),
        },
        "training_configuration": training_configuration,
        "status": status,
        "job_id": job_id,
        "applied_at": utc_now(),
    }
    memory.record_applied(stage, applied)
    artifacts = {"manifest": str(manifest_path), "proposal": str(proposal_path)}
    if checkpoint_path is not None:
        artifacts["checkpoint"] = str(checkpoint_path)
    errors = [error] if error else []
    memory.record_execution(
        stage,
        ExecutionResult(
            stage=stage,
            status=status,
            applied_configuration=applied,
            metrics={},
            artifacts=artifacts,
            errors=errors,
            completed_at=utc_now() if status in {"SUCCEEDED", "FAILED"} else None,
        ).to_dict(),
    )
    if error:
        memory.record_error(stage=stage, source="CurriculumExecutionHarness", message=error)
    memory.save(memory_path)
    print(
        "[HARNESS] "
        f"stage={stage} status={status} manifest={manifest_path} "
        f"job_id={job_id or 'none'} checkpoint={checkpoint_path or 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record applied FrontierDrive curriculum execution state.")
    parser.add_argument("--memory-path", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", choices=["RUNNING", "SUCCEEDED", "FAILED"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--error")
    parser.add_argument("--frontier-allocation")
    parser.add_argument("--max-epochs")
    parser.add_argument("--rotation-fraction")
    parser.add_argument("--epoch-size")
    parser.add_argument("--sampler-seed")
    args = parser.parse_args()
    record_platform_observation(
        memory_path=args.memory_path,
        run_name=args.run_name,
        stage=args.stage,
        proposal_path=args.proposal,
        manifest_path=args.manifest,
        status=args.status,
        checkpoint_path=args.checkpoint,
        job_id=args.job_id,
        error=args.error,
        frontier_allocation=args.frontier_allocation,
        max_epochs=args.max_epochs,
        rotation_fraction=args.rotation_fraction,
        epoch_size=args.epoch_size,
        sampler_seed=args.sampler_seed,
    )


if __name__ == "__main__":
    main()
