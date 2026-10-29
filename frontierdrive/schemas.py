#!/usr/bin/env python
"""Typed schemas for the FrontierDrive curriculum-control loop.

The deterministic capability/frontier tools intentionally keep their existing
artifact formats.  These dataclasses form the typed boundary around those
artifacts and around the LLM decision stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


JsonDict = Dict[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class StructuredSchema:
    """Mixin providing JSON-safe serialization for all control-loop schemas."""

    def to_dict(self) -> JsonDict:
        return _json_value(asdict(self))


class RevisionTarget(str, Enum):
    CAPABILITY_ANALYST = "CAPABILITY_ANALYST"
    FRONTIER_MAPPER = "FRONTIER_MAPPER"
    CURRICULUM_PLANNER = "CURRICULUM_PLANNER"


@dataclass(frozen=True)
class CapabilityAnalystInput(StructuredSchema):
    stage: int
    policy_evaluation_results: Mapping[str, Any]
    calibrated_capability_estimates: Mapping[str, Any]
    stage_performance_history: Sequence[Mapping[str, Any]]
    cross_stage_memory: Mapping[str, Any]
    revision_feedback: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class CapabilityReport(StructuredSchema):
    stage: int
    checkpoint: str
    policy_capability: Optional[float]
    capability_change: Optional[float]
    capability_history: List[float]
    frontier_probability_range: List[float]
    frontier_scene_ids: List[str]
    frontier_scene_count: int
    minimum_frontier_scene_count: int
    stability: str
    confidence: str
    progress_summary: str
    persistent_limitations: List[str]
    emerging_limitations: List[str]
    evidence_issues: List[str]
    warnings: List[str]
    rationale: str
    deterministic_sources: Dict[str, str]
    llm_model: Optional[str] = None
    schema_version: str = "capability_report.v2"
    agent: str = "CapabilityAnalyst"


@dataclass(frozen=True)
class FrontierRegion(StructuredSchema):
    region_id: int
    prototype: List[float]
    support_radius: float
    frontier_scene_ids: List[str]
    frontier_scene_count: int
    covered_frontier_scene_count: int
    support_scene_ids: List[str]
    support_scene_count: int
    mean_support_distance: Optional[float]
    point_like: bool
    radius_floor_enforced: bool
    reliability: str = "unknown"
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FrontierMapperInput(StructuredSchema):
    stage: int
    capability_report: CapabilityReport
    deterministic_capability_frontier: Mapping[str, Any]
    frontier_prototypes: Sequence[Sequence[float]]
    frontier_support_set: Mapping[str, Any]
    coverage_statistics: Mapping[str, Any]
    migration_statistics: Mapping[str, Any]
    retrieval_statistics: Mapping[str, Any]
    cross_stage_memory: Mapping[str, Any]
    revision_feedback: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class FrontierReport(StructuredSchema):
    stage: int
    regions: List[FrontierRegion]
    curriculum_relevant_region_ids: List[int]
    sparse_region_ids: List[int]
    support_summary: str
    coverage: float
    support_scene_ids: List[str]
    support_scene_count: int
    support_manifest: str
    support_tokens_file: str
    coverage_statistics: Dict[str, Any]
    migration_statistics: Dict[str, Any]
    retrieval_statistics: Dict[str, Any]
    confidence: str
    warnings: List[str]
    rationale: str
    deterministic_sources: Dict[str, str]
    llm_model: Optional[str] = None
    schema_version: str = "frontier_report.v1"
    agent: str = "FrontierMapper"


@dataclass(frozen=True)
class TrainingBudget(StructuredSchema):
    training_epochs: int
    max_epochs: int
    epoch_size: int
    rotation_fraction: float
    sampler_seed: int
    budget_type: str = "fixed_epoch_sampler_budget"


@dataclass(frozen=True)
class CurriculumPlannerInput(StructuredSchema):
    stage: int
    capability_report: CapabilityReport
    frontier_report: FrontierReport
    frontier_support_set: Mapping[str, Any]
    retention_pool: Mapping[str, Any]
    previous_curriculum: Optional[Mapping[str, Any]]
    previous_outcome: Optional[Mapping[str, Any]]
    current_training_budget: TrainingBudget
    cross_stage_memory: Mapping[str, Any]
    revision_feedback: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class RegionAllocation(StructuredSchema):
    region_id: int
    weight: float
    support_scene_count: int
    frontier_scene_count: int
    support_radius: Optional[float]


@dataclass(frozen=True)
class CandidateCurriculumPlan(StructuredSchema):
    stage: int
    source_checkpoint: str
    budget: TrainingBudget
    frontier_allocation: float
    retention_replay_allocation: float
    region_allocations: List[RegionAllocation]
    support_manifest: str
    support_tokens_file: str
    support_scene_count: int
    training_configuration: Dict[str, str]
    rationale: List[str]
    warnings: List[str]
    llm_model: Optional[str] = None
    created_at: str = ""
    schema_version: str = "candidate_curriculum_plan.v1"
    agent: str = "CurriculumPlanner"

    def to_legacy_training_proposal(self) -> JsonDict:
        """Preserve the training_proposal.v1 interface used by stage scripts."""
        return {
            "schema_version": "training_proposal.v1",
            "agent": self.agent,
            "stage": self.stage,
            "source_checkpoint": self.source_checkpoint,
            "training_epochs": self.budget.training_epochs,
            "max_epochs": self.budget.max_epochs,
            "total_budget": {
                "type": self.budget.budget_type,
                "tbat_epoch_size": self.budget.epoch_size,
                "note": "TBAT_EPOCH_SIZE=-1 means the training script uses the navtrain dataset length.",
            },
            "boundary_weight": self.frontier_allocation,
            "replay_weight": self.retention_replay_allocation,
            "region_weights": [
                {
                    "cluster": item.region_id,
                    "weight": item.weight,
                    "candidate_train_count": item.support_scene_count,
                    "boundary_count": item.frontier_scene_count,
                    "radius": item.support_radius,
                }
                for item in self.region_allocations
            ],
            "selected_scene_ids_or_sampling_file": {
                "train_manifest": self.support_manifest,
                "selected_train_tokens": self.support_tokens_file,
                "candidate_train_count": self.support_scene_count,
            },
            "training_script_inputs": dict(self.training_configuration),
            "rationale": list(self.rationale),
            "warnings": list(self.warnings),
            "llm_model": self.llm_model,
            "created_at": self.created_at,
            "frontierdrive_schema": self.schema_version,
        }


@dataclass(frozen=True)
class DeterministicConstraints(StructuredSchema):
    min_frontier_allocation: float
    max_frontier_allocation: float
    max_region_allocation: float
    expected_frontier_allocation: float
    minimum_support_coverage: float
    maximum_revision_rounds: int


@dataclass(frozen=True)
class CurriculumVerifierInput(StructuredSchema):
    candidate_plan: CandidateCurriculumPlan
    deterministic_constraints: DeterministicConstraints
    data_split_information: Mapping[str, Any]
    platform_validation_results: Mapping[str, Any]
    capability_report: CapabilityReport
    frontier_report: FrontierReport
    revision_round: int
    cross_stage_memory: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedCurriculumPlan(StructuredSchema):
    candidate_plan: CandidateCurriculumPlan
    verification_checks: Dict[str, Any]
    verifier_rationale: str
    warnings: List[str]
    verified_at: str
    schema_version: str = "verified_curriculum_plan.v1"
    agent: str = "CurriculumVerifier"

    @property
    def stage(self) -> int:
        return self.candidate_plan.stage


@dataclass(frozen=True)
class RevisionRequest(StructuredSchema):
    stage: int
    target: RevisionTarget
    reason_codes: List[str]
    actionable_feedback: List[str]
    revision_round: int
    max_revisions: int
    rerun_deterministic: bool = False
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    terminal: bool = False
    schema_version: str = "revision_request.v1"
    agent: str = "CurriculumVerifier"


VerifierDecision = Union[VerifiedCurriculumPlan, RevisionRequest]


@dataclass(frozen=True)
class AppliedCurriculum(StructuredSchema):
    stage: int
    verified_plan_path: str
    manifest_path: str
    sampler_configuration: Dict[str, Any]
    training_configuration: Dict[str, Any]
    status: str
    job_id: Optional[str]
    applied_at: str
    schema_version: str = "applied_curriculum.v1"


@dataclass(frozen=True)
class ExecutionResult(StructuredSchema):
    stage: int
    status: str
    applied_configuration: Dict[str, Any]
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]
    errors: List[str]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    schema_version: str = "curriculum_execution_result.v1"
