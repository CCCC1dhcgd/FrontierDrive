#!/usr/bin/env python
"""Structured cross-stage memory for FrontierDrive.

The loader migrates the legacy ``experiment_memory.v1`` shape in memory and
keeps its legacy sections when saving so existing result-analysis scripts can
continue to read historical artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


MEMORY_SCHEMA = "cross_stage_learning_memory.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CrossStageLearningMemory:
    run_name: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    capability_state: Dict[str, Any] = field(default_factory=dict)
    frontier_state: Dict[str, Any] = field(default_factory=dict)
    capability_reports: Dict[str, Any] = field(default_factory=dict)
    frontier_reports: Dict[str, Any] = field(default_factory=dict)
    proposed_curricula: Dict[str, Any] = field(default_factory=dict)
    verified_curricula: Dict[str, Any] = field(default_factory=dict)
    applied_curricula: Dict[str, Any] = field(default_factory=dict)
    execution_results: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    # Compatibility views retained for existing experiment readers.
    evaluation_history: Dict[str, Any] = field(default_factory=dict)
    capability_history: Dict[str, Any] = field(default_factory=dict)
    scene_history: Dict[str, Any] = field(default_factory=dict)
    training_history: Dict[str, Any] = field(default_factory=dict)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: str = MEMORY_SCHEMA

    @classmethod
    def load(cls, path: Path, run_name: str) -> "CrossStageLearningMemory":
        payload: Dict[str, Any] = {}
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        memory = cls(
            run_name=str(payload.get("run_name") or run_name),
            created_at=str(payload.get("created_at") or utc_now()),
            updated_at=str(payload.get("updated_at") or utc_now()),
        )
        for name in (
            "capability_state",
            "frontier_state",
            "capability_reports",
            "frontier_reports",
            "proposed_curricula",
            "verified_curricula",
            "applied_curricula",
            "execution_results",
            "evaluation_results",
            "evaluation_history",
            "capability_history",
            "scene_history",
            "training_history",
        ):
            value = payload.get(name, {})
            setattr(memory, name, dict(value) if isinstance(value, dict) else {})
        for name in ("errors", "revision_history", "decision_history"):
            value = payload.get(name, [])
            setattr(memory, name, list(value) if isinstance(value, list) else [])

        # Populate new views from completed legacy records without rewriting
        # any old experiment artifact.
        if not memory.evaluation_results:
            memory.evaluation_results.update(memory.evaluation_history)
        if not memory.capability_state:
            memory.capability_state.update(memory.capability_history)
        if not memory.frontier_state:
            memory.frontier_state.update(memory.scene_history)
        return memory

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "capability_state": self.capability_state,
            "frontier_state": self.frontier_state,
            "capability_reports": self.capability_reports,
            "frontier_reports": self.frontier_reports,
            "proposed_curricula": self.proposed_curricula,
            "verified_curricula": self.verified_curricula,
            "applied_curricula": self.applied_curricula,
            "execution_results": self.execution_results,
            "evaluation_results": self.evaluation_results,
            "errors": self.errors,
            "revision_history": self.revision_history,
            "evaluation_history": self.evaluation_history,
            "capability_history": self.capability_history,
            "scene_history": self.scene_history,
            "training_history": self.training_history,
            "decision_history": self.decision_history,
        }

    def save(self, path: Path) -> None:
        self.updated_at = utc_now()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def snapshot(self, *, max_history: int = 8) -> Dict[str, Any]:
        def latest(mapping: Mapping[str, Any]) -> List[Any]:
            def stage_key(item: Any) -> int:
                try:
                    return int(item[0])
                except (TypeError, ValueError):
                    return -1

            return [item[1] for item in sorted(mapping.items(), key=stage_key, reverse=True)[:max_history]]

        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "capability_state": latest(self.capability_state),
            "frontier_state": latest(self.frontier_state),
            "capability_reports": latest(self.capability_reports),
            "frontier_reports": latest(self.frontier_reports),
            "proposed_curricula": latest(self.proposed_curricula),
            "verified_curricula": latest(self.verified_curricula),
            "applied_curricula": latest(self.applied_curricula),
            "execution_results": latest(self.execution_results),
            "evaluation_results": latest(self.evaluation_results),
            "errors": self.errors[-max_history:],
            "revision_history": self.revision_history[-max_history:],
        }

    def previous(self, section: str, stage: int) -> Optional[Dict[str, Any]]:
        mapping = getattr(self, section)
        if not isinstance(mapping, dict):
            return None
        for candidate in range(stage - 1, -1, -1):
            value = mapping.get(str(candidate))
            if isinstance(value, dict):
                return value
        return None

    def record_capability(self, stage: int, state: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        key = str(stage)
        self.capability_state[key] = dict(state)
        self.capability_reports[key] = dict(report)

    def record_frontier(self, stage: int, state: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        key = str(stage)
        self.frontier_state[key] = dict(state)
        self.frontier_reports[key] = dict(report)

    def record_proposed(self, stage: int, plan: Mapping[str, Any]) -> None:
        self.proposed_curricula[str(stage)] = dict(plan)

    def record_verified(self, stage: int, plan: Mapping[str, Any]) -> None:
        self.verified_curricula[str(stage)] = dict(plan)

    def record_applied(self, stage: int, applied: Mapping[str, Any]) -> None:
        self.applied_curricula[str(stage)] = dict(applied)

    def record_execution(self, stage: int, result: Mapping[str, Any]) -> None:
        self.execution_results[str(stage)] = dict(result)

    def record_evaluation(self, stage: int, result: Mapping[str, Any]) -> None:
        key = str(stage)
        self.evaluation_results[key] = dict(result)
        self.evaluation_history[key] = dict(result)

    def record_revision(self, request: Mapping[str, Any]) -> None:
        self.revision_history.append(dict(request))

    def record_error(self, *, stage: int, source: str, message: str, details: Optional[Mapping[str, Any]] = None) -> None:
        self.errors.append(
            {
                "stage": stage,
                "source": source,
                "message": message,
                "details": dict(details or {}),
                "created_at": utc_now(),
            }
        )
