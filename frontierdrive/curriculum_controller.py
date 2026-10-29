#!/usr/bin/env python
"""FrontierDrive multi-agent curriculum-control preparation loop.

This script replaces the old monolithic prepare-stage shell body while keeping
the numerical source of truth in the existing deterministic CBS tools:

  evaluation_summary.py -> capability_frontier.py -> frontier_retrieval.py

The agent layer only interprets deterministic evidence and makes curriculum
decisions.  The execution harness is a separate component and this prepare
entry point never submits or starts training/evaluation jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .learning_memory import CrossStageLearningMemory
    from .execution_harness import CurriculumExecutionHarness
    from .schemas import (
        CandidateCurriculumPlan,
        CapabilityAnalystInput,
        CapabilityReport,
        CurriculumPlannerInput,
        CurriculumVerifierInput,
        DeterministicConstraints,
        FrontierMapperInput,
        FrontierRegion,
        FrontierReport,
        RegionAllocation,
        RevisionRequest,
        RevisionTarget,
        TrainingBudget,
        VerifiedCurriculumPlan,
        VerifierDecision,
    )
except ImportError:  # Direct execution by the existing stage shell scripts.
    from learning_memory import CrossStageLearningMemory
    from execution_harness import CurriculumExecutionHarness
    from schemas import (
        CandidateCurriculumPlan,
        CapabilityAnalystInput,
        CapabilityReport,
        CurriculumPlannerInput,
        CurriculumVerifierInput,
        DeterministicConstraints,
        FrontierMapperInput,
        FrontierRegion,
        FrontierReport,
        RegionAllocation,
        RevisionRequest,
        RevisionTarget,
        TrainingBudget,
        VerifiedCurriculumPlan,
        VerifierDecision,
    )

np: Any = None


AUDIT_ACTIONS = {"PASS", "REVISE_A1", "REVISE_A2", "REVISE_A3", "ABORT"}
CONFIDENCE_CHOICES = {"low", "medium", "high"}
STABILITY_CHOICES = {"stable", "watch", "unstable"}
REVISION_TARGETS = {"REVISE_A1", "REVISE_A2", "REVISE_A3"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def compact_json(payload: Any, max_chars: int = 8000) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...<truncated {len(text) - max_chars} chars>"


def parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def require_choice(payload: Dict[str, Any], key: str, choices: set[str]) -> str:
    value = payload.get(key)
    if value not in choices:
        raise ValueError(f"{key} must be one of {sorted(choices)}, got {value!r}")
    return str(value)


def require_list(payload: Dict[str, Any], key: str) -> List[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def require_string(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def string_list(values: Iterable[Any], *, max_items: int = 50) -> List[str]:
    result: List[str] = []
    for value in values:
        if len(result) >= max_items:
            break
        result.append(str(value))
    return result


def write_stop(agent_dir: Path, reason: str, missing: Iterable[str], context: Dict[str, Any]) -> None:
    payload = {
        "schema_version": "boundary_prepare_stop.v1",
        "status": "STOP",
        "reason": reason,
        "missing": list(missing),
        "context": context,
        "created_at": utc_now(),
    }
    write_json(agent_dir / "stop_report.json", payload)
    lines = [
        "# Boundary Preparation STOP",
        "",
        f"- reason: {reason}",
        f"- created_at: {payload['created_at']}",
        "",
        "Missing required artifacts:",
    ]
    lines.extend(f"- `{item}`" for item in payload["missing"])
    (agent_dir / "stop_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_artifact_root(run_root: Path, stage: int) -> Path:
    structured_root = run_root / "evaluate" / f"stage{stage}"
    if structured_root.exists() or (run_root / "evaluate").exists():
        return structured_root
    return run_root / f"stage{stage}"


def eval_csv_path(run_root: Path, eval_name_prefix: str, stage: int) -> Path:
    return stage_artifact_root(run_root, stage) / f"{eval_name_prefix}_stage{stage}_navtest" / "metrics_results.csv"


def model_checkpoint(run_root: Path, stage: int) -> Path:
    model_dir = "warmup" if stage == 0 else f"stage{stage}"
    structured_checkpoint = run_root / "training" / model_dir / "models" / "last.ckpt"
    if structured_checkpoint.exists() or (run_root / "training").exists():
        return structured_checkpoint
    return run_root / "models" / model_dir / "last.ckpt"


def evaluation_summary_csv(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file(), "valid": False}
    if not path.is_file():
        return result
    scores: List[float] = []
    total_rows = 0
    valid_rows = 0
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "score" not in reader.fieldnames:
            result["error"] = "missing_score_column"
            return result
        for row in reader:
            if row.get("token") == "average_all_frames":
                continue
            total_rows += 1
            if row.get("valid") != "True":
                continue
            try:
                scores.append(float(row["score"]))
            except (TypeError, ValueError):
                continue
            valid_rows += 1
    if not scores:
        result["error"] = "no_valid_scores"
        result["total_scene_rows"] = total_rows
        result["valid_scene_rows"] = valid_rows
        return result
    ordered = sorted(scores)
    result.update(
        {
            "valid": True,
            "total_scene_rows": total_rows,
            "valid_scene_rows": valid_rows,
            "mean": float(sum(scores) / len(scores)),
            "min": float(ordered[0]),
            "p20": float(ordered[int(0.20 * (len(ordered) - 1))]),
            "p50": float(ordered[int(0.50 * (len(ordered) - 1))]),
            "p80": float(ordered[int(0.80 * (len(ordered) - 1))]),
            "max": float(ordered[-1]),
        }
    )
    return result


def run_command(command: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def load_npz(path: Path, required: Iterable[str]) -> Dict[str, np.ndarray]:
    require_numpy()
    data = np.load(path, allow_pickle=False)
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"{path} is missing arrays: {missing}")
    return {name: data[name] for name in data.files}


def require_numpy() -> None:
    global np
    if np is None:
        import numpy as numpy_module

        np = numpy_module


def token_list(values: np.ndarray) -> List[str]:
    return [str(value) for value in values.astype(str).tolist()]


@dataclass
class DeepSeekConfig:
    model: str
    base_url: str
    api_key_env: str
    api_key: Optional[str]
    temperature: float
    timeout: float
    max_retries: int
    max_tokens: int
    mock: bool
    mock_responses_path: Optional[Path]

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


class DeepSeekJsonClient:
    """Small OpenAI-compatible JSON client with retry and per-call audit logs."""

    def __init__(self, config: DeepSeekConfig, log_dir: Path) -> None:
        self.config = config
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.call_index = 0
        self.mock_responses = self._load_mock_responses(config.mock_responses_path)
        if not config.mock and not config.api_key:
            raise RuntimeError(
                f"DeepSeek API key is required. Set {config.api_key_env} or pass --deepseek-api-key-env."
            )

    def _load_mock_responses(self, path: Optional[Path]) -> Any:
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"Mock response file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def call_json(
        self,
        *,
        stage: int,
        agent_name: str,
        system_prompt: str,
        context: Dict[str, Any],
        input_summary: Dict[str, Any],
        validator,
    ) -> Dict[str, Any]:
        errors: List[str] = []
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(attempts):
            self.call_index += 1
            raw_response = ""
            parsed: Optional[Dict[str, Any]] = None
            try:
                user_prompt = self._build_user_prompt(context, errors)
                raw_response = self._mock_response(agent_name, attempt, context) if self.config.mock else self._api_response(system_prompt, user_prompt)
                parsed = parse_json_object(raw_response)
                validated = validator(parsed)
                self._write_call_log(
                    stage=stage,
                    agent_name=agent_name,
                    attempt=attempt,
                    input_summary=input_summary,
                    raw_response=raw_response,
                    parsed=validated,
                    errors=errors,
                    ok=True,
                )
                return validated
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                self._write_call_log(
                    stage=stage,
                    agent_name=agent_name,
                    attempt=attempt,
                    input_summary=input_summary,
                    raw_response=raw_response,
                    parsed=parsed,
                    errors=errors,
                    ok=False,
                )
        raise RuntimeError(f"{agent_name} failed to produce valid JSON after {attempts} attempts: {errors[-1]}")

    def _build_user_prompt(self, context: Dict[str, Any], previous_errors: List[str]) -> str:
        payload = {"structured_context": context}
        if previous_errors:
            payload["previous_errors_to_fix"] = previous_errors[-3:]
        return (
            "Return only one strict JSON object. Do not wrap it in markdown.\n"
            "Use the structured context below; deterministic numeric fields are immutable.\n\n"
            + compact_json(payload, max_chars=120000)
        )

    def _api_response(self, system_prompt: str, user_prompt: str) -> str:
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.config.chat_completions_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
        choices = payload.get("choices")
        if not choices:
            raise RuntimeError(f"DeepSeek response has no choices: {payload}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"DeepSeek response content is empty: {payload}")
        return content

    def _mock_response(self, agent_name: str, attempt: int, context: Dict[str, Any]) -> str:
        if isinstance(self.mock_responses, dict):
            value = self.mock_responses.get(agent_name)
            if value is None:
                value = self.mock_responses.get("*")
        elif isinstance(self.mock_responses, list) and self.mock_responses:
            value = self.mock_responses[min(self.call_index - 1, len(self.mock_responses) - 1)]
        else:
            value = default_mock_response(agent_name, context)
        if isinstance(value, list):
            value = value[min(attempt, len(value) - 1)]
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _write_call_log(
        self,
        *,
        stage: int,
        agent_name: str,
        attempt: int,
        input_summary: Dict[str, Any],
        raw_response: str,
        parsed: Optional[Dict[str, Any]],
        errors: List[str],
        ok: bool,
    ) -> None:
        safe_agent = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in agent_name)
        path = self.log_dir / f"{self.call_index:04d}_stage{stage}_{safe_agent}_attempt{attempt}.json"
        payload = {
            "schema_version": "deepseek_agent_call_log.v1",
            "created_at": utc_now(),
            "provider": "mock" if self.config.mock else "deepseek",
            "model": self.config.model,
            "base_url": self.config.base_url,
            "stage": stage,
            "agent_name": agent_name,
            "attempt": attempt,
            "input_summary": input_summary,
            "raw_response": raw_response,
            "parsed_output": parsed,
            "retry_errors": errors,
            "ok": ok,
        }
        write_json(path, payload)


def default_mock_response(agent_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = context or {}
    if agent_name in {"CapabilityAnalyst", "Agent1-CapabilityDiagnosis"}:
        return {
            "stability": "stable",
            "confidence": "high",
            "progress_summary": "Mock analysis found stable capability evidence.",
            "persistent_limitations": [],
            "emerging_limitations": [],
            "evidence_issues": [],
            "warnings": ["mock_llm_agent1_called"],
            "rationale": "Mock LLM accepted deterministic IRT/ZPD evidence.",
        }
    if agent_name in {"FrontierMapper", "Agent2-BoundaryLocalizationRetrieval"}:
        region_ids = [as_int(item.get("cluster"), -1) for item in context.get("region_summaries", [])]
        region_ids = [item for item in region_ids if item >= 0]
        return {
            "confidence": "high",
            "warnings": ["mock_llm_agent2_called"],
            "region_assessments": [],
            "curriculum_relevant_region_ids": region_ids,
            "sparse_region_ids": [],
            "support_summary": "Mock analysis accepted deterministic frontier support.",
            "rationale": "Mock LLM accepted deterministic sphere/retrieval evidence.",
        }
    if agent_name in {"CurriculumPlanner", "Agent3-CurriculumOrchestration"}:
        defaults = context.get("deterministic_defaults", {})
        region_weights = [
            {"cluster": as_int(item.get("cluster")), "weight": as_float(item.get("weight"))}
            for item in defaults.get("region_weights", [])
        ]
        return {
            "boundary_weight": as_float(defaults.get("boundary_mass"), 0.5),
            "replay_weight": as_float(defaults.get("replay_mass"), 0.5),
            "region_weights": region_weights,
            "warnings": ["mock_llm_agent3_called"],
            "rationale": ["Mock LLM generated the proposal decision fields."],
        }
    if agent_name in {"CurriculumVerifier", "Agent4-ConsistencySafetyAuditor"}:
        return {
            "status": "PASS",
            "reasons": ["mock_llm_agent4_called"],
            "actionable_feedback": [],
            "revision_plan": {},
            "rationale": "Mock LLM audit passed deterministic guard inputs.",
        }
    return {"rationale": "mock"}


@dataclass
class PrepareConfig:
    project_root: Path
    navsim_devkit_root: Path
    navsim_exp_root: Path
    python: str
    stage: int
    run_name: str
    eval_name_prefix: str
    feature_dir: Path
    stage_dir: Path
    eval_csv: Path
    previous_checkpoint: Path
    script_dir: Path
    memory_path: Path
    max_revision_rounds: int
    warmup_epochs: int
    stage_epochs: int
    max_epochs: Optional[int]
    tbat_rotation_fraction: float
    tbat_epoch_size: int
    tbat_sampler_seed: int
    min_boundary_weight: float
    max_boundary_weight: float
    max_region_weight: float
    audit_agent_enabled: bool
    deepseek: DeepSeekConfig
    values: Dict[str, Any]

    @property
    def agent_dir(self) -> Path:
        return self.stage_dir / "agent"

    @property
    def boundary_dir(self) -> Path:
        return self.stage_dir / "boundary"

    @property
    def retrieval_dir(self) -> Path:
        return self.stage_dir / "retrieval"

    @property
    def eval_dir(self) -> Path:
        return self.stage_dir / "eval"

    @property
    def run_root(self) -> Path:
        return self.navsim_exp_root / self.run_name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the memory-augmented TBAT boundary preparation loop.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--navsim-devkit-root", type=Path, required=True)
    parser.add_argument("--navsim-exp-root", type=Path, required=True)
    parser.add_argument("--python", type=str, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--eval-name-prefix", type=str, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--previous-checkpoint", type=Path, required=True)
    parser.add_argument("--memory-path", type=Path, required=True)
    parser.add_argument("--max-revision-rounds", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--stage-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--tbat-rotation-fraction", type=float, default=0.85)
    parser.add_argument("--tbat-epoch-size", type=int, default=-1)
    parser.add_argument("--tbat-sampler-seed", type=int, default=42)
    parser.add_argument("--min-boundary-weight", type=float, default=0.30)
    parser.add_argument("--max-boundary-weight", type=float, default=0.70)
    parser.add_argument("--max-region-weight", type=float, default=0.65)
    parser.add_argument("--deepseek-model", type=str, default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--deepseek-base-url", type=str, default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--deepseek-api-key-env", type=str, default=os.environ.get("DEEPSEEK_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--deepseek-temperature", type=float, default=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.2")))
    parser.add_argument("--deepseek-timeout", type=float, default=float(os.environ.get("DEEPSEEK_TIMEOUT", "120")))
    parser.add_argument("--deepseek-max-retries", type=int, default=int(os.environ.get("DEEPSEEK_MAX_RETRIES", "2")))
    parser.add_argument("--deepseek-max-tokens", type=int, default=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "16384")))
    parser.add_argument("--deepseek-mock", action="store_true", default=os.environ.get("DEEPSEEK_MOCK", "0") == "1")
    parser.add_argument("--deepseek-mock-responses", type=Path, default=Path(os.environ["DEEPSEEK_MOCK_RESPONSES"]) if os.environ.get("DEEPSEEK_MOCK_RESPONSES") else None)
    parser.add_argument(
        "--disable-audit-agent",
        action="store_true",
        default=os.environ.get("AUDIT_AGENT_ENABLED", "1") == "0" or os.environ.get("DISABLE_AUDIT_AGENT", "0") == "1",
        help="Skip the CurriculumVerifier LLM call and use deterministic guardrails only.",
    )
    parser.add_argument("--clusters", type=int, default=None, help="Optional initial retrieval KMeans cluster count. Default: auto.")

    for name, default, value_type in [
        ("tau", 0.95, float),
        ("p-lo", 0.20, float),
        ("p-hi", 0.60, float),
        ("n-min", 300, int),
        ("band-expand-step", 0.05, float),
        ("irt-l2", 0.02, float),
        ("irt-steps", 600, int),
        ("irt-lr", 0.15, float),
        ("min-agent-distance", 1.0, float),
        ("max-drac", 30.0, float),
        ("anneal-rate", 0.15, float),
        ("min-anneal", 0.65, float),
        ("radius-scale", 1.5, float),
        ("radius-knn", 5, int),
        ("difficulty-gain", 0.35, float),
        ("boundary-mass", 0.50, float),
        ("min-train-per-sphere", 300, int),
        ("merge-distance-scale", 1.0, float),
        ("isolated-boundary-count", 2, int),
        ("max-floor-radius-ratio", 3.0, float),
        ("max-sphere-radius-scale", 1.0, float),
        ("point-like-radius-scale", 1.0, float),
        ("point-like-train-count", 10, int),
    ]:
        parser.add_argument(f"--{name}", type=value_type, default=default)
    return parser


def config_from_args(args: argparse.Namespace) -> PrepareConfig:
    script_dir = Path(__file__).resolve().parent
    deepseek_config = DeepSeekConfig(
        model=args.deepseek_model,
        base_url=args.deepseek_base_url,
        api_key_env=args.deepseek_api_key_env,
        api_key=os.environ.get(args.deepseek_api_key_env),
        temperature=args.deepseek_temperature,
        timeout=args.deepseek_timeout,
        max_retries=args.deepseek_max_retries,
        max_tokens=args.deepseek_max_tokens,
        mock=args.deepseek_mock,
        mock_responses_path=args.deepseek_mock_responses,
    )
    values = {
        "TAU": args.tau,
        "ZPD_P_LO": args.p_lo,
        "ZPD_P_HI": args.p_hi,
        "ZPD_N_MIN": args.n_min,
        "BAND_EXPAND_STEP": args.band_expand_step,
        "IRT_L2": args.irt_l2,
        "IRT_STEPS": args.irt_steps,
        "IRT_LR": args.irt_lr,
        "MIN_AGENT_DISTANCE": args.min_agent_distance,
        "MAX_DRAC": args.max_drac,
        "ANNEAL_RATE": args.anneal_rate,
        "MIN_ANNEAL": args.min_anneal,
        "CLUSTERS": args.clusters,
        "RADIUS_SCALE": args.radius_scale,
        "RADIUS_KNN": args.radius_knn,
        "DIFFICULTY_GAIN": args.difficulty_gain,
        "BOUNDARY_MASS": args.boundary_mass,
        "MIN_TRAIN_PER_SPHERE": args.min_train_per_sphere,
        "MERGE_DISTANCE_SCALE": args.merge_distance_scale,
        "ISOLATED_BOUNDARY_COUNT": args.isolated_boundary_count,
        "MAX_FLOOR_RADIUS_RATIO": args.max_floor_radius_ratio,
        "MAX_SPHERE_RADIUS_SCALE": args.max_sphere_radius_scale,
        "POINT_LIKE_RADIUS_SCALE": args.point_like_radius_scale,
        "POINT_LIKE_TRAIN_COUNT": args.point_like_train_count,
        "TBAT_ROTATION_FRACTION": args.tbat_rotation_fraction,
        "TBAT_EPOCH_SIZE": args.tbat_epoch_size,
        "TBAT_SAMPLER_SEED": args.tbat_sampler_seed,
        "MIN_BOUNDARY_WEIGHT": args.min_boundary_weight,
        "MAX_BOUNDARY_WEIGHT": args.max_boundary_weight,
        "MAX_REGION_WEIGHT": args.max_region_weight,
        "AUDIT_AGENT_ENABLED": not args.disable_audit_agent,
        "MAX_REVISION_ROUNDS": args.max_revision_rounds,
        "WARMUP_EPOCHS": args.warmup_epochs,
        "STAGE_EPOCHS": args.stage_epochs,
        "MAX_EPOCHS": args.max_epochs,
        "DEEPSEEK_MODEL": deepseek_config.model,
        "DEEPSEEK_BASE_URL": deepseek_config.base_url,
        "DEEPSEEK_TEMPERATURE": deepseek_config.temperature,
        "DEEPSEEK_TIMEOUT": deepseek_config.timeout,
        "DEEPSEEK_MAX_RETRIES": deepseek_config.max_retries,
        "DEEPSEEK_MAX_TOKENS": deepseek_config.max_tokens,
        "DEEPSEEK_MOCK": deepseek_config.mock,
    }
    return PrepareConfig(
        project_root=args.project_root,
        navsim_devkit_root=args.navsim_devkit_root,
        navsim_exp_root=args.navsim_exp_root,
        python=args.python,
        stage=args.stage,
        run_name=args.run_name,
        eval_name_prefix=args.eval_name_prefix,
        feature_dir=args.feature_dir,
        stage_dir=args.stage_dir,
        eval_csv=args.eval_csv,
        previous_checkpoint=args.previous_checkpoint,
        script_dir=script_dir,
        memory_path=args.memory_path,
        max_revision_rounds=args.max_revision_rounds,
        warmup_epochs=args.warmup_epochs,
        stage_epochs=args.stage_epochs,
        max_epochs=args.max_epochs,
        tbat_rotation_fraction=args.tbat_rotation_fraction,
        tbat_epoch_size=args.tbat_epoch_size,
        tbat_sampler_seed=args.tbat_sampler_seed,
        min_boundary_weight=args.min_boundary_weight,
        max_boundary_weight=args.max_boundary_weight,
        max_region_weight=args.max_region_weight,
        audit_agent_enabled=not args.disable_audit_agent,
        deepseek=deepseek_config,
        values=values,
    )


def validate_config(cfg: PrepareConfig) -> None:
    if cfg.stage < 1:
        raise ValueError(f"prepare stage must be >= 1, got {cfg.stage}")
    if cfg.max_revision_rounds < 1:
        raise ValueError("max revision rounds must be >= 1")
    if not 0.0 <= as_float(cfg.values["ZPD_P_LO"]) < as_float(cfg.values["ZPD_P_HI"]) <= 1.0:
        raise ValueError("ZPD probability band is invalid")
    if not 0.0 < as_float(cfg.values["BOUNDARY_MASS"]) < 1.0:
        raise ValueError("BOUNDARY_MASS must be in (0, 1)")
    if not 0.0 < cfg.tbat_rotation_fraction <= 1.0:
        raise ValueError("TBAT_ROTATION_FRACTION must be in (0, 1]")
    if not 0.0 <= cfg.min_boundary_weight < cfg.max_boundary_weight <= 1.0:
        raise ValueError("boundary weight range is invalid")
    if not 0.0 < cfg.max_region_weight <= 1.0:
        raise ValueError("MAX_REGION_WEIGHT must be in (0, 1]")
    if not cfg.deepseek.model.strip():
        raise ValueError("DeepSeek model must be non-empty")
    if not cfg.deepseek.base_url.strip():
        raise ValueError("DeepSeek base URL must be non-empty")
    if cfg.deepseek.max_retries < 0:
        raise ValueError("DeepSeek max retries must be >= 0")
    if cfg.deepseek.max_tokens < 1:
        raise ValueError("DeepSeek max tokens must be positive")
    if not cfg.deepseek.mock and not cfg.deepseek.api_key:
        raise RuntimeError(f"DeepSeek API key is required. Set {cfg.deepseek.api_key_env}.")


def update_memory_from_completed_artifacts(memory: CrossStageLearningMemory, cfg: PrepareConfig) -> None:
    previous_stage = cfg.stage - 1
    for stage in range(previous_stage + 1):
        csv_path = eval_csv_path(cfg.run_root, cfg.eval_name_prefix, stage)
        eval_summary = evaluation_summary_csv(csv_path)
        if eval_summary.get("valid"):
            evaluation_record = {
                "stage": stage,
                "navtest_path": str(csv_path),
                "overall_metrics": eval_summary,
                "completed": True,
                "updated_at": utc_now(),
            }
            memory.record_evaluation(stage, evaluation_record)

        ckpt = model_checkpoint(cfg.run_root, stage)
        if ckpt.is_file():
            record = {
                "stage": stage,
                "checkpoint": str(ckpt),
                "completed": True,
                "updated_at": utc_now(),
            }
            if stage == 0:
                record["epochs"] = cfg.warmup_epochs
                record["stage_kind"] = "warmup"
            else:
                record["epochs"] = cfg.stage_epochs
                record["stage_kind"] = "tbat_stage"
                retrieval_root = stage_artifact_root(cfg.run_root, stage) / "retrieval"
                curriculum_manifest = retrieval_root / "curriculum_manifest.csv"
                manifest = curriculum_manifest if curriculum_manifest.is_file() else retrieval_root / "train_manifest.csv"
                if manifest.is_file():
                    record["actual_sampling_manifest"] = str(manifest)
            memory.training_history[str(stage)] = record
            if stage > 0:
                stage_root = stage_artifact_root(cfg.run_root, stage)
                proposal_path = stage_root / "agent" / "training_proposal.json"
                verified_path = stage_root / "agent" / "verified_curriculum_plan.json"
                curriculum_manifest = stage_root / "retrieval" / "curriculum_manifest.csv"
                manifest_path = curriculum_manifest if curriculum_manifest.is_file() else stage_root / "retrieval" / "train_manifest.csv"
                if proposal_path.is_file() and manifest_path.is_file():
                    proposal = read_json(proposal_path)
                    applied = {
                        "schema_version": "applied_curriculum.v1",
                        "stage": stage,
                        "verified_plan_path": str(verified_path) if verified_path.is_file() else "legacy_unavailable",
                        "manifest_path": str(manifest_path),
                        "sampler_configuration": {
                            "frontier_allocation": proposal.get("boundary_weight"),
                            "retention_replay_allocation": proposal.get("replay_weight"),
                            "rotation_fraction": proposal.get("training_script_inputs", {}).get("TBAT_ROTATION_FRACTION"),
                            "epoch_size": proposal.get("training_script_inputs", {}).get("TBAT_EPOCH_SIZE"),
                            "sampler_seed": proposal.get("training_script_inputs", {}).get("TBAT_SAMPLER_SEED"),
                        },
                        "training_configuration": proposal.get("training_script_inputs", {}),
                        "status": "SUCCEEDED",
                        "job_id": None,
                        "applied_at": record["updated_at"],
                    }
                    memory.record_applied(stage, applied)
                    memory.record_execution(
                        stage,
                        {
                            "schema_version": "curriculum_execution_result.v1",
                            "stage": stage,
                            "status": "SUCCEEDED",
                            "applied_configuration": applied,
                            "metrics": memory.evaluation_results.get(str(stage), {}),
                            "artifacts": {"checkpoint": str(ckpt), "manifest": str(manifest_path)},
                            "errors": [],
                            "completed_at": record["updated_at"],
                        },
                    )


def required_inputs(cfg: PrepareConfig) -> List[Path]:
    required = [
        cfg.eval_csv,
        cfg.previous_checkpoint,
        cfg.feature_dir / "navtest_aligned.npz",
        cfg.feature_dir / "navtrain_aligned.npz",
        cfg.script_dir / "evaluation_summary.py",
        cfg.script_dir / "capability_frontier.py",
        cfg.script_dir / "frontier_retrieval.py",
        cfg.script_dir / "visualization.py",
    ]
    for stage in range(cfg.stage):
        required.append(eval_csv_path(cfg.run_root, cfg.eval_name_prefix, stage))
    return required


def run_eval_summary(cfg: PrepareConfig, env: Dict[str, str]) -> None:
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            cfg.python,
            "-u",
            str(cfg.script_dir / "evaluation_summary.py"),
            "--eval-csv",
            str(cfg.eval_csv),
            "--output-dir",
            str(cfg.eval_dir),
        ],
        cwd=cfg.project_root,
        env=env,
    )


def run_capability_frontier(cfg: PrepareConfig, env: Dict[str, str]) -> None:
    cfg.boundary_dir.mkdir(parents=True, exist_ok=True)
    identify_args = [
        cfg.python,
        "-u",
        str(cfg.script_dir / "capability_frontier.py"),
        "--navtest-aligned",
        str(cfg.feature_dir / "navtest_aligned.npz"),
        "--eval-history-root",
        str(cfg.run_root),
        "--eval-name-prefix",
        cfg.eval_name_prefix,
        "--output-dir",
        str(cfg.boundary_dir),
        "--stage",
        str(cfg.stage),
        "--tau",
        str(cfg.values["TAU"]),
        "--p-lo",
        str(cfg.values["ZPD_P_LO"]),
        "--p-hi",
        str(cfg.values["ZPD_P_HI"]),
        "--n-min",
        str(cfg.values["ZPD_N_MIN"]),
        "--band-expand-step",
        str(cfg.values["BAND_EXPAND_STEP"]),
        "--irt-l2",
        str(cfg.values["IRT_L2"]),
        "--irt-steps",
        str(cfg.values["IRT_STEPS"]),
        "--irt-lr",
        str(cfg.values["IRT_LR"]),
        "--min-agent-distance",
        str(cfg.values["MIN_AGENT_DISTANCE"]),
        "--max-drac",
        str(cfg.values["MAX_DRAC"]),
        "--anneal-rate",
        str(cfg.values["ANNEAL_RATE"]),
        "--min-anneal",
        str(cfg.values["MIN_ANNEAL"]),
    ]
    run_command(identify_args, cwd=cfg.project_root, env=env)


def run_retrieve_boundary(cfg: PrepareConfig, env: Dict[str, str]) -> None:
    cfg.retrieval_dir.mkdir(parents=True, exist_ok=True)
    retrieve_args = [
        cfg.python,
        "-u",
        str(cfg.script_dir / "frontier_retrieval.py"),
        "--navtrain-aligned",
        str(cfg.feature_dir / "navtrain_aligned.npz"),
        "--boundary",
        str(cfg.boundary_dir / "navtest_boundary.npz"),
        "--output-dir",
        str(cfg.retrieval_dir),
        "--boundary-mass",
        str(cfg.values["BOUNDARY_MASS"]),
        "--radius-scale",
        str(cfg.values["RADIUS_SCALE"]),
        "--radius-knn",
        str(cfg.values["RADIUS_KNN"]),
        "--difficulty-gain",
        str(cfg.values["DIFFICULTY_GAIN"]),
        "--min-train-per-sphere",
        str(cfg.values["MIN_TRAIN_PER_SPHERE"]),
        "--merge-distance-scale",
        str(cfg.values["MERGE_DISTANCE_SCALE"]),
        "--isolated-boundary-count",
        str(cfg.values["ISOLATED_BOUNDARY_COUNT"]),
        "--max-floor-radius-ratio",
        str(cfg.values["MAX_FLOOR_RADIUS_RATIO"]),
        "--max-sphere-radius-scale",
        str(cfg.values["MAX_SPHERE_RADIUS_SCALE"]),
        "--point-like-radius-scale",
        str(cfg.values["POINT_LIKE_RADIUS_SCALE"]),
        "--point-like-train-count",
        str(cfg.values["POINT_LIKE_TRAIN_COUNT"]),
    ]
    if cfg.values.get("CLUSTERS") is not None:
        retrieve_args.extend(["--clusters", str(cfg.values["CLUSTERS"])])
    run_command(retrieve_args, cwd=cfg.project_root, env=env)


def run_visualization(cfg: PrepareConfig, env: Dict[str, str]) -> None:
    cfg.retrieval_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            cfg.python,
            "-u",
            str(cfg.script_dir / "visualization.py"),
            "--navtrain-aligned",
            str(cfg.feature_dir / "navtrain_aligned.npz"),
            "--boundary",
            str(cfg.boundary_dir / "navtest_boundary.npz"),
            "--selection",
            str(cfg.retrieval_dir / "train_selection.npz"),
            "--output-dir",
            str(cfg.retrieval_dir),
        ],
        cwd=cfg.project_root,
        env=env,
    )


def run_boundary_retrieval_prepare(cfg: PrepareConfig, env: Dict[str, str]) -> None:
    run_capability_frontier(cfg, env)
    run_retrieve_boundary(cfg, env)
    run_visualization(cfg, env)


def run_deterministic_prepare(cfg: PrepareConfig, env: Dict[str, str]) -> None:
    run_eval_summary(cfg, env)
    run_boundary_retrieval_prepare(cfg, env)


def deterministic_artifacts(cfg: PrepareConfig) -> List[Path]:
    return [
        cfg.eval_dir / "eval_summary.json",
        cfg.boundary_dir / "boundary_summary.json",
        cfg.boundary_dir / "navtest_boundary.npz",
        cfg.retrieval_dir / "sphere_summary.json",
        cfg.retrieval_dir / "train_manifest.csv",
        cfg.retrieval_dir / "train_selection.npz",
    ]


def assert_deterministic_artifacts(cfg: PrepareConfig) -> None:
    for artifact in deterministic_artifacts(cfg):
        if not artifact.is_file():
            raise FileNotFoundError(f"Deterministic prepare artifact missing after run: {artifact}")


def build_capability_analyst_input(
    cfg: PrepareConfig,
    memory: CrossStageLearningMemory,
    feedback: Optional[RevisionRequest] = None,
) -> CapabilityAnalystInput:
    summary = read_json(cfg.boundary_dir / "boundary_summary.json")
    boundary_npz = load_npz(
        cfg.boundary_dir / "navtest_boundary.npz",
        ["boundary_tokens", "theta_history", "pass_probability", "p_lo", "p_hi"],
    )
    estimates = dict(summary)
    estimates.update(
        {
            "theta_history": [float(value) for value in np.asarray(boundary_npz["theta_history"]).tolist()],
            "frontier_scene_ids": token_list(np.asarray(boundary_npz["boundary_tokens"])),
            "calibrated_success_probability_count": int(np.asarray(boundary_npz["pass_probability"]).size),
        }
    )
    current_eval = read_json(cfg.eval_dir / "eval_summary.json")
    return CapabilityAnalystInput(
        stage=cfg.stage,
        policy_evaluation_results=current_eval,
        calibrated_capability_estimates=estimates,
        stage_performance_history=list(memory.snapshot().get("evaluation_results", [])),
        cross_stage_memory=memory.snapshot(),
        revision_feedback=feedback.to_dict() if feedback else None,
    )


def build_frontier_mapper_input(
    cfg: PrepareConfig,
    memory: CrossStageLearningMemory,
    capability_report: CapabilityReport,
    feedback: Optional[RevisionRequest] = None,
) -> FrontierMapperInput:
    frontier_summary = read_json(cfg.boundary_dir / "boundary_summary.json")
    retrieval_summary = read_json(cfg.retrieval_dir / "sphere_summary.json")
    selection = load_npz(
        cfg.retrieval_dir / "train_selection.npz",
        ["tokens", "selected", "centers", "radii", "boundary_tokens", "boundary_covered"],
    )
    selected = np.asarray(selection["selected"], dtype=bool)
    train_tokens = np.asarray(selection["tokens"]).astype(str)
    support_ids = train_tokens[selected].tolist()
    coverage = as_float(retrieval_summary.get("boundary_coverage_fraction"))
    previous = memory.previous("frontier_state", cfg.stage) or {}
    previous_ids = set(previous.get("support_scene_ids", []))
    current_ids = set(support_ids)
    union = previous_ids | current_ids
    migration = {
        "previous_stage": previous.get("stage"),
        "coverage_change": coverage - as_float(previous.get("coverage")) if previous else None,
        "support_turnover": (1.0 - len(previous_ids & current_ids) / len(union)) if union and previous else None,
        "source": "deterministic_support_set_comparison",
    }
    return FrontierMapperInput(
        stage=cfg.stage,
        capability_report=capability_report,
        deterministic_capability_frontier=frontier_summary,
        frontier_prototypes=np.asarray(selection["centers"], dtype=float).tolist(),
        frontier_support_set={
            "support_scene_ids": support_ids,
            "support_scene_count": len(support_ids),
            "retention_scene_count": int((~selected).sum()),
            "manifest": str(cfg.retrieval_dir / "train_manifest.csv"),
        },
        coverage_statistics={
            "frontier_coverage": coverage,
            "covered_frontier_scene_count": int(np.asarray(selection["boundary_covered"], dtype=bool).sum()),
            "frontier_scene_count": int(np.asarray(selection["boundary_tokens"]).size),
        },
        migration_statistics=migration,
        retrieval_statistics=retrieval_summary,
        cross_stage_memory=memory.snapshot(),
        revision_feedback=feedback.to_dict() if feedback else None,
    )


def current_training_budget(cfg: PrepareConfig) -> TrainingBudget:
    max_epochs = cfg.max_epochs if cfg.max_epochs is not None else cfg.warmup_epochs + cfg.stage * cfg.stage_epochs
    return TrainingBudget(
        training_epochs=cfg.stage_epochs,
        max_epochs=int(max_epochs),
        epoch_size=cfg.tbat_epoch_size,
        rotation_fraction=cfg.tbat_rotation_fraction,
        sampler_seed=cfg.tbat_sampler_seed,
    )


def build_curriculum_planner_input(
    cfg: PrepareConfig,
    memory: CrossStageLearningMemory,
    capability_report: CapabilityReport,
    frontier_report: FrontierReport,
    feedback: Optional[RevisionRequest] = None,
) -> CurriculumPlannerInput:
    manifest_check = validate_manifest(frontier_report.support_manifest)
    return CurriculumPlannerInput(
        stage=cfg.stage,
        capability_report=capability_report,
        frontier_report=frontier_report,
        frontier_support_set={
            "scene_ids": frontier_report.support_scene_ids,
            "scene_count": frontier_report.support_scene_count,
            "manifest": frontier_report.support_manifest,
        },
        retention_pool={
            "scene_count": manifest_check.get("replay_count", 0),
            "manifest": frontier_report.support_manifest,
        },
        previous_curriculum=memory.previous("applied_curricula", cfg.stage),
        previous_outcome=memory.previous("execution_results", cfg.stage),
        current_training_budget=current_training_budget(cfg),
        cross_stage_memory=memory.snapshot(),
        revision_feedback=feedback.to_dict() if feedback else None,
    )


def build_curriculum_verifier_input(
    cfg: PrepareConfig,
    memory: CrossStageLearningMemory,
    candidate: CandidateCurriculumPlan,
    capability_report: CapabilityReport,
    frontier_report: FrontierReport,
    revision_round: int,
) -> CurriculumVerifierInput:
    manifest_check = validate_manifest(candidate.support_manifest)
    manifest_region_check = validate_manifest_regions(
        Path(candidate.support_manifest), {item.region_id for item in candidate.region_allocations}
    )
    leakage = train_test_leakage(Path(candidate.support_manifest), set(capability_report.frontier_scene_ids))
    return CurriculumVerifierInput(
        candidate_plan=candidate,
        deterministic_constraints=DeterministicConstraints(
            min_frontier_allocation=cfg.min_boundary_weight,
            max_frontier_allocation=cfg.max_boundary_weight,
            max_region_allocation=cfg.max_region_weight,
            expected_frontier_allocation=as_float(cfg.values["BOUNDARY_MASS"]),
            minimum_support_coverage=0.80,
            maximum_revision_rounds=cfg.max_revision_rounds,
        ),
        data_split_information={
            "evaluation_frontier_scene_count": len(capability_report.frontier_scene_ids),
            "train_evaluation_overlap": leakage,
            "separation_valid": not leakage,
        },
        platform_validation_results={
            "manifest": manifest_check,
            "manifest_regions": manifest_region_check,
            "source_checkpoint_exists": Path(candidate.source_checkpoint).is_file(),
            "training_configuration_fields": sorted(candidate.training_configuration),
        },
        capability_report=capability_report,
        frontier_report=frontier_report,
        revision_round=revision_round,
        cross_stage_memory=memory.snapshot(),
    )


CAPABILITY_SYSTEM_PROMPT = """You are the FrontierDrive Capability Analyst.

Role:
- Diagnose current Policy Capability from deterministic calibrated evidence.
- Summarize progress, persistent/emerging limitations, and insufficient or abnormal evidence.

Input fields:
- boundary_summary: deterministic output from capability_frontier.py.
- theta_history/theta/theta_change: deterministic IRT ability history.
- boundary_count/n_min/boundary_fraction/gate_pass: deterministic ZPD and stability signals.
- memory_snapshot: persisted ExperimentMemory summaries from completed experiments only.
- deterministic_rule_signals: rule-derived warnings and suggested labels. These are constraints/signals, not free text.

Rules and decision range:
- IRT theta, item difficulty, pass_probability, ZPD band, boundary count, and boundary scene IDs are immutable.
- Do not invent, smooth, or recompute numeric values.
- Choose stability from stable|watch|unstable and confidence from low|medium|high.
- Warnings must be grounded in provided fields and rules.

Output schema:
{
  "stability": "stable|watch|unstable",
  "confidence": "low|medium|high",
  "progress_summary": "string",
  "persistent_limitations": ["string"],
  "emerging_limitations": ["string"],
  "evidence_issues": ["string"],
  "warnings": ["string"],
  "rationale": "string"
}
Return only JSON."""


BOUNDARY_SYSTEM_PROMPT = """You are the FrontierDrive Frontier Mapper.

Role:
- Interpret the deterministically computed Capability Frontier, Frontier Prototypes,
  Adaptive Frontier Neighborhoods, Frontier Support Set, and retrieval statistics.

Input fields:
- capability_report: CapabilityAnalyst report with immutable capability/frontier identifiers.
- sphere_summary: deterministic output from frontier_retrieval.py.
- region_summaries: deterministic centers, radii, boundary anchors, coverage, and candidate train counts.
- distance_relevance: deterministic retrieval relevance statistics.
- feedback: optional CurriculumVerifier feedback from the previous revision round.
- memory_snapshot: persisted scene/capability history from completed experiments.

Rules and decision range:
- Centers, radii, distances, coverage, selected train IDs, and manifest path are immutable.
- Do not invent new candidate train scenes or modify retrieval radii.
- Never estimate the frontier and never retrieve scenes. Assess only supplied outputs.
- Identify curriculum-relevant and sparse region IDs using only existing region IDs.
- Confidence must be low|medium|high.

Output schema:
{
  "confidence": "low|medium|high",
  "warnings": ["string"],
  "curriculum_relevant_region_ids": [0],
  "sparse_region_ids": [0],
  "support_summary": "string",
  "region_assessments": [
    {"cluster": 0, "reliability": "reliable|sparse|over_broad|point_like|unknown", "warnings": ["string"]}
  ],
  "rationale": "string"
}
Return only JSON."""


CURRICULUM_SYSTEM_PROMPT = """You are the FrontierDrive Curriculum Planner for the next training stage.

Role:
- Allocate training effort across supplied frontier regions and Retention Replay
  under the supplied fixed budget and constrained action space.

Input fields:
- capability_report: CapabilityAnalyst diagnosis.
- frontier_report: FrontierMapper interpretation of deterministic retrieval output.
- deterministic_defaults: immutable training-script defaults and budget values.
- allowed_ranges: configured limits for boundary/replay and per-region weights.
- feedback: optional CurriculumVerifier feedback from a previous revision round.
- memory_snapshot: completed training and decision history only.

Rules and decision range:
- Do not modify train_manifest, selected scene IDs, source checkpoint, stage, stage epochs, or max epoch formula.
- boundary_weight and replay_weight must sum to 1 and stay inside allowed_ranges.
- In this implementation the generated train_manifest already encodes the deterministic BOUNDARY_MASS. Therefore boundary_weight must equal deterministic_defaults.boundary_mass unless the feedback asks for ABORT-level rejection; otherwise request a future rerun via warnings rather than changing the manifest.
- region_weights are proposal-level allocation explanations; they must refer only to existing cluster IDs, be non-negative, sum to 1 when regions exist, and respect max_region_weight.

Output schema:
{
  "boundary_weight": 0.5,
  "replay_weight": 0.5,
  "region_weights": [{"cluster": 0, "weight": 0.25}],
  "warnings": ["string"],
  "rationale": ["string"]
}
Return only JSON."""


AUDITOR_SYSTEM_PROMPT = """You are the FrontierDrive Curriculum Verifier.

Role:
- Interpret deterministic validation results for a CandidateCurriculumPlan and
  either accept it or request a targeted revision.

Input fields:
- memory_snapshot: persisted experiment memory; proposals are not completed training.
- capability_report, frontier_report, candidate curriculum: structured upstream outputs.
- deterministic_guard_report: deterministic post-LLM checks including sparse/diffuse boundary, sphere coverage, manifest validity, leakage, and legal weight ranges.
- revision_round/max_revision_rounds: targeted rollback state.

Rules and decision range:
- Output status must be one of PASS, REVISE_A1, REVISE_A2, REVISE_A3, ABORT.
- Do not approve train/test leakage, invalid manifest, illegal weights, or exhausted revision rounds.
- Prefer targeted rollback: REVISE_A1 for IRT/ZPD capability issues, REVISE_A2 for localization/retrieval issues, REVISE_A3 for proposal/weight issues.
- Revision requests change only the target Agent's interpretation or curriculum decision.
- Do not propose changes to capability/frontier/retrieval algorithm parameters.
- Do not invent new deterministic values; cite provided guard reasons.

Output schema:
{
  "status": "PASS|REVISE_A1|REVISE_A2|REVISE_A3|ABORT",
  "reasons": ["string"],
  "actionable_feedback": ["string"],
  "revision_plan": {
    "target_agent": "REVISE_A1|REVISE_A2|REVISE_A3",
    "notes": ["string"]
  },
  "rationale": "string"
}
Return only JSON."""


def memory_snapshot(memory: Any, *, max_history: int = 8) -> Dict[str, Any]:
    if isinstance(memory, CrossStageLearningMemory):
        return memory.snapshot(max_history=max_history)
    def latest_records(section: str) -> List[Any]:
        value = memory.get(section, {})
        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda item: as_int(item[0]), reverse=True)
            return [item[1] for item in items[:max_history]]
        if isinstance(value, list):
            return value[-max_history:]
        return []

    return {
        "schema_version": memory.get("schema_version"),
        "run_name": memory.get("run_name"),
        "evaluation_history": latest_records("evaluation_history"),
        "capability_history": latest_records("capability_history"),
        "scene_history": latest_records("scene_history"),
        "training_history": latest_records("training_history"),
        "decision_history": memory.get("decision_history", [])[-max_history:],
    }


def deterministic_capability_signals(summary: Dict[str, Any], theta_history: List[float]) -> Dict[str, Any]:
    warnings: List[str] = []
    n_boundary = as_int(summary.get("n_boundary"))
    n_min = as_int(summary.get("n_min"))
    gate_pass = bool(summary.get("gate_pass"))
    estimator = str(summary.get("estimator", "unknown"))
    if estimator == "feature_prior_until_three_rounds":
        warnings.append("IRT history has fewer than three rounds; difficulty uses feature prior.")
    if not bool(summary.get("gate_evaluated")):
        warnings.append("Capability stability gate was not evaluated because only one history round exists.")
    elif not gate_pass:
        warnings.append("Capability gate did not pass; theta or p20 regressed relative to previous navtest.")
    if n_boundary < n_min:
        warnings.append(f"Boundary count {n_boundary} is below configured ZPD_N_MIN {n_min}.")
    if as_float(summary.get("boundary_fraction")) > 0.55:
        warnings.append("Boundary fraction exceeds 0.55 and may be too diffuse.")

    if n_boundary >= n_min and gate_pass and len(theta_history) >= 3:
        suggested_stability = "stable"
        suggested_confidence = "high"
    elif n_boundary >= n_min:
        suggested_stability = "watch"
        suggested_confidence = "medium"
    else:
        suggested_stability = "unstable"
        suggested_confidence = "low"

    return {
        "warnings": warnings,
        "suggested_stability": suggested_stability,
        "suggested_confidence": suggested_confidence,
        "n_boundary": n_boundary,
        "n_min": n_min,
        "gate_pass": gate_pass,
        "boundary_fraction": as_float(summary.get("boundary_fraction")),
    }


class CapabilityAnalyst:
    name = "CapabilityAnalyst"

    def __init__(self, llm: DeepSeekJsonClient) -> None:
        self.llm = llm

    def run(self, cfg: PrepareConfig, agent_input: CapabilityAnalystInput) -> CapabilityReport:
        summary = dict(agent_input.calibrated_capability_estimates)
        theta_history = [float(value) for value in summary.get("theta_history", [])]
        theta = theta_history[-1] if theta_history else None
        theta_change = theta_history[-1] - theta_history[-2] if len(theta_history) >= 2 else None
        boundary_scene_ids = string_list(summary.get("frontier_scene_ids", []), max_items=1000000)
        n_boundary = as_int(summary.get("n_boundary"))
        n_min = as_int(summary.get("n_min"))
        deterministic_signals = deterministic_capability_signals(summary, theta_history)
        context = {
            "stage": cfg.stage,
            "checkpoint": str(cfg.previous_checkpoint),
            "boundary_summary": summary,
            "theta_history": theta_history,
            "theta": theta,
            "theta_change": theta_change,
            "boundary_range": [as_float(summary.get("zpd_p_lo")), as_float(summary.get("zpd_p_hi"))],
            "boundary_count": n_boundary,
            "n_min": n_min,
            "boundary_scene_id_sample": boundary_scene_ids[:50],
            "boundary_scene_id_count": len(boundary_scene_ids),
            "deterministic_rule_signals": deterministic_signals,
            "policy_evaluation_results": dict(agent_input.policy_evaluation_results),
            "stage_performance_history": list(agent_input.stage_performance_history),
            "memory_snapshot": dict(agent_input.cross_stage_memory),
            "revision_feedback": dict(agent_input.revision_feedback or {}),
        }
        llm_output = self.llm.call_json(
            stage=cfg.stage,
            agent_name=self.name,
            system_prompt=CAPABILITY_SYSTEM_PROMPT,
            context=context,
            input_summary={
                "stage": cfg.stage,
                "theta": theta,
                "theta_change": theta_change,
                "boundary_count": n_boundary,
                "n_min": n_min,
                "gate_pass": summary.get("gate_pass"),
            },
            validator=self.validate,
        )

        return CapabilityReport(
            stage=cfg.stage,
            checkpoint=str(cfg.previous_checkpoint),
            policy_capability=theta,
            capability_change=theta_change,
            capability_history=theta_history,
            frontier_probability_range=[as_float(summary.get("zpd_p_lo")), as_float(summary.get("zpd_p_hi"))],
            frontier_scene_ids=boundary_scene_ids,
            frontier_scene_count=n_boundary,
            minimum_frontier_scene_count=n_min,
            stability=llm_output["stability"],
            confidence=llm_output["confidence"],
            progress_summary=llm_output["progress_summary"],
            persistent_limitations=llm_output["persistent_limitations"],
            emerging_limitations=llm_output["emerging_limitations"],
            evidence_issues=llm_output["evidence_issues"],
            warnings=llm_output["warnings"],
            rationale=llm_output["rationale"],
            llm_model=cfg.deepseek.model,
            deterministic_sources={
                "boundary_summary": str(cfg.boundary_dir / "boundary_summary.json"),
                "boundary_npz": str(cfg.boundary_dir / "navtest_boundary.npz"),
                "memory": str(cfg.memory_path),
            },
        )

    def validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        require_choice(payload, "stability", STABILITY_CHOICES)
        require_choice(payload, "confidence", CONFIDENCE_CHOICES)
        warnings = require_list(payload, "warnings")
        if not all(isinstance(item, str) for item in warnings):
            raise ValueError("warnings must contain only strings")
        require_string(payload, "rationale")
        progress_summary = require_string(payload, "progress_summary")
        structured_lists: Dict[str, List[str]] = {}
        for key in ("persistent_limitations", "emerging_limitations", "evidence_issues"):
            values = require_list(payload, key)
            if not all(isinstance(item, str) for item in values):
                raise ValueError(f"{key} must contain only strings")
            structured_lists[key] = values
        return {
            "stability": payload["stability"],
            "confidence": payload["confidence"],
            "warnings": warnings,
            "rationale": payload["rationale"],
            "progress_summary": progress_summary,
            **structured_lists,
        }


class FrontierMapper:
    name = "FrontierMapper"

    def __init__(self, llm: DeepSeekJsonClient) -> None:
        self.llm = llm

    def run(
        self,
        cfg: PrepareConfig,
        agent_input: FrontierMapperInput,
    ) -> FrontierReport:
        capability_report = agent_input.capability_report
        summary = read_json(cfg.retrieval_dir / "sphere_summary.json")
        selection = load_npz(
            cfg.retrieval_dir / "train_selection.npz",
            ["tokens", "selected", "nearest_cluster", "nearest_distance", "radii", "centers", "boundary_tokens", "boundary_cluster", "boundary_covered"],
        )
        train_tokens = token_list(np.asarray(selection["tokens"]))
        selected = np.asarray(selection["selected"], dtype=bool)
        nearest_cluster = np.asarray(selection["nearest_cluster"], dtype=np.int64)
        nearest_distance = np.asarray(selection["nearest_distance"], dtype=np.float32)
        boundary_tokens = token_list(np.asarray(selection["boundary_tokens"]))
        boundary_cluster = np.asarray(selection["boundary_cluster"], dtype=np.int64)
        boundary_covered = np.asarray(selection["boundary_covered"], dtype=bool)
        centers = np.asarray(selection["centers"], dtype=np.float32)
        radii = np.asarray(selection["radii"], dtype=np.float32)

        regions: List[Dict[str, Any]] = []
        selected_tokens = np.asarray(train_tokens, dtype=str)[selected]
        for cluster in summary.get("clusters", []):
            cluster_id = as_int(cluster.get("cluster"))
            train_mask = selected & (nearest_cluster == cluster_id)
            cluster_train_tokens = np.asarray(train_tokens, dtype=str)[train_mask]
            cluster_boundary_tokens = [
                token for token, cid in zip(boundary_tokens, boundary_cluster.tolist()) if int(cid) == cluster_id
            ]
            distances = nearest_distance[train_mask]
            regions.append(
                {
                    "cluster": cluster_id,
                    "center": centers[cluster_id].astype(float).tolist(),
                    "radius": float(radii[cluster_id]),
                    "test_anchors": cluster_boundary_tokens,
                    "boundary_count": len(cluster_boundary_tokens),
                    "boundary_covered_count": int(np.sum(boundary_covered[boundary_cluster == cluster_id])),
                    "candidate_train_scene_ids": cluster_train_tokens[:200].astype(str).tolist(),
                    "candidate_train_count": int(cluster_train_tokens.shape[0]),
                    "mean_train_distance": float(np.mean(distances)) if distances.size else None,
                    "point_like_cluster": bool(cluster.get("point_like_cluster", False)),
                    "radius_floor_enforced": bool(cluster.get("radius_floor_enforced", False)),
                    "warnings": [],
                }
            )

        coverage = as_float(summary.get("boundary_coverage_fraction"))
        selected_fraction = as_float(summary.get("selected_train_fraction"))
        deterministic_warnings: List[str] = []
        if coverage < 0.80:
            deterministic_warnings.append(f"Boundary coverage {coverage:.4f} is below configured audit floor 0.80.")
        if selected_fraction <= 0.0:
            deterministic_warnings.append("No train scenes were selected.")
        if selected_fraction >= 0.95:
            deterministic_warnings.append("Sphere retrieval selected almost all navtrain scenes; replay may be ineffective.")
        if any(region["boundary_count"] <= as_int(cfg.values["ISOLATED_BOUNDARY_COUNT"]) for region in regions):
            deterministic_warnings.append("At least one boundary region is point-like or sparse.")

        region_context = []
        for region in regions:
            item = {key: value for key, value in region.items() if key != "candidate_train_scene_ids"}
            item["candidate_train_scene_id_sample"] = region["candidate_train_scene_ids"][:30]
            region_context.append(item)

        context = {
            "stage": cfg.stage,
            "capability_report": {
                key: value
                for key, value in capability_report.to_dict().items()
                if key != "frontier_scene_ids"
            },
            "deterministic_capability_frontier": dict(agent_input.deterministic_capability_frontier),
            "frontier_prototypes": [list(item) for item in agent_input.frontier_prototypes],
            "sphere_summary": summary,
            "region_summaries": region_context,
            "coverage": coverage,
            "selected_train_fraction": selected_fraction,
            "deterministic_rule_signals": {
                "warnings": deterministic_warnings,
                "audit_coverage_floor": 0.80,
                "selected_fraction_unreasonable_hi": 0.95,
                "isolated_boundary_count": cfg.values["ISOLATED_BOUNDARY_COUNT"],
            },
            "migration_statistics": dict(agent_input.migration_statistics),
            "feedback": dict(agent_input.revision_feedback or {}),
            "memory_snapshot": dict(agent_input.cross_stage_memory),
        }
        llm_output = self.llm.call_json(
            stage=cfg.stage,
            agent_name=self.name,
            system_prompt=BOUNDARY_SYSTEM_PROMPT,
            context=context,
            input_summary={
                "stage": cfg.stage,
                "n_regions": len(regions),
                "coverage": coverage,
                "selected_train_fraction": selected_fraction,
                "candidate_train_count": int(np.sum(selected)),
            },
            validator=lambda payload: self.validate(payload, {as_int(region["cluster"]) for region in regions}),
        )
        assessments = {
            as_int(item["cluster"]): item
            for item in llm_output.get("region_assessments", [])
        }
        for region in regions:
            assessment = assessments.get(as_int(region["cluster"]))
            if assessment:
                region["reliability"] = assessment["reliability"]
                region["warnings"] = assessment["warnings"]
            else:
                region["reliability"] = "unknown"

        relevant_ids = llm_output["curriculum_relevant_region_ids"]
        sparse_ids = llm_output["sparse_region_ids"]
        typed_regions = [
            FrontierRegion(
                region_id=as_int(region["cluster"]),
                prototype=[float(value) for value in region.get("center", [])],
                support_radius=as_float(region.get("radius")),
                frontier_scene_ids=string_list(region.get("test_anchors", []), max_items=1000000),
                frontier_scene_count=as_int(region.get("boundary_count")),
                covered_frontier_scene_count=as_int(region.get("boundary_covered_count")),
                support_scene_ids=string_list(region.get("candidate_train_scene_ids", []), max_items=1000000),
                support_scene_count=as_int(region.get("candidate_train_count")),
                mean_support_distance=region.get("mean_train_distance"),
                point_like=bool(region.get("point_like_cluster", False)),
                radius_floor_enforced=bool(region.get("radius_floor_enforced", False)),
                reliability=str(region.get("reliability", "unknown")),
                warnings=string_list(region.get("warnings", [])),
            )
            for region in regions
        ]
        return FrontierReport(
            stage=cfg.stage,
            regions=typed_regions,
            curriculum_relevant_region_ids=relevant_ids,
            sparse_region_ids=sparse_ids,
            support_summary=llm_output["support_summary"],
            coverage=coverage,
            support_scene_ids=selected_tokens.astype(str).tolist(),
            support_scene_count=int(np.sum(selected)),
            support_manifest=str(cfg.retrieval_dir / "train_manifest.csv"),
            support_tokens_file=str(cfg.retrieval_dir / "selected_train_tokens.txt"),
            coverage_statistics=dict(agent_input.coverage_statistics),
            migration_statistics=dict(agent_input.migration_statistics),
            retrieval_statistics={
                "mean_covering_spheres_per_selected_train": summary.get("mean_covering_spheres_per_selected_train"),
                "multi_sphere_selected_fraction": summary.get("multi_sphere_selected_fraction"),
                "selected_train_fraction": selected_fraction,
            },
            confidence=llm_output["confidence"],
            warnings=llm_output["warnings"],
            rationale=llm_output["rationale"],
            llm_model=cfg.deepseek.model,
            deterministic_sources={
                "sphere_summary": str(cfg.retrieval_dir / "sphere_summary.json"),
                "train_selection": str(cfg.retrieval_dir / "train_selection.npz"),
            },
        )

    def validate(self, payload: Dict[str, Any], valid_clusters: set[int]) -> Dict[str, Any]:
        require_choice(payload, "confidence", CONFIDENCE_CHOICES)
        warnings = require_list(payload, "warnings")
        if not all(isinstance(item, str) for item in warnings):
            raise ValueError("warnings must contain only strings")
        assessments = require_list(payload, "region_assessments")
        normalized: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for item in assessments:
            if not isinstance(item, dict):
                raise ValueError("region_assessments entries must be objects")
            cluster = as_int(item.get("cluster"), default=-1)
            if cluster not in valid_clusters:
                raise ValueError(f"region assessment uses unknown cluster {cluster}")
            reliability = require_choice(item, "reliability", {"reliable", "sparse", "over_broad", "point_like", "unknown"})
            item_warnings = require_list(item, "warnings")
            if not all(isinstance(warning, str) for warning in item_warnings):
                raise ValueError("region warnings must contain only strings")
            seen.add(cluster)
            normalized.append({"cluster": cluster, "reliability": reliability, "warnings": item_warnings})
        require_string(payload, "rationale")
        support_summary = require_string(payload, "support_summary")
        relevant = [as_int(item, -1) for item in require_list(payload, "curriculum_relevant_region_ids")]
        sparse = [as_int(item, -1) for item in require_list(payload, "sparse_region_ids")]
        if any(item not in valid_clusters for item in relevant + sparse):
            raise ValueError("frontier region ID lists contain unknown clusters")
        return {
            "confidence": payload["confidence"],
            "warnings": warnings,
            "region_assessments": normalized,
            "curriculum_relevant_region_ids": relevant,
            "sparse_region_ids": sparse,
            "support_summary": support_summary,
            "rationale": payload["rationale"],
        }


class CurriculumPlanner:
    name = "CurriculumPlanner"

    def __init__(self, llm: DeepSeekJsonClient) -> None:
        self.llm = llm

    def run(
        self,
        cfg: PrepareConfig,
        agent_input: CurriculumPlannerInput,
    ) -> CandidateCurriculumPlan:
        capability_report = agent_input.capability_report
        frontier_report = agent_input.frontier_report
        boundary_weight_default = as_float(cfg.values["BOUNDARY_MASS"])
        replay_weight_default = 1.0 - boundary_weight_default
        regions = frontier_report.regions
        counts = [max(0, region.support_scene_count) for region in regions]
        total = sum(counts)
        default_region_weights: List[Dict[str, Any]] = []
        for region, count in zip(regions, counts):
            raw_weight = (count / total) if total > 0 else (1.0 / max(len(regions), 1))
            weight = min(raw_weight, cfg.max_region_weight)
            default_region_weights.append(
                {
                    "cluster": region.region_id,
                    "weight": float(weight),
                    "candidate_train_count": count,
                    "boundary_count": region.frontier_scene_count,
                    "radius": region.support_radius,
                }
            )
        weight_sum = sum(as_float(item["weight"]) for item in default_region_weights)
        if weight_sum > 0.0:
            for item in default_region_weights:
                item["weight"] = float(as_float(item["weight"]) / weight_sum)

        target_max_epochs = cfg.max_epochs
        if target_max_epochs is None:
            target_max_epochs = cfg.warmup_epochs + cfg.stage * cfg.stage_epochs

        deterministic_warnings: List[str] = []
        if capability_report.confidence == "low":
            deterministic_warnings.append("Capability report confidence is low; human review should inspect boundary_summary before training.")
        if frontier_report.confidence == "low":
            deterministic_warnings.append("Boundary localization confidence is low; human review should inspect sphere_summary before training.")
        if agent_input.revision_feedback:
            deterministic_warnings.extend(str(item) for item in agent_input.revision_feedback.get("actionable_feedback", []))

        valid_clusters = {as_int(item.get("cluster")) for item in default_region_weights}
        context = {
            "stage": cfg.stage,
            "capability_report": {key: value for key, value in capability_report.to_dict().items() if key != "frontier_scene_ids"},
            "frontier_report": {
                key: value
                for key, value in frontier_report.to_dict().items()
                if key != "support_scene_ids"
            },
            "deterministic_defaults": {
                "source_checkpoint": str(cfg.previous_checkpoint),
                "training_epochs": cfg.stage_epochs,
                "max_epochs": int(target_max_epochs),
                "boundary_mass": boundary_weight_default,
                "replay_mass": replay_weight_default,
                "region_weights": default_region_weights,
                "train_manifest": str(cfg.retrieval_dir / "train_manifest.csv"),
                "selected_train_tokens": str(cfg.retrieval_dir / "selected_train_tokens.txt"),
                "tbat_epoch_size": cfg.tbat_epoch_size,
                "tbat_rotation_fraction": cfg.tbat_rotation_fraction,
                "tbat_sampler_seed": cfg.tbat_sampler_seed,
            },
            "allowed_ranges": {
                "min_boundary_weight": cfg.min_boundary_weight,
                "max_boundary_weight": cfg.max_boundary_weight,
                "max_region_weight": cfg.max_region_weight,
            },
            "deterministic_rule_signals": {
                "warnings": deterministic_warnings,
                "manifest_already_generated": True,
                "boundary_weight_must_match_manifest": boundary_weight_default,
            },
            "previous_curriculum": agent_input.previous_curriculum,
            "previous_outcome": agent_input.previous_outcome,
            "retention_pool": dict(agent_input.retention_pool),
            "feedback": dict(agent_input.revision_feedback or {}),
            "memory_snapshot": dict(agent_input.cross_stage_memory),
        }
        llm_output = self.llm.call_json(
            stage=cfg.stage,
            agent_name=self.name,
            system_prompt=CURRICULUM_SYSTEM_PROMPT,
            context=context,
            input_summary={
                "stage": cfg.stage,
                "boundary_weight_default": boundary_weight_default,
                "n_regions": len(default_region_weights),
                "target_max_epochs": target_max_epochs,
                "feedback_status": agent_input.revision_feedback.get("status") if agent_input.revision_feedback else None,
            },
            validator=lambda payload: self.validate(payload, cfg, boundary_weight_default, valid_clusters),
        )
        region_weight_by_cluster = {as_int(item["cluster"]): as_float(item["weight"]) for item in llm_output["region_weights"]}
        region_weights: List[Dict[str, Any]] = []
        for item in default_region_weights:
            enriched = dict(item)
            enriched["weight"] = float(region_weight_by_cluster[as_int(item["cluster"])])
            region_weights.append(enriched)
        boundary_weight = as_float(llm_output["boundary_weight"])
        replay_weight = as_float(llm_output["replay_weight"])

        return CandidateCurriculumPlan(
            stage=cfg.stage,
            source_checkpoint=str(cfg.previous_checkpoint),
            budget=agent_input.current_training_budget,
            frontier_allocation=float(boundary_weight),
            retention_replay_allocation=float(replay_weight),
            region_allocations=[
                RegionAllocation(
                    region_id=as_int(item["cluster"]),
                    weight=as_float(item["weight"]),
                    support_scene_count=as_int(item["candidate_train_count"]),
                    frontier_scene_count=as_int(item["boundary_count"]),
                    support_radius=item.get("radius"),
                )
                for item in region_weights
            ],
            support_manifest=str(cfg.retrieval_dir / "train_manifest.csv"),
            support_tokens_file=str(cfg.retrieval_dir / "selected_train_tokens.txt"),
            support_scene_count=frontier_report.support_scene_count,
            training_configuration={
                "STAGE": str(cfg.stage),
                "CBAT_RUN_NAME": cfg.run_name,
                "MAX_EPOCHS": str(int(target_max_epochs)),
                "TBAT_MANIFEST": str(cfg.retrieval_dir / "train_manifest.csv"),
                "TBAT_ROTATION_FRACTION": str(cfg.tbat_rotation_fraction),
                "TBAT_EPOCH_SIZE": str(cfg.tbat_epoch_size),
                "TBAT_SAMPLER_SEED": str(cfg.tbat_sampler_seed),
                "TBAT_TRAINING_PROPOSAL": str(cfg.agent_dir / "training_proposal.json"),
            },
            rationale=llm_output["rationale"],
            warnings=llm_output["warnings"],
            llm_model=cfg.deepseek.model,
            created_at=utc_now(),
        )

    def validate(self, payload: Dict[str, Any], cfg: PrepareConfig, boundary_weight_default: float, valid_clusters: set[int]) -> Dict[str, Any]:
        boundary_weight = as_float(payload.get("boundary_weight"), default=float("nan"))
        replay_weight = as_float(payload.get("replay_weight"), default=float("nan"))
        if not math.isfinite(boundary_weight) or not math.isfinite(replay_weight):
            raise ValueError("boundary_weight and replay_weight must be finite numbers")
        if abs(boundary_weight - boundary_weight_default) > 1e-6:
            raise ValueError("boundary_weight must match deterministic manifest BOUNDARY_MASS for this prepared run")
        if not cfg.min_boundary_weight <= boundary_weight <= cfg.max_boundary_weight:
            raise ValueError("boundary_weight is outside configured range")
        if abs(boundary_weight + replay_weight - 1.0) > 1e-6:
            raise ValueError("boundary_weight and replay_weight must sum to 1")
        region_weights = require_list(payload, "region_weights")
        normalized: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for item in region_weights:
            if not isinstance(item, dict):
                raise ValueError("region_weights entries must be objects")
            cluster = as_int(item.get("cluster"), default=-1)
            if cluster not in valid_clusters:
                raise ValueError(f"region weight uses unknown cluster {cluster}")
            weight = as_float(item.get("weight"), default=float("nan"))
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("region weight must be non-negative finite number")
            if weight > cfg.max_region_weight + 1e-6:
                raise ValueError("region weight exceeds MAX_REGION_WEIGHT")
            seen.add(cluster)
            normalized.append({"cluster": cluster, "weight": float(weight)})
        if valid_clusters and seen != valid_clusters:
            raise ValueError(f"region_weights must include exactly all clusters {sorted(valid_clusters)}")
        total = sum(as_float(item["weight"]) for item in normalized)
        if valid_clusters and abs(total - 1.0) > 1e-5:
            raise ValueError("region_weights must sum to 1")
        warnings = require_list(payload, "warnings")
        if not all(isinstance(item, str) for item in warnings):
            raise ValueError("warnings must contain only strings")
        rationale = require_list(payload, "rationale")
        if not all(isinstance(item, str) for item in rationale):
            raise ValueError("rationale must contain only strings")
        return {
            "boundary_weight": float(boundary_weight),
            "replay_weight": float(replay_weight),
            "region_weights": normalized,
            "warnings": warnings,
            "rationale": rationale,
        }


class _VerifierDecisionEngine:
    name = "CurriculumVerifier"

    def __init__(self, llm: DeepSeekJsonClient) -> None:
        self.llm = llm

    def run(
        self,
        cfg: PrepareConfig,
        memory: Dict[str, Any],
        capability_report: Dict[str, Any],
        boundary_report: Dict[str, Any],
        training_proposal: Dict[str, Any],
        revision_round: int,
    ) -> Dict[str, Any]:
        guard_report = deterministic_audit_guard(cfg, capability_report, boundary_report, training_proposal, revision_round)
        if not cfg.audit_agent_enabled:
            final_status = str(guard_report.get("hard_status") or guard_report.get("suggested_status") or "PASS")
            final_reasons = list(guard_report.get("reasons", []))
            final_feedback = list(guard_report.get("actionable_feedback", []))
            if final_status == "PASS" and "pass_with_warnings" in final_reasons:
                final_feedback.append("Audit LLM is disabled; deterministic guard passed with warnings for human review.")
            if revision_round >= cfg.max_revision_rounds and final_status not in {"PASS", "ABORT"}:
                final_reasons.append("max_revision_rounds_exhausted")
                final_feedback.append("Revision budget is exhausted; stop instead of returning another targeted rollback.")
                final_status = "ABORT"
            return {
                "schema_version": "audit_report.v1",
                "agent": self.name,
                "stage": cfg.stage,
                "revision_round": revision_round,
                "status": final_status,
                "llm_status": "DISABLED",
                "reasons": sorted(set(final_reasons)),
                "actionable_feedback": final_feedback,
                "revision_plan": {},
                "llm_rationale": "CurriculumVerifier LLM review is disabled; deterministic guardrails are the source of truth.",
                "deterministic_guard_report": guard_report,
                "llm_model": None,
                "checked_at": utc_now(),
            }
        context = {
            "stage": cfg.stage,
            "revision_round": revision_round,
            "max_revision_rounds": cfg.max_revision_rounds,
            "memory_snapshot": memory_snapshot(memory),
            "capability_report": {key: value for key, value in capability_report.items() if key != "boundary_scene_ids"},
            "boundary_report": {key: value for key, value in boundary_report.items() if key != "candidate_train_scene_ids"},
            "training_proposal": training_proposal,
            "deterministic_guard_report": guard_report,
        }
        llm_output = self.llm.call_json(
            stage=cfg.stage,
            agent_name=self.name,
            system_prompt=AUDITOR_SYSTEM_PROMPT,
            context=context,
            input_summary={
                "stage": cfg.stage,
                "revision_round": revision_round,
                "guard_suggested_status": guard_report["suggested_status"],
                "guard_reasons": guard_report["reasons"],
            },
            validator=self.validate,
        )
        final_status, final_reasons, final_feedback = apply_hard_audit_guards(guard_report, llm_output)
        if revision_round >= cfg.max_revision_rounds and final_status not in {"PASS", "ABORT"}:
            final_reasons.append("max_revision_rounds_exhausted")
            final_feedback.append("Revision budget is exhausted; stop instead of returning another targeted rollback.")
            final_status = "ABORT"
        return {
            "schema_version": "audit_report.v1",
            "agent": self.name,
            "stage": cfg.stage,
            "revision_round": revision_round,
            "status": final_status,
            "llm_status": llm_output["status"],
            "reasons": sorted(set(final_reasons)),
            "actionable_feedback": final_feedback,
            "revision_plan": llm_output.get("revision_plan", {}),
            "llm_rationale": llm_output["rationale"],
            "deterministic_guard_report": guard_report,
            "llm_model": cfg.deepseek.model,
            "checked_at": utc_now(),
        }

    def validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        require_choice(payload, "status", AUDIT_ACTIONS)
        reasons = require_list(payload, "reasons")
        feedback = require_list(payload, "actionable_feedback")
        if not all(isinstance(item, str) for item in reasons):
            raise ValueError("reasons must contain only strings")
        if not all(isinstance(item, str) for item in feedback):
            raise ValueError("actionable_feedback must contain only strings")
        revision_plan = payload.get("revision_plan", {})
        if revision_plan is None:
            revision_plan = {}
        if not isinstance(revision_plan, dict):
            raise ValueError("revision_plan must be an object when provided")
        normalized_plan: Dict[str, Any] = {}
        if revision_plan:
            target = normalize_revision_target(revision_plan.get("target_agent", payload["status"]))
            if target is None:
                raise ValueError(f"revision_plan.target_agent must be one of {sorted(REVISION_TARGETS)}")
            config_overrides = revision_plan.get("config_overrides", {})
            if config_overrides is None:
                config_overrides = {}
            if not isinstance(config_overrides, dict):
                raise ValueError("revision_plan.config_overrides must be an object")
            if config_overrides or bool(revision_plan.get("rerun_deterministic", False)):
                raise ValueError("CurriculumVerifier may not modify or rerun deterministic frontier algorithms")
            notes = revision_plan.get("notes", [])
            if notes is None:
                notes = []
            if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
                raise ValueError("revision_plan.notes must be a list of strings")
            normalized_plan = {
                "target_agent": target,
                "rerun_deterministic": False,
                "config_overrides": {},
                "notes": notes,
            }
        require_string(payload, "rationale")
        return {
            "status": payload["status"],
            "reasons": reasons,
            "actionable_feedback": feedback,
            "revision_plan": normalized_plan,
            "rationale": payload["rationale"],
        }


class CurriculumVerifier:
    """Typed verifier boundary over deterministic guards and optional LLM review."""

    name = "CurriculumVerifier"

    def __init__(self, llm: DeepSeekJsonClient) -> None:
        self.engine = _VerifierDecisionEngine(llm)
        self.last_audit_report: Dict[str, Any] = {}

    def run(self, cfg: PrepareConfig, agent_input: CurriculumVerifierInput) -> VerifierDecision:
        capability = agent_input.capability_report
        frontier = agent_input.frontier_report
        legacy_capability = {
            "boundary_count": capability.frontier_scene_count,
            "n_min": capability.minimum_frontier_scene_count,
            "boundary_scene_ids": capability.frontier_scene_ids,
            "warnings": capability.warnings + capability.evidence_issues,
            "confidence": capability.confidence,
        }
        legacy_frontier = {
            "coverage": frontier.coverage,
            "warnings": frontier.warnings,
            "confidence": frontier.confidence,
            "candidate_train_count": frontier.support_scene_count,
        }
        audit = self.engine.run(
            cfg,
            dict(agent_input.cross_stage_memory),
            legacy_capability,
            legacy_frontier,
            agent_input.candidate_plan.to_legacy_training_proposal(),
            agent_input.revision_round,
        )
        audit["data_split_information"] = dict(agent_input.data_split_information)
        audit["platform_validation_results"] = dict(agent_input.platform_validation_results)
        self.last_audit_report = audit
        if audit["status"] == "PASS":
            return VerifiedCurriculumPlan(
                candidate_plan=agent_input.candidate_plan,
                verification_checks=dict(audit["deterministic_guard_report"]),
                verifier_rationale=str(audit.get("llm_rationale", "")),
                warnings=string_list(audit.get("actionable_feedback", [])),
                verified_at=utc_now(),
            )

        target_by_status = {
            "REVISE_A1": RevisionTarget.CAPABILITY_ANALYST,
            "REVISE_A2": RevisionTarget.FRONTIER_MAPPER,
            "REVISE_A3": RevisionTarget.CURRICULUM_PLANNER,
        }
        target = target_by_status.get(str(audit["status"]), RevisionTarget.CURRICULUM_PLANNER)
        raw_plan = audit.get("revision_plan", {}) if isinstance(audit.get("revision_plan", {}), dict) else {}
        notes = string_list(raw_plan.get("notes", []))
        if audit["status"] == "ABORT":
            notes.append("Terminal deterministic verification failure.")
        return RevisionRequest(
            stage=cfg.stage,
            target=target,
            reason_codes=string_list(audit.get("reasons", [])),
            actionable_feedback=string_list(audit.get("actionable_feedback", [])),
            revision_round=agent_input.revision_round,
            max_revisions=agent_input.deterministic_constraints.maximum_revision_rounds,
            # Agents may revise interpretations and curriculum decisions, but
            # never change capability/frontier/retrieval algorithms or knobs.
            rerun_deterministic=False,
            config_overrides={},
            notes=notes,
            terminal=audit["status"] == "ABORT",
        )


def deterministic_audit_guard(
    cfg: PrepareConfig,
    capability_report: Dict[str, Any],
    boundary_report: Dict[str, Any],
    training_proposal: Dict[str, Any],
    revision_round: int,
) -> Dict[str, Any]:
        reasons: List[str] = []
        feedback: List[str] = []
        suggested_status = "PASS"
        hard_status: Optional[str] = None

        boundary_count = as_int(capability_report.get("boundary_count"))
        n_min = as_int(capability_report.get("n_min"))
        if boundary_count <= 0:
            reasons.append("boundary_empty")
            suggested_status = "ABORT"
            hard_status = "ABORT"
        elif boundary_count < n_min:
            reasons.append("boundary_sparse")
            feedback.append("Check ZPD band and evaluation history; deterministic selection did not reach configured minimum.")
            suggested_status = "REVISE_A1"

        boundary_fraction = None
        boundary_summary = cfg.boundary_dir / "boundary_summary.json"
        if boundary_summary.is_file():
            boundary_fraction = as_float(read_json(boundary_summary).get("boundary_fraction"))
            if boundary_fraction > 0.55:
                reasons.append("boundary_diffuse")
                feedback.append("Boundary fraction is too broad; inspect ZPD config before using the proposal.")
                suggested_status = "REVISE_A1" if suggested_status == "PASS" else suggested_status

        coverage = as_float(boundary_report.get("coverage"))
        if coverage < 0.80:
            reasons.append("sphere_under_supported")
            feedback.append("Boundary coverage is below audit floor; inspect sphere radius/retrieval config.")
            suggested_status = "REVISE_A2" if suggested_status == "PASS" else suggested_status

        manifest_path = Path(training_proposal["selected_scene_ids_or_sampling_file"]["train_manifest"])
        manifest_check = validate_manifest(manifest_path)
        if not manifest_check["valid"]:
            reasons.extend(manifest_check["errors"])
            feedback.append("Generated manifest is invalid for TBATSampler.")
            suggested_status = "REVISE_A2" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A2"
        planned_region_ids = {as_int(item.get("cluster"), -1) for item in training_proposal.get("region_weights", [])}
        manifest_region_check = validate_manifest_regions(manifest_path, planned_region_ids)
        if not manifest_region_check["valid"]:
            reasons.extend(manifest_region_check["errors"])
            feedback.append("Retrieved support regions are incompatible with harness curriculum materialization.")
            suggested_status = "REVISE_A2" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A2"

        boundary_weight = as_float(training_proposal.get("boundary_weight"))
        replay_weight = as_float(training_proposal.get("replay_weight"))
        if not cfg.min_boundary_weight <= boundary_weight <= cfg.max_boundary_weight:
            reasons.append("boundary_weight_out_of_config_range")
            suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A3"
        if abs(boundary_weight + replay_weight - 1.0) > 1e-6:
            reasons.append("boundary_replay_weights_do_not_sum_to_one")
            suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A3"
        for item in training_proposal.get("region_weights", []):
            if as_float(item.get("weight")) > cfg.max_region_weight:
                reasons.append("region_weight_over_concentrated")
                suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
                hard_status = hard_status or "REVISE_A3"
                break
        region_weights = [as_float(item.get("weight"), default=float("nan")) for item in training_proposal.get("region_weights", [])]
        if region_weights and (not all(math.isfinite(value) and value >= 0.0 for value in region_weights) or abs(sum(region_weights) - 1.0) > 1e-5):
            reasons.append("region_weights_invalid_or_not_normalized")
            feedback.append("Normalize all frontier-region allocations to one within the configured action space.")
            suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A3"

        expected_max_epochs = cfg.max_epochs if cfg.max_epochs is not None else cfg.warmup_epochs + cfg.stage * cfg.stage_epochs
        if as_int(training_proposal.get("training_epochs"), default=-1) != cfg.stage_epochs:
            reasons.append("training_epoch_budget_mismatch")
            suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A3"
        if as_int(training_proposal.get("max_epochs"), default=-1) != expected_max_epochs:
            reasons.append("maximum_epoch_budget_mismatch")
            suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A3"
        if str(training_proposal.get("source_checkpoint")) != str(cfg.previous_checkpoint):
            reasons.append("source_checkpoint_mismatch")
            suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A3"
        script_inputs = training_proposal.get("training_script_inputs", {})
        required_inputs = {
            "STAGE", "CBAT_RUN_NAME", "MAX_EPOCHS", "TBAT_MANIFEST",
            "TBAT_ROTATION_FRACTION", "TBAT_EPOCH_SIZE", "TBAT_SAMPLER_SEED",
            "TBAT_TRAINING_PROPOSAL",
        }
        missing_script_inputs = sorted(required_inputs - set(script_inputs)) if isinstance(script_inputs, dict) else sorted(required_inputs)
        if missing_script_inputs:
            reasons.append("training_configuration_incompatible")
            feedback.append(f"Missing platform training inputs: {', '.join(missing_script_inputs)}")
            suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A3"
        expected_manifest = cfg.retrieval_dir / "train_manifest.csv"
        if manifest_path != expected_manifest:
            reasons.append("manifest_path_mismatch")
            suggested_status = "REVISE_A3" if suggested_status == "PASS" else suggested_status
            hard_status = hard_status or "REVISE_A3"

        leakage = train_test_leakage(manifest_path, set(capability_report.get("boundary_scene_ids", [])))
        if leakage:
            reasons.append("train_test_token_overlap")
            feedback.append(f"Found train/test token overlap examples: {', '.join(leakage[:5])}")
            suggested_status = "ABORT"
            hard_status = "ABORT"

        if revision_round >= cfg.max_revision_rounds and suggested_status != "PASS":
            suggested_status = "ABORT"
            hard_status = "ABORT"
            reasons.append("max_revision_rounds_exhausted")

        if suggested_status == "PASS" and (capability_report.get("warnings") or boundary_report.get("warnings")):
            reasons.append("pass_with_warnings")

        return {
            "suggested_status": suggested_status,
            "hard_status": hard_status,
            "reasons": sorted(set(reasons)),
            "actionable_feedback": feedback,
            "manifest_check": manifest_check,
            "manifest_region_check": manifest_region_check,
            "boundary_fraction": boundary_fraction,
            "coverage": coverage,
            "boundary_weight": boundary_weight,
            "replay_weight": replay_weight,
            "expected_max_epochs": expected_max_epochs,
            "missing_training_configuration_fields": missing_script_inputs,
            "leakage_examples": leakage,
        }


def apply_hard_audit_guards(guard_report: Dict[str, Any], llm_output: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    status = str(llm_output["status"])
    reasons = list(llm_output.get("reasons", [])) + list(guard_report.get("reasons", []))
    feedback = list(llm_output.get("actionable_feedback", [])) + list(guard_report.get("actionable_feedback", []))
    hard_status = guard_report.get("hard_status")
    if hard_status:
        status = str(hard_status)
        reasons.append(f"hard_guard_override_from_llm_{llm_output['status']}_to_{hard_status}")
    elif status == "PASS" and guard_report.get("suggested_status") != "PASS":
        status = str(guard_report["suggested_status"])
        reasons.append(f"deterministic_guard_override_from_llm_PASS_to_{status}")
    return status, reasons, feedback


def normalize_revision_target(value: Any, fallback: Optional[str] = None) -> Optional[str]:
    text = str(value or "").strip().upper().replace("-", "_")
    aliases = {
        "A1": "REVISE_A1",
        "AGENT1": "REVISE_A1",
        "AGENT_1": "REVISE_A1",
        "CAPABILITY": "REVISE_A1",
        "CAPABILITY_DIAGNOSIS": "REVISE_A1",
        "A2": "REVISE_A2",
        "AGENT2": "REVISE_A2",
        "AGENT_2": "REVISE_A2",
        "BOUNDARY": "REVISE_A2",
        "RETRIEVAL": "REVISE_A2",
        "BOUNDARY_RETRIEVAL": "REVISE_A2",
        "A3": "REVISE_A3",
        "AGENT3": "REVISE_A3",
        "AGENT_3": "REVISE_A3",
        "CURRICULUM": "REVISE_A3",
    }
    if text in REVISION_TARGETS:
        return text
    if text in aliases:
        return aliases[text]
    return fallback if fallback in REVISION_TARGETS else None


def validate_manifest(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"valid": False, "errors": [], "path": str(path), "selected_count": 0, "replay_count": 0}
    if not path.is_file():
        result["errors"].append("manifest_missing")
        return result
    weights: List[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"token", "selected", "train_weight"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            result["errors"].append("manifest_missing_required_columns")
            return result
        for row in reader:
            selected = row.get("selected") == "1"
            weight = as_float(row.get("train_weight"), default=-1.0)
            if not math.isfinite(weight) or weight < 0.0:
                result["errors"].append("manifest_bad_weight")
            weights.append(weight)
            if selected:
                result["selected_count"] += 1
            else:
                result["replay_count"] += 1
    if result["selected_count"] <= 0:
        result["errors"].append("manifest_no_selected_tokens")
    if result["replay_count"] <= 0:
        result["errors"].append("manifest_no_replay_tokens")
    if weights and abs(sum(weights) - 1.0) > 1e-3:
        result["errors"].append("manifest_weight_sum_not_one")
    result["weight_sum"] = float(sum(weights))
    result["valid"] = not result["errors"]
    return result


def validate_manifest_regions(path: Path, expected_region_ids: set[int]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"valid": False, "errors": [], "region_ids": []}
    if not path.is_file():
        result["errors"].append("manifest_missing")
        return result
    selected_region_mass: Dict[int, float] = {}
    replay_mass = 0.0
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"selected", "nearest_cluster", "train_weight"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            result["errors"].append("manifest_missing_region_columns")
            return result
        for row in reader:
            weight = as_float(row.get("train_weight"), default=-1.0)
            if row.get("selected") == "1":
                region_id = as_int(row.get("nearest_cluster"), default=-1)
                selected_region_mass[region_id] = selected_region_mass.get(region_id, 0.0) + max(weight, 0.0)
            else:
                replay_mass += max(weight, 0.0)
    actual_region_ids = set(selected_region_mass)
    result["region_ids"] = sorted(actual_region_ids)
    if actual_region_ids != expected_region_ids:
        result["errors"].append("manifest_region_ids_do_not_match_plan")
    if any(mass <= 0.0 for mass in selected_region_mass.values()):
        result["errors"].append("manifest_region_has_zero_source_mass")
    if replay_mass <= 0.0:
        result["errors"].append("manifest_replay_has_zero_source_mass")
    result["valid"] = not result["errors"]
    return result


def train_test_leakage(manifest_path: Path, boundary_tokens: set[str]) -> List[str]:
    if not manifest_path.is_file() or not boundary_tokens:
        return []
    overlaps: List[str] = []
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            token = str(row.get("token", ""))
            if token in boundary_tokens:
                overlaps.append(token)
                if len(overlaps) >= 20:
                    break
    return overlaps


def capability_report_artifact(report: CapabilityReport) -> Dict[str, Any]:
    """New report plus read-only aliases used by historical analysis scripts."""
    payload = report.to_dict()
    payload.update(
        {
            "theta": report.policy_capability,
            "theta_change": report.capability_change,
            "theta_history": report.capability_history,
            "boundary_range": report.frontier_probability_range,
            "boundary_scene_ids": report.frontier_scene_ids,
            "boundary_count": report.frontier_scene_count,
            "n_min": report.minimum_frontier_scene_count,
        }
    )
    return payload


def legacy_boundary_report_artifact(report: FrontierReport) -> Dict[str, Any]:
    return {
        "schema_version": "boundary_report.v1",
        "agent": report.agent,
        "stage": report.stage,
        "boundary_regions": [
            {
                "cluster": region.region_id,
                "center": region.prototype,
                "radius": region.support_radius,
                "test_anchors": region.frontier_scene_ids,
                "boundary_count": region.frontier_scene_count,
                "boundary_covered_count": region.covered_frontier_scene_count,
                "candidate_train_scene_ids": region.support_scene_ids,
                "candidate_train_count": region.support_scene_count,
                "mean_train_distance": region.mean_support_distance,
                "point_like_cluster": region.point_like,
                "radius_floor_enforced": region.radius_floor_enforced,
                "reliability": region.reliability,
                "warnings": region.warnings,
            }
            for region in report.regions
        ],
        "coverage": report.coverage,
        "candidate_train_scene_ids": report.support_scene_ids[:500],
        "candidate_train_count": report.support_scene_count,
        "candidate_train_manifest": report.support_manifest,
        "selected_train_tokens_file": report.support_tokens_file,
        "distance_relevance": report.retrieval_statistics,
        "confidence": report.confidence,
        "warnings": report.warnings,
        "rationale": report.rationale,
        "llm_model": report.llm_model,
        "deterministic_sources": report.deterministic_sources,
    }


def write_proposal_markdown(path: Path, proposal: Dict[str, Any], audit: Dict[str, Any]) -> None:
    inputs = proposal["training_script_inputs"]
    lines = [
        "# TBAT Next-Stage Training Proposal",
        "",
        f"- audit_status: {audit['status']}",
        f"- stage: {proposal['stage']}",
        f"- source_checkpoint: `{proposal['source_checkpoint']}`",
        f"- training_epochs: {proposal['training_epochs']}",
        f"- max_epochs: {proposal['max_epochs']}",
        f"- boundary_weight: {proposal['boundary_weight']:.4f}",
        f"- replay_weight: {proposal['replay_weight']:.4f}",
        f"- train_manifest: `{proposal['selected_scene_ids_or_sampling_file']['train_manifest']}`",
        "",
        "## Manual Training Inputs",
        "",
        "```bash",
        "# Pass these as sbatch --export key-value pairs for the matching stage training script.",
    ]
    for key in [
        "STAGE",
        "CBAT_RUN_NAME",
        "MAX_EPOCHS",
        "TBAT_MANIFEST",
        "TBAT_ROTATION_FRACTION",
        "TBAT_EPOCH_SIZE",
        "TBAT_SAMPLER_SEED",
        "TBAT_TRAINING_PROPOSAL",
    ]:
        lines.append(f"{key}={inputs[key]}")
    lines.extend(["```", "", "## Region Weights", ""])
    for item in proposal.get("region_weights", []):
        lines.append(
            f"- cluster {item['cluster']}: weight={item['weight']:.6f}, "
            f"candidate_train_count={item['candidate_train_count']}, boundary_count={item['boundary_count']}"
        )
    if proposal.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in proposal["warnings"])
    if audit.get("reasons"):
        lines.extend(["", "## Audit Reasons", ""])
        lines.extend(f"- {reason}" for reason in audit["reasons"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class MultiAgentCurriculumController:
    """Orchestrate typed agents with bounded, targeted revision."""

    def __init__(self, cfg: PrepareConfig, memory: CrossStageLearningMemory, env: Dict[str, str]) -> None:
        self.cfg = cfg
        self.memory = memory
        self.env = env
        llm = DeepSeekJsonClient(cfg.deepseek, cfg.agent_dir / "llm_calls")
        self.capability_analyst = CapabilityAnalyst(llm)
        self.frontier_mapper = FrontierMapper(llm)
        self.curriculum_planner = CurriculumPlanner(llm)
        self.curriculum_verifier = CurriculumVerifier(llm)

    def _log(self, agent: str, direction: str, payload: Dict[str, Any]) -> None:
        print(f"[AGENT] stage={self.cfg.stage} agent={agent} {direction}={compact_json(payload, 1500)}", flush=True)

    def _apply_revision(self, request: RevisionRequest) -> None:
        record = request.to_dict()
        if request.config_overrides or request.rerun_deterministic:
            raise RuntimeError("Agent revision requests may not alter or rerun deterministic frontier algorithms")
        self.memory.record_revision(record)
        write_json(self.cfg.agent_dir / f"revision_request_round{request.revision_round}.json", record)
        self._log(self.name_for_target(request.target), "revision", record)

    @staticmethod
    def name_for_target(target: RevisionTarget) -> str:
        return target.value

    def run(
        self,
    ) -> Tuple[CapabilityReport, FrontierReport, CandidateCurriculumPlan, VerifierDecision, Dict[str, Any]]:
        capability_report: Optional[CapabilityReport] = None
        frontier_report: Optional[FrontierReport] = None
        candidate: Optional[CandidateCurriculumPlan] = None
        feedback: Optional[RevisionRequest] = None
        decision: Optional[VerifierDecision] = None

        for revision_round in range(self.cfg.max_revision_rounds + 1):
            if feedback is not None:
                self._apply_revision(feedback)

            if capability_report is None or (feedback and feedback.target == RevisionTarget.CAPABILITY_ANALYST):
                analyst_input = build_capability_analyst_input(self.cfg, self.memory, feedback)
                self._log(self.capability_analyst.name, "input", analyst_input.to_dict())
                capability_report = self.capability_analyst.run(self.cfg, analyst_input)
                self._log(self.capability_analyst.name, "output", capability_report.to_dict())
                frontier_report = None

            if frontier_report is None or (feedback and feedback.target == RevisionTarget.FRONTIER_MAPPER):
                mapper_input = build_frontier_mapper_input(self.cfg, self.memory, capability_report, feedback)
                self._log(self.frontier_mapper.name, "input", mapper_input.to_dict())
                frontier_report = self.frontier_mapper.run(self.cfg, mapper_input)
                self._log(self.frontier_mapper.name, "output", frontier_report.to_dict())

            planner_input = build_curriculum_planner_input(
                self.cfg, self.memory, capability_report, frontier_report, feedback
            )
            self._log(self.curriculum_planner.name, "input", planner_input.to_dict())
            candidate = self.curriculum_planner.run(self.cfg, planner_input)
            self.memory.record_proposed(self.cfg.stage, candidate.to_dict())
            self._log(self.curriculum_planner.name, "output", candidate.to_dict())

            verifier_input = build_curriculum_verifier_input(
                self.cfg, self.memory, candidate, capability_report, frontier_report, revision_round
            )
            self._log(self.curriculum_verifier.name, "input", verifier_input.to_dict())
            decision = self.curriculum_verifier.run(self.cfg, verifier_input)
            self._log(self.curriculum_verifier.name, "output", decision.to_dict())
            write_json(
                self.cfg.agent_dir / f"curriculum_verification_round{revision_round}.json",
                self.curriculum_verifier.last_audit_report,
            )
            if isinstance(decision, VerifiedCurriculumPlan):
                self.memory.record_verified(self.cfg.stage, decision.to_dict())
                break
            if decision.terminal:
                self.memory.record_revision(decision.to_dict())
                break
            feedback = decision

        if capability_report is None or frontier_report is None or candidate is None or decision is None:
            raise RuntimeError("Curriculum controller ended without complete stage outputs")
        return capability_report, frontier_report, candidate, decision, self.curriculum_verifier.last_audit_report


def run_agent_loop(
    cfg: PrepareConfig,
    memory: CrossStageLearningMemory,
    env: Dict[str, str],
) -> Tuple[CapabilityReport, FrontierReport, CandidateCurriculumPlan, VerifierDecision, Dict[str, Any]]:
    """Backward-compatible entry point delegating to the central controller."""
    return MultiAgentCurriculumController(cfg, memory, env).run()


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    validate_config(cfg)
    cfg.agent_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(cfg.project_root),
            "NAVSIM_DEVKIT_ROOT": str(cfg.navsim_devkit_root),
            "NAVSIM_EXP_ROOT": str(cfg.navsim_exp_root),
        }
    )

    memory = CrossStageLearningMemory.load(cfg.memory_path, cfg.run_name)
    update_memory_from_completed_artifacts(memory, cfg)
    memory.save(cfg.memory_path)
    write_json(cfg.agent_dir / "resolved_prepare_config.json", cfg.values)

    missing = [str(path) for path in required_inputs(cfg) if not path.is_file()]
    if missing:
        write_stop(
            cfg.agent_dir,
            "missing_required_completed_artifacts",
            missing,
            {
                "stage": cfg.stage,
                "run_name": cfg.run_name,
                "previous_checkpoint": str(cfg.previous_checkpoint),
                "eval_csv": str(cfg.eval_csv),
            },
        )
        print("[STOP] Missing required completed artifacts. See stop_report.json.", file=sys.stderr)
        sys.exit(3)

    try:
        run_deterministic_prepare(cfg, env)
        assert_deterministic_artifacts(cfg)
        capability_report, frontier_report, candidate, decision, audit = run_agent_loop(cfg, memory, env)
    except Exception as exc:
        error_payload = {
            "schema_version": "curriculum_control_error.v1",
            "stage": cfg.stage,
            "source": "MultiAgentCurriculumController",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "created_at": utc_now(),
        }
        write_json(cfg.agent_dir / "controller_error.json", error_payload)
        memory.record_error(
            stage=cfg.stage,
            source="MultiAgentCurriculumController",
            message=f"{type(exc).__name__}: {exc}",
            details={"artifact": str(cfg.agent_dir / "controller_error.json")},
        )
        memory.save(cfg.memory_path)
        print(f"[STOP] Curriculum controller failed: {exc}", file=sys.stderr)
        raise
    write_json(cfg.agent_dir / "resolved_prepare_config.json", cfg.values)
    write_json(cfg.agent_dir / "capability_report.json", capability_report_artifact(capability_report))
    write_json(cfg.agent_dir / "frontier_report.json", frontier_report.to_dict())
    write_json(cfg.agent_dir / "frontier_mapping_report.json", frontier_report.to_dict())
    write_json(cfg.agent_dir / "boundary_report.json", legacy_boundary_report_artifact(frontier_report))
    write_json(cfg.agent_dir / "curriculum_verification_report.json", audit)
    write_json(cfg.agent_dir / "audit_report.json", audit)  # Legacy artifact alias.
    write_json(cfg.agent_dir / "candidate_curriculum_plan.json", candidate.to_dict())

    memory.record_capability(
        cfg.stage,
        {
            "stage": cfg.stage,
            "policy_capability": capability_report.policy_capability,
            "capability_change": capability_report.capability_change,
            "frontier_probability_range": capability_report.frontier_probability_range,
            "frontier_scene_count": capability_report.frontier_scene_count,
            "updated_at": utc_now(),
        },
        capability_report.to_dict(),
    )
    memory.record_frontier(
        cfg.stage,
        {
            "stage": cfg.stage,
            "coverage": frontier_report.coverage,
            "support_scene_count": frontier_report.support_scene_count,
            "support_scene_ids": frontier_report.support_scene_ids,
            "migration_statistics": frontier_report.migration_statistics,
            "updated_at": utc_now(),
        },
        frontier_report.to_dict(),
    )
    # Compatibility summaries for existing experiment readers.
    memory.capability_history[str(cfg.stage)] = {
        "stage": cfg.stage,
        "theta": capability_report.policy_capability,
        "theta_change": capability_report.capability_change,
        "boundary_range": capability_report.frontier_probability_range,
        "boundary_count": capability_report.frontier_scene_count,
        "stability": capability_report.stability,
        "confidence": capability_report.confidence,
        "artifact": str(cfg.agent_dir / "capability_report.json"),
        "updated_at": utc_now(),
    }
    memory.scene_history[str(cfg.stage)] = {
        "stage": cfg.stage,
        "coverage": frontier_report.coverage,
        "candidate_train_count": frontier_report.support_scene_count,
        "artifact": str(cfg.agent_dir / "frontier_report.json"),
        "updated_at": utc_now(),
    }
    passed = isinstance(decision, VerifiedCurriculumPlan)
    memory.decision_history.append(
        {
            "stage": cfg.stage,
            "created_at": utc_now(),
            "audit_status": "PASS" if passed else "ABORT",
            "audit_reasons": audit.get("reasons", []),
            "capability_report": str(cfg.agent_dir / "capability_report.json"),
            "frontier_report": str(cfg.agent_dir / "frontier_report.json"),
            "proposed_curriculum": str(cfg.agent_dir / "candidate_curriculum_plan.json"),
            "verified_curriculum": str(cfg.agent_dir / "verified_curriculum_plan.json") if passed else None,
            "training_proposal": str(cfg.agent_dir / "training_proposal.json") if passed else None,
            "proposal_is_completed_training": False,
        }
    )
    memory.save(cfg.memory_path)

    if not passed:
        write_json(cfg.agent_dir / "blocked_candidate_curriculum_plan.json", candidate.to_dict())
        memory.record_error(
            stage=cfg.stage,
            source="MultiAgentCurriculumController",
            message="Curriculum verification did not produce a verified plan.",
            details=decision.to_dict(),
        )
        memory.save(cfg.memory_path)
        print(f"[ABORT] Curriculum verification failed; see {cfg.agent_dir / 'curriculum_verification_report.json'}", file=sys.stderr)
        sys.exit(4)

    harness = CurriculumExecutionHarness()
    try:
        harness_paths = harness.materialize_verified_plan(decision, cfg.agent_dir)
    except Exception as exc:
        memory.record_error(
            stage=cfg.stage,
            source="CurriculumExecutionHarness",
            message=f"{type(exc).__name__}: {exc}",
        )
        memory.save(cfg.memory_path)
        print(f"[STOP] Verified curriculum could not be materialized: {exc}", file=sys.stderr)
        raise
    legacy_proposal = read_json(harness_paths["legacy_proposal"])
    write_proposal_markdown(cfg.agent_dir / "training_proposal.md", legacy_proposal, audit)
    memory.save(cfg.memory_path)
    print("[DONE] FrontierDrive curriculum preparation and verification finished.")
    print(f"Verified: {harness_paths['verified']}")
    print(f"Proposal: {harness_paths['legacy_proposal']}")
    print(f"Memory:   {cfg.memory_path}")


if __name__ == "__main__":
    main()
