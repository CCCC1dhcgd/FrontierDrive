#!/usr/bin/env python
"""Identify CASB v2 ZPD boundary scenes from navtest evaluation history."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


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


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def logit(probability: float) -> float:
    clipped = min(max(probability, 1e-4), 1.0 - 1e-4)
    return float(np.log(clipped / (1.0 - clipped)))


def standardize(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def minmax(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def load_aligned(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = ["features_raw", "features_pca", "scenario_tokens", "feature_names"]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"{path} is missing arrays: {missing}")
    return {
        "features_raw": np.asarray(data["features_raw"], dtype=np.float32),
        "features_pca": np.asarray(data["features_pca"], dtype=np.float32),
        "scenario_tokens": data["scenario_tokens"].astype(str),
        "feature_names": data["feature_names"].astype(str),
    }


def read_eval_csv(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        required = ["token", "valid", "score"] + SUBMETRIC_COLUMNS
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing}")
        for row in reader:
            token = row["token"]
            if token == "average_all_frames" or row["valid"] != "True":
                continue
            values = {"score": float(row["score"])}
            for name in SUBMETRIC_COLUMNS:
                values[name] = float(row[name]) if row[name] else 0.0
            rows[token] = values
    return rows


def feature_column(feature_names: np.ndarray, name: str) -> int:
    matches = np.where(feature_names == name)[0]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one feature named {name}, found {len(matches)}")
    return int(matches[0])


def stage_artifact_root(root: Path, stage: int) -> Path:
    structured_root = root / "evaluate" / f"stage{stage}"
    if structured_root.exists() or (root / "evaluate").exists():
        return structured_root
    return root / f"stage{stage}"


def eval_history_paths(root: Path, stage: int, eval_name_prefix: str) -> List[Path]:
    paths: List[Path] = []
    for eval_stage in range(stage):
        path = (
            stage_artifact_root(root, eval_stage)
            / f"{eval_name_prefix}_stage{eval_stage}_navtest"
            / "metrics_results.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing navtest history CSV for stage {eval_stage}: {path}")
        paths.append(path)
    return paths


def feature_difficulty_prior(
    raw: np.ndarray,
    feature_names: np.ndarray,
    score_history: np.ndarray,
) -> np.ndarray:
    route_length = raw[:, feature_column(feature_names, "route_length")]
    min_agent_distance = raw[:, feature_column(feature_names, "min_agent_distance_to_ego")]
    scene_drac_max = raw[:, feature_column(feature_names, "scene_drac_max")]
    traffic_agents = raw[:, feature_column(feature_names, "n_total_agents")]

    score_term = standardize(1.0 - np.mean(score_history, axis=1))
    route_term = standardize(minmax(route_length))
    proximity_term = standardize(1.0 - minmax(min_agent_distance))
    drac_term = standardize(minmax(scene_drac_max))
    density_term = standardize(minmax(traffic_agents))
    prior = score_term + 0.20 * route_term + 0.25 * proximity_term + 0.20 * drac_term + 0.15 * density_term
    return standardize(prior)


def fit_rasch(
    outcomes: np.ndarray,
    prior_difficulty: np.ndarray,
    l2: float,
    steps: int,
    lr: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_scenes, n_rounds = outcomes.shape
    theta = np.asarray([logit(float(np.mean(outcomes[:, j]))) for j in range(n_rounds)], dtype=np.float64)
    difficulty = prior_difficulty.astype(np.float64)
    difficulty -= float(np.mean(difficulty))
    theta -= float(np.mean(difficulty))

    for _ in range(steps):
        prob = sigmoid(theta[np.newaxis, :] - difficulty[:, np.newaxis])
        error = prob - outcomes
        grad_theta = np.sum(error, axis=0) / n_scenes + l2 * theta
        grad_difficulty = -np.sum(error, axis=1) / n_rounds + l2 * difficulty
        theta -= lr * grad_theta
        difficulty -= lr * grad_difficulty
        shift = float(np.mean(difficulty))
        difficulty -= shift
        theta -= shift
    return theta.astype(np.float32), difficulty.astype(np.float32)


def choose_zpd_band(
    pass_probability: np.ndarray,
    eligible: np.ndarray,
    p_lo: float,
    p_hi: float,
    n_min: int,
    expand_step: float,
) -> tuple[np.ndarray, float, float]:
    lo = p_lo
    hi = p_hi
    zpd = eligible & (pass_probability >= lo) & (pass_probability <= hi)
    while int(np.sum(zpd)) < n_min and (lo > 0.0 or hi < 1.0):
        lo = max(0.0, lo - expand_step)
        hi = min(1.0, hi + expand_step)
        zpd = eligible & (pass_probability >= lo) & (pass_probability <= hi)
    return zpd, lo, hi


def write_boundary_csv(
    path: Path,
    tokens: np.ndarray,
    scores: np.ndarray,
    submetric_std: np.ndarray,
    pass_probability: np.ndarray,
    irt_difficulty: np.ndarray,
    eligible: np.ndarray,
    zpd: np.ndarray,
    selected: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "token",
                "score",
                "submetric_std",
                "pass_probability",
                "irt_difficulty",
                "eligible",
                "zpd",
                "selected",
            ]
        )
        for i, token in enumerate(tokens):
            writer.writerow(
                [
                    token,
                    f"{scores[i]:.8f}",
                    f"{submetric_std[i]:.8f}",
                    f"{pass_probability[i]:.8f}",
                    f"{irt_difficulty[i]:.8f}",
                    int(eligible[i]),
                    int(zpd[i]),
                    int(selected[i]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify CASB v2 IRT/ZPD boundary scenes.")
    parser.add_argument("--navtest-aligned", type=Path, required=True, help="navtest_aligned.npz")
    parser.add_argument("--eval-history-root", type=Path, required=True, help="CBAT run root containing stageN evaluation outputs")
    parser.add_argument("--eval-name-prefix", type=str, default="tbat_ltf_eval", help="Evaluation directory prefix before _stageN_navtest")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--stage", type=int, required=True, help="1-indexed CASB stage")
    parser.add_argument("--tau", type=float, default=0.95, help="Pass threshold for binary IRT outcomes")
    parser.add_argument("--p-lo", type=float, default=0.20, help="Lower ZPD predicted pass probability")
    parser.add_argument("--p-hi", type=float, default=0.60, help="Upper ZPD predicted pass probability")
    parser.add_argument("--n-min", type=int, default=300, help="Minimum ZPD boundary scenes after probability-band expansion")
    parser.add_argument("--band-expand-step", type=float, default=0.05, help="Probability-band expansion step for n_min")
    parser.add_argument("--irt-l2", type=float, default=0.02, help="Rasch L2 regularization")
    parser.add_argument("--irt-steps", type=int, default=600, help="Rasch gradient steps when at least three rounds exist")
    parser.add_argument("--irt-lr", type=float, default=0.15, help="Rasch gradient learning rate")
    parser.add_argument("--min-agent-distance", type=float, default=1.0, help="Physical eligibility threshold")
    parser.add_argument("--max-drac", type=float, default=30.0, help="Physical eligibility threshold")
    parser.add_argument("--anneal-rate", type=float, default=0.15, help="Global anneal shrink per measured theta gain")
    parser.add_argument("--min-anneal", type=float, default=0.65, help="Smallest allowed global anneal factor")
    args = parser.parse_args()

    if args.stage < 1:
        raise ValueError(f"--stage must be >= 1, got {args.stage}")
    if not 0.0 < args.tau < 1.0:
        raise ValueError("--tau must be in (0, 1)")
    if not 0.0 <= args.p_lo < args.p_hi <= 1.0:
        raise ValueError("--p-lo and --p-hi must satisfy 0 <= p_lo < p_hi <= 1")
    if args.n_min < 1:
        raise ValueError("--n-min must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    aligned = load_aligned(args.navtest_aligned)
    history_paths = eval_history_paths(args.eval_history_root, args.stage, args.eval_name_prefix)
    history_rows = [read_eval_csv(path) for path in history_paths]

    tokens_all = aligned["scenario_tokens"]
    eval_indices: List[int] = []
    eval_tokens: List[str] = []
    score_history: List[List[float]] = []
    submetric_values: List[List[float]] = []

    for index, token_raw in enumerate(tokens_all):
        token = str(token_raw)
        if token not in history_rows[-1]:
            continue
        row_scores = []
        for rows, path in zip(history_rows, history_paths):
            if token not in rows:
                raise RuntimeError(f"Token {token} is missing from required history CSV: {path}")
            row_scores.append(rows[token]["score"])
        eval_indices.append(index)
        eval_tokens.append(token)
        score_history.append(row_scores)
        submetric_values.append([history_rows[-1][token][name] for name in SUBMETRIC_COLUMNS])

    if not eval_indices:
        raise RuntimeError("No overlap between navtest aligned tokens and evaluation history")

    eval_indices_np = np.asarray(eval_indices, dtype=np.int64)
    tokens = np.asarray(eval_tokens, dtype=str)
    scores_by_round = np.asarray(score_history, dtype=np.float32)
    current_scores = scores_by_round[:, -1]
    outcomes = (scores_by_round >= args.tau).astype(np.float32)
    submetrics_np = np.asarray(submetric_values, dtype=np.float32)
    submetric_std = np.std(submetrics_np, axis=1).astype(np.float32)

    feature_names = aligned["feature_names"]
    raw = aligned["features_raw"][eval_indices_np]
    pca = aligned["features_pca"][eval_indices_np]

    route_length = raw[:, feature_column(feature_names, "route_length")]
    min_agent_distance = raw[:, feature_column(feature_names, "min_agent_distance_to_ego")]
    scene_drac_max = raw[:, feature_column(feature_names, "scene_drac_max")]
    eligible = (
        (route_length > 0.0)
        & (min_agent_distance >= args.min_agent_distance)
        & (scene_drac_max <= args.max_drac)
    )

    prior_difficulty = feature_difficulty_prior(raw, feature_names, scores_by_round)
    if outcomes.shape[1] >= 3:
        theta_history, irt_difficulty = fit_rasch(
            outcomes,
            prior_difficulty,
            l2=args.irt_l2,
            steps=args.irt_steps,
            lr=args.irt_lr,
        )
        estimator = "rasch_map"
    else:
        theta_history = np.asarray([logit(float(np.mean(outcomes[:, j]))) for j in range(outcomes.shape[1])], dtype=np.float32)
        irt_difficulty = prior_difficulty.astype(np.float32)
        estimator = "feature_prior_until_three_rounds"

    current_theta = float(theta_history[-1])
    pass_probability = sigmoid(current_theta - irt_difficulty).astype(np.float32)

    p20_history = np.quantile(scores_by_round, 0.20, axis=0).astype(np.float32)
    gate_evaluated = len(theta_history) >= 2
    if gate_evaluated:
        theta_non_decreasing = current_theta >= float(theta_history[-2]) - 1e-6
        p20_non_decreasing = float(p20_history[-1]) >= float(p20_history[-2]) - 1e-6
    else:
        theta_non_decreasing = False
        p20_non_decreasing = False
    gate_pass = bool(gate_evaluated and theta_non_decreasing and p20_non_decreasing)

    selection_p_lo = args.p_lo
    selection_p_hi = args.p_hi
    if gate_evaluated and not gate_pass:
        selection_p_lo = max(0.0, selection_p_lo - args.band_expand_step)
        selection_p_hi = min(1.0, selection_p_hi + args.band_expand_step)

    zpd, final_p_lo, final_p_hi = choose_zpd_band(
        pass_probability,
        eligible,
        selection_p_lo,
        selection_p_hi,
        args.n_min,
        args.band_expand_step,
    )
    selected = zpd
    selected_indices = np.where(selected)[0].astype(np.int64)
    if selected_indices.size == 0:
        raise RuntimeError("CASB v2 ZPD boundary selected zero scenes")

    theta_gain = max(0.0, current_theta - float(theta_history[0]))
    anneal_factor = 1.0
    if gate_pass:
        anneal_factor = max(args.min_anneal, 1.0 - args.anneal_rate * theta_gain)

    np.savez_compressed(
        args.output_dir / "navtest_boundary.npz",
        tokens=tokens,
        eval_indices=eval_indices_np,
        boundary_local_indices=selected_indices,
        boundary_tokens=tokens[selected_indices],
        boundary_features_pca=pca[selected_indices],
        boundary_scores=current_scores[selected_indices],
        boundary_irt_difficulty=irt_difficulty[selected_indices],
        boundary_pass_probability=pass_probability[selected_indices],
        scores=current_scores,
        score_history=scores_by_round,
        outcomes=outcomes,
        theta_history=theta_history,
        irt_difficulty=irt_difficulty,
        pass_probability=pass_probability,
        submetric_std=submetric_std,
        submetrics=submetrics_np,
        eval_features_pca=pca,
        eligible=eligible,
        zpd=zpd,
        selected=selected,
        tau=np.asarray(args.tau, dtype=np.float32),
        p_lo=np.asarray(final_p_lo, dtype=np.float32),
        p_hi=np.asarray(final_p_hi, dtype=np.float32),
        gate_pass=np.asarray(gate_pass, dtype=bool),
        anneal_factor=np.asarray(anneal_factor, dtype=np.float32),
        submetric_columns=np.asarray(SUBMETRIC_COLUMNS, dtype=str),
    )

    write_boundary_csv(
        args.output_dir / "navtest_boundary_all_eval_scenes.csv",
        tokens,
        current_scores,
        submetric_std,
        pass_probability,
        irt_difficulty,
        eligible,
        zpd,
        selected,
    )
    write_boundary_csv(
        args.output_dir / "navtest_boundary_selected.csv",
        tokens[selected],
        current_scores[selected],
        submetric_std[selected],
        pass_probability[selected],
        irt_difficulty[selected],
        eligible[selected],
        zpd[selected],
        selected[selected],
    )

    summary = {
        "eval_history": [str(path) for path in history_paths],
        "navtest_aligned": str(args.navtest_aligned),
        "stage": args.stage,
        "estimator": estimator,
        "tau": args.tau,
        "theta_history": theta_history.tolist(),
        "theta_current": current_theta,
        "p20_history": p20_history.tolist(),
        "gate_evaluated": gate_evaluated,
        "gate_pass": gate_pass,
        "anneal_factor": anneal_factor,
        "zpd_p_lo_requested": args.p_lo,
        "zpd_p_hi_requested": args.p_hi,
        "zpd_p_lo_gate_adjusted": selection_p_lo,
        "zpd_p_hi_gate_adjusted": selection_p_hi,
        "zpd_p_lo": final_p_lo,
        "zpd_p_hi": final_p_hi,
        "n_min": args.n_min,
        "n_eval_scenes": int(len(tokens)),
        "n_eligible": int(np.sum(eligible)),
        "n_boundary": int(np.sum(selected)),
        "boundary_fraction": float(np.mean(selected)),
        "current_score_mean": float(np.mean(current_scores)),
        "current_score_p20": float(p20_history[-1]),
        "current_mastered_fraction": float(np.mean(current_scores >= args.tau)),
        "boundary_score_median": float(np.median(current_scores[selected])),
        "boundary_pass_probability_median": float(np.median(pass_probability[selected])),
        "boundary_irt_difficulty_median": float(np.median(irt_difficulty[selected])),
        "layer1_definition": {
            "route_length": "> 0",
            "min_agent_distance_to_ego": f">= {args.min_agent_distance}",
            "scene_drac_max": f"<= {args.max_drac}",
        },
        "submetric_columns": SUBMETRIC_COLUMNS,
    }
    (args.output_dir / "boundary_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
