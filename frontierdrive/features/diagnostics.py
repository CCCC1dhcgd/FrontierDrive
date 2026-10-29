#!/usr/bin/env python
"""
Semantic sanity checks for aligned CBS features.

This script does not use EPDMS. It only checks whether handcrafted feature
measurements have plausible internal relationships and basic distributions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
from scipy.stats import spearmanr


CORRELATION_CHECKS = [
    ("scene_min_ttc", "min_agent_distance_to_ego", "positive"),
    ("scene_min_ttc", "forward_agent_count", "negative"),
    ("n_total_agents", "forward_agent_count", "positive"),
    ("n_total_agents", "future_max_forward_agent_count", "positive"),
    ("n_vehicles", "nearest_vehicle_distance", "negative"),
    ("current_speed", "curvature_mean", "negative"),
    ("future_min_agent_distance", "min_agent_distance_to_ego", "positive"),
    ("agent_speed_variance", "n_total_agents", "positive"),
    ("scene_drac_max", "scene_min_ttc", "negative"),
]


def load_aligned(path: Path, required: bool, split_name: str) -> Optional[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{split_name} aligned file does not exist: {path}")
        return None

    data = np.load(path)
    required_arrays = ["features_raw", "features_norm", "features_pca", "scenario_tokens", "log_names", "feature_names"]
    missing = [name for name in required_arrays if name not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")

    feature_names = data["feature_names"].tolist()
    features_raw = np.asarray(data["features_raw"], dtype=np.float32)
    features_norm = np.asarray(data["features_norm"], dtype=np.float32)
    features_pca = np.asarray(data["features_pca"], dtype=np.float32)
    scenario_tokens = data["scenario_tokens"]
    log_names = data["log_names"]

    if features_raw.ndim != 2 or features_raw.shape[1] != len(feature_names):
        raise ValueError(f"{path}: features_raw shape {features_raw.shape} does not match {len(feature_names)} feature names")
    if features_norm.shape != features_raw.shape:
        raise ValueError(f"{path}: features_norm shape {features_norm.shape} does not match features_raw {features_raw.shape}")
    if features_pca.ndim != 2 or features_pca.shape[1] != 12:
        raise ValueError(f"{path}: expected features_pca shape [N, 12], got {features_pca.shape}")
    if features_pca.shape[0] != features_raw.shape[0]:
        raise ValueError(f"{path}: features_pca row count {features_pca.shape[0]} does not match raw row count {features_raw.shape[0]}")
    if len(scenario_tokens) != features_raw.shape[0] or len(log_names) != features_raw.shape[0]:
        raise ValueError(f"{path}: token/log count does not match feature row count {features_raw.shape[0]}")
    if not np.isfinite(features_raw).all() or not np.isfinite(features_norm).all() or not np.isfinite(features_pca).all():
        raise ValueError(f"{path}: found non-finite values in aligned arrays")

    return {
        "features_raw": features_raw,
        "features_norm": features_norm,
        "features_pca": features_pca,
        "feature_names": feature_names,
        "scenario_tokens": scenario_tokens,
        "log_names": log_names,
    }


def feature_index(feature_names: list[str], name: str) -> Optional[int]:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def direction_matches(rho: float, expected: str) -> bool:
    if np.isnan(rho):
        return False
    if expected == "positive":
        return rho > 0.0
    if expected == "negative":
        return rho < 0.0
    raise ValueError(f"Unknown expected direction: {expected}")


def _safe_spearman(left_values: np.ndarray, right_values: np.ndarray) -> tuple[float, float]:
    if np.unique(left_values).size < 2 or np.unique(right_values).size < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p_value = spearmanr(left_values, right_values)
    return float(rho), float(p_value)


def add_split_summary(lines: list[str], split_name: str, split: dict) -> None:
    raw = split["features_raw"]
    norm = split["features_norm"]
    pca = split["features_pca"]
    lines.append(f"\nSplit summary: {split_name}")
    lines.append(f"  scenes:       {raw.shape[0]}")
    lines.append(f"  raw shape:    {raw.shape}")
    lines.append(f"  norm shape:   {norm.shape}")
    lines.append(f"  pca shape:    {pca.shape}")
    lines.append(f"  pca mean abs: {float(np.mean(np.abs(np.mean(pca, axis=0)))):.6f}")
    lines.append(f"  pca std mean: {float(np.mean(np.std(pca, axis=0))):.6f}")


def add_correlation_checks(lines: list[str], features_raw: np.ndarray, feature_names: list[str]) -> None:
    lines.append("\nStep A: Semantic correlation checks")
    lines.append(f"{'Feature A':<28} {'Feature B':<32} {'Expected':<10} {'rho':>10} {'p-value':>12} {'Result':>8}")
    lines.append("-" * 108)

    seen_pairs = set()
    for left, right, expected in CORRELATION_CHECKS:
        pair_key = (left, right)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        left_idx = feature_index(feature_names, left)
        right_idx = feature_index(feature_names, right)
        if left_idx is None or right_idx is None:
            lines.append(f"{left:<28} {right:<32} {expected:<10} {'NA':>10} {'NA':>12} {'MISSING':>8}")
            continue

        rho, p_value = _safe_spearman(features_raw[:, left_idx], features_raw[:, right_idx])
        ok = direction_matches(float(rho), expected)
        if np.isnan(rho):
            result = "CONST"
            rho_text = "NA"
            p_text = "NA"
        else:
            result = "OK" if ok else "CHECK"
            rho_text = f"{rho:.4f}"
            p_text = f"{p_value:.3e}"
        lines.append(f"{left:<28} {right:<32} {expected:<10} {rho_text:>10} {p_text:>12} {result:>8}")


def add_basic_statistics(lines: list[str], features_raw: np.ndarray, feature_names: list[str]) -> None:
    lines.append("\nStep B: Basic feature statistics")
    lines.append(f"{'Feature':<34} {'mean':>12} {'std':>12} {'p5':>12} {'p95':>12} {'frac_zero':>12} {'Flag':>10}")
    lines.append("-" * 112)

    for index, name in enumerate(feature_names):
        values = features_raw[:, index]
        mean = float(np.mean(values))
        std = float(np.std(values))
        p5 = float(np.percentile(values, 5))
        p95 = float(np.percentile(values, 95))
        fraction_zero = float(np.mean(values == 0.0))
        flags = []
        if std == 0.0:
            flags.append("CONST")
        if fraction_zero > 0.90:
            flags.append("HIGH_ZERO")
        flag = ",".join(flags)
        lines.append(f"{name:<34} {mean:>12.4f} {std:>12.4f} {p5:>12.4f} {p95:>12.4f} {fraction_zero:>12.4f} {flag:>10}")


def maybe_plot_umap(navtrain: dict, navtest: Optional[dict], navhard: Optional[dict], output_dir: Path) -> None:
    try:
        import umap
    except ImportError:
        print("\nStep C: skipped UMAP plot because umap-learn is not installed")
        return

    try:
        import matplotlib
    except ImportError:
        print("\nStep C: skipped UMAP plot because matplotlib is not installed")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reducer = umap.UMAP(n_components=2, random_state=42)
    train_xy = reducer.fit_transform(navtrain["features_pca"])

    plt.figure(figsize=(9, 7))
    plt.scatter(train_xy[:, 0], train_xy[:, 1], s=3, c="#8a8a8a", alpha=0.35, label="navtrain")

    if navtest is not None:
        test_xy = reducer.transform(navtest["features_pca"])
        plt.scatter(test_xy[:, 0], test_xy[:, 1], s=5, c="#1f77b4", alpha=0.55, label="navtest")

    if navhard is not None:
        hard_xy = reducer.transform(navhard["features_pca"])
        plt.scatter(hard_xy[:, 0], hard_xy[:, 1], s=5, c="#d62728", alpha=0.55, label="navhard")

    plt.legend(markerscale=3)
    plt.title("CBS aligned feature UMAP")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    output_path = output_dir / "umap_all_splits.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nStep C: saved UMAP plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic and statistical sanity checks for aligned CBS features.")
    parser.add_argument("--navtrain", type=Path, required=True, help="navtrain_aligned.npz")
    parser.add_argument("--navtest", type=Path, default=None, help="Optional navtest_aligned.npz")
    parser.add_argument("--navhard", type=Path, default=None, help="Optional navhard_aligned.npz")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for sanity-check outputs.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    navtrain = load_aligned(args.navtrain, required=True, split_name="navtrain")
    navtest = load_aligned(args.navtest, required=False, split_name="navtest") if args.navtest is not None else None
    navhard = load_aligned(args.navhard, required=False, split_name="navhard") if args.navhard is not None else None

    lines = ["CBS aligned feature sanity report"]
    add_split_summary(lines, "navtrain", navtrain)
    if navtest is not None:
        add_split_summary(lines, "navtest", navtest)
    if navhard is not None:
        add_split_summary(lines, "navhard", navhard)
    add_correlation_checks(lines, navtrain["features_raw"], navtrain["feature_names"])
    add_basic_statistics(lines, navtrain["features_raw"], navtrain["feature_names"])

    report_text = "\n".join(lines)
    print(report_text)
    report_path = args.output_dir / "sanity_report.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print(f"\nSaved sanity report to {report_path}")
    maybe_plot_umap(navtrain, navtest, navhard, args.output_dir)


if __name__ == "__main__":
    main()
