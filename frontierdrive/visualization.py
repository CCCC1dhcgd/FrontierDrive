#!/usr/bin/env python
"""Visualize adaptive-sphere retrieval in the first two PCA coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_npz(path: Path, required: list[str]) -> dict[str, np.ndarray]:
    data = np.load(path)
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"{path} is missing arrays: {missing}")
    return {name: data[name] for name in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize TBAT adaptive-sphere retrieval.")
    parser.add_argument("--navtrain-aligned", type=Path, required=True, help="navtrain_aligned.npz")
    parser.add_argument("--boundary", type=Path, required=True, help="navtest_boundary.npz")
    parser.add_argument("--selection", type=Path, required=True, help="train_selection.npz from frontier_retrieval.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for visualization PNG files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    navtrain = load_npz(args.navtrain_aligned, ["features_pca"])
    boundary = load_npz(args.boundary, ["boundary_features_pca"])
    selection = load_npz(
        args.selection,
        [
            "selected",
            "nearest_distance",
            "nearest_cluster",
            "centers",
            "radii",
            "boundary_cluster",
            "boundary_to_train_distance",
            "boundary_covered",
            "train_weight",
        ],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_xy = np.asarray(navtrain["features_pca"], dtype=np.float32)[:, :2]
    boundary_xy = np.asarray(boundary["boundary_features_pca"], dtype=np.float32)[:, :2]
    centers_xy = np.asarray(selection["centers"], dtype=np.float32)[:, :2]
    radii = np.asarray(selection["radii"], dtype=np.float32)
    selected = np.asarray(selection["selected"], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(train_xy[~selected, 0], train_xy[~selected, 1], s=3, c="#b8b8b8", alpha=0.25, label="navtrain replay")
    ax.scatter(train_xy[selected, 0], train_xy[selected, 1], s=6, c="#f28e2b", alpha=0.70, label="retrieved navtrain")
    ax.scatter(boundary_xy[:, 0], boundary_xy[:, 1], s=8, c="#d62728", alpha=0.55, label="navtest boundary")
    ax.scatter(centers_xy[:, 0], centers_xy[:, 1], s=80, c="#1f77b4", marker="D", edgecolor="white", label="sphere centers")

    for center, radius in zip(centers_xy, radii):
        circle = plt.Circle(center, radius, fill=False, color="#1f77b4", alpha=0.35, linewidth=1.2)
        ax.add_patch(circle)

    ax.set_title("TBAT adaptive sphere retrieval, PC1-PC2 projection")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.legend(markerscale=2)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    plot_path = args.output_dir / "retrieval_pc12_projection.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    nearest_distance = np.asarray(selection["nearest_distance"], dtype=np.float32)
    nearest_cluster = np.asarray(selection["nearest_cluster"], dtype=np.int64)
    train_ratio = nearest_distance / radii[nearest_cluster]

    boundary_cluster = np.asarray(selection["boundary_cluster"], dtype=np.int64)
    boundary_distance = np.asarray(selection["boundary_to_train_distance"], dtype=np.float32)
    boundary_ratio = boundary_distance / radii[boundary_cluster]
    boundary_covered = np.asarray(selection["boundary_covered"], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(train_ratio[~selected], bins=60, color="#b8b8b8", alpha=0.75, label="replay")
    axes[0].hist(train_ratio[selected], bins=60, color="#f28e2b", alpha=0.75, label="retrieved")
    axes[0].axvline(1.0, color="#222222", linewidth=1.2)
    axes[0].set_title("navtrain distance / sphere radius")
    axes[0].set_xlabel("ratio")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].hist(boundary_ratio[~boundary_covered], bins=50, color="#b8b8b8", alpha=0.75, label="uncovered")
    axes[1].hist(boundary_ratio[boundary_covered], bins=50, color="#d62728", alpha=0.75, label="covered")
    axes[1].axvline(1.0, color="#222222", linewidth=1.2)
    axes[1].set_title("boundary nearest-train distance / sphere radius")
    axes[1].set_xlabel("ratio")
    axes[1].set_ylabel("count")
    axes[1].legend()

    fig.tight_layout()
    ratio_path = args.output_dir / "retrieval_distance_ratio.png"
    fig.savefig(ratio_path, dpi=220)
    plt.close(fig)

    train_weight = np.asarray(selection["train_weight"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 8))
    order = np.argsort(train_weight)
    scatter = ax.scatter(
        train_xy[order, 0],
        train_xy[order, 1],
        s=4,
        c=train_weight[order],
        cmap="magma",
        alpha=0.75,
    )
    ax.scatter(boundary_xy[:, 0], boundary_xy[:, 1], s=8, c="#2ca02c", alpha=0.45, label="navtest boundary")
    ax.scatter(centers_xy[:, 0], centers_xy[:, 1], s=80, c="#1f77b4", marker="D", edgecolor="white", label="sphere centers")
    ax.set_title("navtrain training intensity, PC1-PC2 projection")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(markerscale=2)
    fig.colorbar(scatter, ax=ax, label="train_weight")
    fig.tight_layout()
    heatmap_path = args.output_dir / "training_intensity_heatmap.png"
    fig.savefig(heatmap_path, dpi=220)
    plt.close(fig)

    print(f"Saved {plot_path}")
    print(f"Saved {ratio_path}")
    print(f"Saved {heatmap_path}")


if __name__ == "__main__":
    main()
