#!/usr/bin/env python
"""Map CASB v2 navtest ZPD boundary scenes to weighted navtrain sampling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def load_navtrain(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = ["features_pca", "scenario_tokens", "log_names"]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"{path} is missing arrays: {missing}")
    return {
        "features_pca": np.asarray(data["features_pca"], dtype=np.float32),
        "scenario_tokens": data["scenario_tokens"].astype(str),
        "log_names": data["log_names"].astype(str),
    }


def load_boundary(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = [
        "boundary_features_pca",
        "boundary_tokens",
        "boundary_scores",
        "boundary_irt_difficulty",
        "boundary_pass_probability",
        "anneal_factor",
        "gate_pass",
    ]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"{path} is missing arrays: {missing}")
    features = np.asarray(data["boundary_features_pca"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"{path}: boundary_features_pca must have shape [N>0, D], got {features.shape}")
    return {
        "features_pca": features,
        "tokens": data["boundary_tokens"].astype(str),
        "scores": np.asarray(data["boundary_scores"], dtype=np.float32),
        "irt_difficulty": np.asarray(data["boundary_irt_difficulty"], dtype=np.float32),
        "pass_probability": np.asarray(data["boundary_pass_probability"], dtype=np.float32),
        "anneal_factor": float(np.asarray(data["anneal_factor"]).item()),
        "gate_pass": bool(np.asarray(data["gate_pass"]).item()),
    }


def pairwise_squared_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_sq = np.sum(left * left, axis=1, keepdims=True)
    right_sq = np.sum(right * right, axis=1, keepdims=True).T
    distances = left_sq + right_sq - 2.0 * left @ right.T
    return np.maximum(distances, 0.0)


def minmax(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def centers_from_labels(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centers = []
    for cluster_id in range(int(np.max(labels)) + 1):
        centers.append(np.mean(features[labels == cluster_id], axis=0))
    return np.asarray(centers, dtype=np.float32)


def renumber_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted(int(value) for value in np.unique(labels))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def pc12_support_radius(train_features: np.ndarray, scale: float) -> float:
    xy = train_features[:, :2]
    pc1_span = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    pc2_span = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    return 0.5 * max(pc1_span, pc2_span) * scale


def radii_from_train(
    centers: np.ndarray,
    train_features: np.ndarray,
    radius_knn: int,
    radius_scale: float,
    difficulty_factor: np.ndarray,
    anneal_factor: float,
    min_train_per_sphere: int,
    enforce_floor: bool,
    floor_mask: Optional[np.ndarray],
    max_floor_radius_ratio: float,
    max_sphere_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors = min(max(radius_knn, min_train_per_sphere), train_features.shape[0])
    knn = NearestNeighbors(n_neighbors=neighbors)
    knn.fit(train_features)
    center_distances, _ = knn.kneighbors(centers)
    base_radii = (radius_scale * np.median(center_distances[:, :radius_knn], axis=1)).astype(np.float32)
    radii = base_radii * difficulty_factor * anneal_factor

    if enforce_floor:
        floor_index = min(min_train_per_sphere, train_features.shape[0]) - 1
        floor_radii = (center_distances[:, floor_index] * 1.0001).astype(np.float32)
        if floor_mask is None:
            floor_mask = np.ones(radii.shape[0], dtype=bool)
        floor_allowed = floor_mask & (floor_radii <= radii * max_floor_radius_ratio)
        floor_enforced = floor_allowed & (floor_radii > radii)
        radii = np.where(floor_enforced, floor_radii, radii)
    else:
        floor_enforced = np.zeros(radii.shape[0], dtype=bool)
    radii = np.minimum(radii, max_sphere_radius).astype(np.float32)
    return base_radii, radii, floor_enforced


def cap_point_like_radii(
    radii: np.ndarray,
    point_like_clusters: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, float]:
    if not np.any(point_like_clusters):
        return radii.astype(np.float32), 0.0

    reference = radii[~point_like_clusters]
    if reference.size == 0:
        reference = radii
    cap = float(np.median(reference) * scale)
    if cap <= 0.0:
        return radii.astype(np.float32), cap
    capped = radii.copy()
    capped[point_like_clusters] = np.minimum(capped[point_like_clusters], cap)
    return capped.astype(np.float32), cap


def forced_point_like_support(
    train_features: np.ndarray,
    centers: np.ndarray,
    point_like_clusters: np.ndarray,
    train_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    support_mask = np.zeros((train_features.shape[0], centers.shape[0]), dtype=bool)
    support_weight = np.zeros((train_features.shape[0], centers.shape[0]), dtype=np.float64)
    cluster_ids = np.where(point_like_clusters)[0]
    if cluster_ids.size == 0 or train_count <= 0:
        return support_mask, support_weight

    neighbors = min(train_count, train_features.shape[0])
    knn = NearestNeighbors(n_neighbors=neighbors)
    knn.fit(train_features)
    distances, indices = knn.kneighbors(centers[cluster_ids])
    for row, cluster_id in enumerate(cluster_ids):
        cluster_indices = indices[row]
        support_mask[cluster_indices, cluster_id] = True
        farthest = float(max(distances[row, -1], 1e-6))
        support_weight[cluster_indices, cluster_id] = np.exp(-0.5 * (distances[row] / farthest) ** 2)
    return support_mask, support_weight


def sphere_assignment(
    train_features: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances = np.sqrt(pairwise_squared_distance(train_features, centers))
    distance_ratio = distances / radii[np.newaxis, :]
    in_sphere_mask = distances <= radii[np.newaxis, :]
    selected = np.any(in_sphere_mask, axis=1)
    nearest_cluster = np.argmin(distance_ratio, axis=1).astype(np.int64)
    nearest_distance = distances[np.arange(distances.shape[0]), nearest_cluster]
    return selected, in_sphere_mask, nearest_cluster, nearest_distance


def merged_boundary_clusters(
    boundary_features: np.ndarray,
    train_features: np.ndarray,
    initial_clusters: int,
    radius_knn: int,
    radius_scale: float,
    difficulty_factor: np.ndarray,
    anneal_factor: float,
    min_train_per_sphere: int,
    merge_distance_scale: float,
    isolated_boundary_count: int,
    max_floor_radius_ratio: float,
    max_sphere_radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=initial_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(boundary_features).astype(np.int64)
    centers = centers_from_labels(boundary_features, labels)
    initial_dist = np.sqrt(pairwise_squared_distance(centers, centers))
    positive_dist = initial_dist[initial_dist > 0.0]
    merge_distance = float(np.median(positive_dist) * merge_distance_scale) if positive_dist.size else 0.0

    while int(np.max(labels)) + 1 > 1:
        centers = centers_from_labels(boundary_features, labels)
        cluster_difficulty = np.asarray(
            [np.mean(difficulty_factor[labels == cluster_id]) for cluster_id in range(centers.shape[0])],
            dtype=np.float32,
        )
        base_radii, radii, _ = radii_from_train(
            centers,
            train_features,
            radius_knn,
            radius_scale,
            cluster_difficulty,
            anneal_factor,
            min_train_per_sphere,
            enforce_floor=False,
            floor_mask=None,
            max_floor_radius_ratio=max_floor_radius_ratio,
            max_sphere_radius=max_sphere_radius,
        )
        selected, in_sphere_mask, _, _ = sphere_assignment(train_features, centers, radii)
        sphere_counts = np.sum(in_sphere_mask, axis=0)
        small_clusters = np.where(sphere_counts < min_train_per_sphere)[0]
        if small_clusters.size == 0:
            boundary_counts = np.asarray([np.sum(labels == cluster_id) for cluster_id in range(centers.shape[0])])
            point_like_clusters = boundary_counts <= isolated_boundary_count
            floor_mask = ~point_like_clusters
            base_radii, radii, floor_enforced = radii_from_train(
                centers,
                train_features,
                radius_knn,
                radius_scale,
                cluster_difficulty,
                anneal_factor,
                min_train_per_sphere,
                enforce_floor=True,
                floor_mask=floor_mask,
                max_floor_radius_ratio=max_floor_radius_ratio,
                max_sphere_radius=max_sphere_radius,
            )
            return labels, centers, base_radii, radii, floor_enforced, point_like_clusters

        source = int(small_clusters[np.argmin(sphere_counts[small_clusters])])
        center_dist = np.sqrt(pairwise_squared_distance(centers, centers))
        center_dist[source, source] = np.inf
        target = int(np.argmin(center_dist[source]))
        if merge_distance > 0.0 and float(center_dist[source, target]) > merge_distance:
            boundary_counts = np.asarray([np.sum(labels == cluster_id) for cluster_id in range(centers.shape[0])])
            point_like_clusters = boundary_counts <= isolated_boundary_count
            floor_mask = ~point_like_clusters
            base_radii, radii, floor_enforced = radii_from_train(
                centers,
                train_features,
                radius_knn,
                radius_scale,
                cluster_difficulty,
                anneal_factor,
                min_train_per_sphere,
                enforce_floor=True,
                floor_mask=floor_mask,
                max_floor_radius_ratio=max_floor_radius_ratio,
                max_sphere_radius=max_sphere_radius,
            )
            return labels, centers, base_radii, radii, floor_enforced, point_like_clusters

        labels[labels == source] = target
        labels = renumber_labels(labels)

    centers = centers_from_labels(boundary_features, labels)
    cluster_difficulty = np.asarray([np.mean(difficulty_factor[labels == 0])], dtype=np.float32)
    boundary_counts = np.asarray([np.sum(labels == 0)])
    point_like_clusters = boundary_counts <= isolated_boundary_count
    floor_mask = ~point_like_clusters
    base_radii, radii, floor_enforced = radii_from_train(
        centers,
        train_features,
        radius_knn,
        radius_scale,
        cluster_difficulty,
        anneal_factor,
        min_train_per_sphere,
        enforce_floor=True,
        floor_mask=floor_mask,
        max_floor_radius_ratio=max_floor_radius_ratio,
        max_sphere_radius=max_sphere_radius,
    )
    return labels, centers, base_radii, radii, floor_enforced, point_like_clusters


def write_manifest(
    path: Path,
    tokens: np.ndarray,
    log_names: np.ndarray,
    selected: np.ndarray,
    nearest_cluster: np.ndarray,
    nearest_distance: np.ndarray,
    cluster_radius: np.ndarray,
    proximity_weight: np.ndarray,
    train_weight: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "token",
                "log_name",
                "selected",
                "nearest_cluster",
                "nearest_distance",
                "cluster_radius",
                "proximity_weight",
                "train_weight",
            ]
        )
        for index, token in enumerate(tokens):
            cluster = int(nearest_cluster[index])
            writer.writerow(
                [
                    token,
                    log_names[index],
                    int(selected[index]),
                    cluster,
                    f"{nearest_distance[index]:.8f}",
                    f"{cluster_radius[cluster]:.8f}",
                    f"{proximity_weight[index]:.10f}",
                    f"{train_weight[index]:.12f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve navtrain scenes around CASB v2 boundary spheres.")
    parser.add_argument("--navtrain-aligned", type=Path, required=True, help="navtrain_aligned.npz")
    parser.add_argument("--boundary", type=Path, required=True, help="navtest_boundary.npz")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--clusters", type=int, default=None, help="Override KMeans cluster count. Default: auto.")
    parser.add_argument("--radius-scale", type=float, default=1.5, help="Multiplier on the median KNN base radius")
    parser.add_argument("--radius-knn", type=int, default=5, help="Navtrain KNN count for adaptive radius")
    parser.add_argument("--difficulty-gain", type=float, default=0.35, help="Extra radius multiplier for harder clusters")
    parser.add_argument("--boundary-mass", type=float, default=0.50, help="Sampler mass for selected scenes")
    parser.add_argument("--min-train-per-sphere", type=int, default=300, help="Merge spheres absorbing fewer train scenes")
    parser.add_argument("--merge-distance-scale", type=float, default=1.0, help="Max merge distance as median center-distance multiplier")
    parser.add_argument("--isolated-boundary-count", type=int, default=2, help="Clusters with at most this many boundary scenes keep point-like radii")
    parser.add_argument("--max-floor-radius-ratio", type=float, default=3.0, help="Do not apply the train-count radius floor if it would exceed this multiple of the dynamic radius")
    parser.add_argument("--max-sphere-radius-scale", type=float, default=1.0, help="Cap sphere radius by this multiple of navtrain PC1/PC2 half-span")
    parser.add_argument("--point-like-radius-scale", type=float, default=1.0, help="Cap isolated-cluster radius by this multiple of the median non-isolated radius")
    parser.add_argument("--point-like-train-count", type=int, default=10, help="Nearest navtrain scenes kept as proxy support for each isolated cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not 0.0 < args.boundary_mass < 1.0:
        raise ValueError("--boundary-mass must be in (0, 1)")
    if args.clusters is not None and args.clusters < 1:
        raise ValueError("--clusters must be >= 1 when provided")
    if args.radius_knn < 1:
        raise ValueError("--radius-knn must be >= 1")
    if args.radius_scale <= 0.0:
        raise ValueError("--radius-scale must be > 0")
    if args.min_train_per_sphere < 1:
        raise ValueError("--min-train-per-sphere must be >= 1")
    if args.isolated_boundary_count < 0:
        raise ValueError("--isolated-boundary-count must be >= 0")
    if args.max_floor_radius_ratio < 1.0:
        raise ValueError("--max-floor-radius-ratio must be >= 1")
    if args.max_sphere_radius_scale <= 0.0:
        raise ValueError("--max-sphere-radius-scale must be > 0")
    if args.point_like_radius_scale <= 0.0:
        raise ValueError("--point-like-radius-scale must be > 0")
    if args.point_like_train_count < 0:
        raise ValueError("--point-like-train-count must be >= 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    navtrain = load_navtrain(args.navtrain_aligned)
    boundary = load_boundary(args.boundary)

    train_features = navtrain["features_pca"]
    boundary_features = boundary["features_pca"]
    max_sphere_radius = pc12_support_radius(train_features, args.max_sphere_radius_scale)
    if args.clusters is None:
        n_clusters_auto = max(4, round(np.sqrt(boundary_features.shape[0] / 36.0)))
        n_clusters = min(n_clusters_auto, boundary_features.shape[0])
    else:
        n_clusters = min(args.clusters, boundary_features.shape[0])
    if n_clusters < 1:
        raise ValueError("Need at least one boundary feature")

    difficulty_norm = minmax(boundary["irt_difficulty"])
    per_boundary_difficulty_factor = 1.0 + args.difficulty_gain * difficulty_norm
    boundary_cluster, centers, base_radii, radii, radius_floor_enforced, point_like_clusters = merged_boundary_clusters(
        boundary_features=boundary_features,
        train_features=train_features,
        initial_clusters=n_clusters,
        radius_knn=args.radius_knn,
        radius_scale=args.radius_scale,
        difficulty_factor=per_boundary_difficulty_factor,
        anneal_factor=boundary["anneal_factor"],
        min_train_per_sphere=args.min_train_per_sphere,
        merge_distance_scale=args.merge_distance_scale,
        isolated_boundary_count=args.isolated_boundary_count,
        max_floor_radius_ratio=args.max_floor_radius_ratio,
        max_sphere_radius=max_sphere_radius,
        seed=args.seed,
    )
    uncapped_radii = radii.copy()
    radii, point_like_radius_cap = cap_point_like_radii(
        radii,
        point_like_clusters=point_like_clusters,
        scale=args.point_like_radius_scale,
    )
    n_clusters = centers.shape[0]

    sphere_selected, in_sphere_mask, nearest_cluster, nearest_distance = sphere_assignment(train_features, centers, radii)
    point_like_support_mask, point_like_support_weight = forced_point_like_support(
        train_features,
        centers,
        point_like_clusters=point_like_clusters,
        train_count=args.point_like_train_count,
    )
    selected = sphere_selected | np.any(point_like_support_mask, axis=1)
    if not np.any(selected):
        raise RuntimeError("No navtrain scenes were retrieved by CASB v2 spheres")

    distances = np.sqrt(pairwise_squared_distance(train_features, centers))
    distance_ratio = distances / radii[np.newaxis, :]
    per_sphere_gaussian = np.exp(-0.5 * distance_ratio**2).astype(np.float64)
    per_sphere_gaussian[~in_sphere_mask] = 0.0
    per_sphere_gaussian = np.maximum(per_sphere_gaussian, point_like_support_weight)
    point_like_rows, point_like_cols = np.where(point_like_support_mask)
    if point_like_rows.size:
        nearest_cluster[point_like_rows] = point_like_cols.astype(np.int64)
        nearest_distance[point_like_rows] = distances[point_like_rows, point_like_cols]

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(train_features)
    boundary_to_train_distance, _ = knn.kneighbors(boundary_features)
    boundary_radius = radii[boundary_cluster]
    boundary_covered = boundary_to_train_distance[:, 0] <= boundary_radius

    proximity_weight = np.max(per_sphere_gaussian, axis=1).astype(np.float64)
    selected_weight = np.where(selected, proximity_weight, 0.0)
    replay_weight = np.where(selected, 0.0, 1.0)
    if np.sum(replay_weight) == 0.0:
        raise RuntimeError("Sphere selection covered all navtrain scenes; replay set is empty")

    train_weight = np.zeros(train_features.shape[0], dtype=np.float64)
    train_weight += args.boundary_mass * selected_weight / np.sum(selected_weight)
    train_weight += (1.0 - args.boundary_mass) * replay_weight / np.sum(replay_weight)

    manifest_path = args.output_dir / "train_manifest.csv"
    write_manifest(
        manifest_path,
        navtrain["scenario_tokens"],
        navtrain["log_names"],
        selected,
        nearest_cluster,
        nearest_distance,
        radii,
        proximity_weight,
        train_weight,
    )

    selected_tokens = navtrain["scenario_tokens"][selected]
    (args.output_dir / "selected_train_tokens.txt").write_text("\n".join(selected_tokens.tolist()) + "\n", encoding="utf-8")
    cluster_centers_path = args.output_dir / "cluster_centers.npy"
    np.save(cluster_centers_path, centers)

    np.savez_compressed(
        args.output_dir / "train_selection.npz",
        tokens=navtrain["scenario_tokens"],
        log_names=navtrain["log_names"],
        selected=selected,
        train_weight=train_weight.astype(np.float32),
        proximity_weight=proximity_weight.astype(np.float32),
        nearest_cluster=nearest_cluster,
        nearest_distance=nearest_distance.astype(np.float32),
        distance_ratio=distance_ratio.astype(np.float32),
        in_sphere_mask=in_sphere_mask,
        point_like_support_mask=point_like_support_mask,
        point_like_clusters=point_like_clusters,
        centers=centers,
        radii=radii.astype(np.float32),
        uncapped_radii=uncapped_radii.astype(np.float32),
        base_radii=base_radii.astype(np.float32),
        radius_floor_enforced=radius_floor_enforced,
        max_sphere_radius=np.asarray(max_sphere_radius, dtype=np.float32),
        anneal_factor=np.asarray(boundary["anneal_factor"], dtype=np.float32),
        gate_pass=np.asarray(boundary["gate_pass"], dtype=bool),
        boundary_tokens=boundary["tokens"],
        boundary_cluster=boundary_cluster,
        boundary_scores=boundary["scores"],
        boundary_pass_probability=boundary["pass_probability"],
        boundary_irt_difficulty=boundary["irt_difficulty"],
        boundary_covered=boundary_covered,
        boundary_to_train_distance=boundary_to_train_distance[:, 0].astype(np.float32),
    )

    cluster_counts = []
    for cluster_id in range(n_clusters):
        in_cluster = nearest_cluster == cluster_id
        covered_by_cluster = in_sphere_mask[:, cluster_id]
        boundary_in_cluster = boundary_cluster == cluster_id
        cluster_counts.append(
            {
                "cluster": cluster_id,
                "boundary_count": int(np.sum(boundary_in_cluster)),
                "mean_boundary_score": float(np.mean(boundary["scores"][boundary_in_cluster])),
                "mean_pass_probability": float(np.mean(boundary["pass_probability"][boundary_in_cluster])),
                "mean_irt_difficulty": float(np.mean(boundary["irt_difficulty"][boundary_in_cluster])),
                "base_radius": float(base_radii[cluster_id]),
                "radius": float(radii[cluster_id]),
                "uncapped_radius": float(uncapped_radii[cluster_id]),
                "radius_capped": bool(np.isclose(radii[cluster_id], max_sphere_radius, rtol=1e-6, atol=1e-6)),
                "point_like_radius_capped": bool(point_like_clusters[cluster_id] and radii[cluster_id] < uncapped_radii[cluster_id]),
                "radius_floor_enforced": bool(radius_floor_enforced[cluster_id]),
                "point_like_cluster": bool(point_like_clusters[cluster_id]),
                "selected_train_count": int(np.sum(selected & in_cluster)),
                "raw_sphere_selected_train_count": int(np.sum(covered_by_cluster)),
                "point_like_support_train_count": int(np.sum(point_like_support_mask[:, cluster_id])),
            }
        )

    covering_counts = np.sum(in_sphere_mask, axis=1)
    selected_covering_counts = covering_counts[selected]
    summary = {
        "navtrain_aligned": str(args.navtrain_aligned),
        "boundary": str(args.boundary),
        "n_boundary": int(boundary_features.shape[0]),
        "n_train": int(train_features.shape[0]),
        "n_clusters": int(n_clusters),
        "clusters_auto": args.clusters is None,
        "radius_scale": args.radius_scale,
        "radius_knn": min(args.radius_knn, train_features.shape[0]),
        "difficulty_gain": args.difficulty_gain,
        "boundary_mass": args.boundary_mass,
        "min_train_per_sphere": args.min_train_per_sphere,
        "merge_distance_scale": args.merge_distance_scale,
        "isolated_boundary_count": args.isolated_boundary_count,
        "max_floor_radius_ratio": args.max_floor_radius_ratio,
        "max_sphere_radius_scale": args.max_sphere_radius_scale,
        "max_sphere_radius": float(max_sphere_radius),
        "point_like_radius_scale": args.point_like_radius_scale,
        "point_like_radius_cap": float(point_like_radius_cap),
        "point_like_train_count": args.point_like_train_count,
        "anneal_factor": boundary["anneal_factor"],
        "gate_pass": boundary["gate_pass"],
        "base_radii": base_radii.tolist(),
        "uncapped_radii": uncapped_radii.tolist(),
        "radii": radii.tolist(),
        "point_like_clusters": point_like_clusters.astype(bool).tolist(),
        "radius_floor_enforced": radius_floor_enforced.astype(bool).tolist(),
        "n_selected_train": int(np.sum(selected)),
        "selected_train_fraction": float(np.mean(selected)),
        "boundary_coverage_fraction": float(np.mean(boundary_covered)),
        "multi_sphere_selected_fraction": float(np.mean(selected_covering_counts > 1)),
        "mean_covering_spheres_per_selected_train": float(np.mean(selected_covering_counts)),
        "max_covering_spheres_per_selected_train": int(np.max(selected_covering_counts)),
        "train_weight_sum": float(np.sum(train_weight)),
        "manifest": str(manifest_path),
        "cluster_centers_path": str(cluster_centers_path),
        "clusters": cluster_counts,
    }
    (args.output_dir / "sphere_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
