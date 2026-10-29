#!/usr/bin/env python
"""
CBS scene feature extraction from NAVSIM MetricCache.

Reads NAVSIM MetricCache pickle files, extracts one 21-dimensional handcrafted
feature vector per scene, standardizes the matrix, runs PCA to 12 dimensions,
and saves everything to a compressed .npz file.

The 21 dimensions are grouped as:
  A. static map / route structure: 4 dims
  B. dynamic traffic participants: 11 dims
  C. ego / history behavior: 4 dims
  D. current interaction safety: 2 dims

The feature vector intentionally excludes future human trajectory labels. CBS
retrieval should compare scene conditions, not the future answer trajectory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import lzma
import math
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from navsim.common.dataloader import MetricCacheLoader


CENTERLINE_LOOKAHEAD_METERS = 50.0
FORWARD_CONE_DEGREES = 30.0
FORWARD_CONE_DISTANCE_METERS = 40.0
NO_AGENT_DISTANCE_FALLBACK = 100.0
TTC_HORIZON_SECONDS = 8.0
EGO_COLLISION_RADIUS_METERS = 2.9
DEFAULT_AGENT_COLLISION_RADIUS_METERS = 1.5
DYNAMIC_AGENT_TYPE_KEYWORDS = ("vehicle", "pedestrian", "bicycle")
N_FEATURES = 21
RAW_PCA_COMPONENTS = 12
EPS = 1e-6

FEATURE_NAMES = [
    # Category A: static map / route structure
    "curvature_mean",
    "curvature_max",
    "route_complexity",
    "route_length",
    # Category B: dynamic traffic participants
    "n_vehicles",
    "n_pedestrians",
    "n_total_agents",
    "min_agent_distance_to_ego",
    "forward_agent_count",
    "future_min_agent_distance",
    "future_max_forward_agent_count",
    "agent_speed_variance",
    "quadrant_front_count",
    "quadrant_left_count",
    "nearest_vehicle_distance",
    # Category C: ego / history behavior
    "current_speed",
    "ego_acceleration_magnitude",
    "past_speed_variance",
    "past_heading_rate_mean",
    # Category D: current interaction safety
    "scene_min_ttc",
    "scene_drac_max",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _xy_heading_from_state(state: Any) -> Tuple[float, float, float]:
    if state is None:
        return 0.0, 0.0, 0.0
    rear_axle = getattr(state, "rear_axle", None)
    source = rear_axle if rear_axle is not None else state
    return (
        _as_float(getattr(source, "x", 0.0)),
        _as_float(getattr(source, "y", 0.0)),
        _as_float(getattr(source, "heading", 0.0)),
    )


def _distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _angle_to_agent(dx: float, dy: float, ego_heading: float) -> float:
    return abs(_wrap_angle(math.atan2(dy, dx) - ego_heading))


def _path_length(points: Sequence[Any]) -> float:
    total = 0.0
    for prev, curr in zip(points[:-1], points[1:]):
        total += _distance_xy((_as_float(prev.x), _as_float(prev.y)), (_as_float(curr.x), _as_float(curr.y)))
    return total


def _first_centerline_segment(points: Sequence[Any], max_length: float) -> List[Any]:
    if not points:
        return []
    segment = [points[0]]
    traveled = 0.0
    for prev, curr in zip(points[:-1], points[1:]):
        traveled += _distance_xy((_as_float(prev.x), _as_float(prev.y)), (_as_float(curr.x), _as_float(curr.y)))
        segment.append(curr)
        if traveled >= max_length:
            break
    return segment


def _object_type_name(obj: Any) -> str:
    object_type = getattr(obj, "tracked_object_type", None)
    return str(getattr(object_type, "name", object_type)).lower()


def _object_xy_heading(obj: Any) -> Tuple[float, float, float]:
    center = getattr(obj, "center", None)
    if center is None:
        box = getattr(obj, "box", None)
        center = getattr(box, "center", None)
    return _xy_heading_from_state(center)


def _object_velocity_xy(obj: Any) -> Tuple[float, float]:
    velocity = getattr(obj, "velocity", None)
    if velocity is None:
        return 0.0, 0.0
    return _as_float(getattr(velocity, "x", 0.0)), _as_float(getattr(velocity, "y", 0.0))


def _is_dynamic_agent_type(obj_type_name: str) -> bool:
    return any(keyword in obj_type_name for keyword in DYNAMIC_AGENT_TYPE_KEYWORDS)


def _object_collision_radius(obj: Any) -> float:
    box = getattr(obj, "box", None)
    if box is None:
        return DEFAULT_AGENT_COLLISION_RADIUS_METERS

    half_length = 0.5 * _as_float(getattr(box, "length", 0.0))
    half_width = 0.5 * _as_float(getattr(box, "width", 0.0))
    radius = math.hypot(half_length, half_width)
    return radius if radius > EPS else DEFAULT_AGENT_COLLISION_RADIUS_METERS


def _ego_velocity_xy(ego_state: Any) -> Tuple[float, float]:
    dynamic_state = getattr(ego_state, "dynamic_car_state", None)
    velocity = getattr(dynamic_state, "rear_axle_velocity_2d", None)
    if velocity is None:
        velocity = getattr(dynamic_state, "center_velocity_2d", None)
    if velocity is None:
        return 0.0, 0.0
    return _as_float(getattr(velocity, "x", 0.0)), _as_float(getattr(velocity, "y", 0.0))


def _tracked_objects_from_detection(detection: Any, missing: Counter) -> List[Any]:
    if detection is None:
        missing["detections_tracks"] += 1
        return []
    tracked_objects = getattr(detection, "tracked_objects", None)
    if tracked_objects is None:
        missing["tracked_objects"] += 1
        return []
    objects = getattr(tracked_objects, "tracked_objects", tracked_objects)
    try:
        return list(objects)
    except TypeError:
        missing["tracked_objects_iterable"] += 1
        return []


def _current_objects(metric_cache: Any, missing: Counter) -> List[Any]:
    detections = getattr(metric_cache, "current_tracked_objects", None)
    if detections:
        return _tracked_objects_from_detection(detections[0], missing)
    observation = getattr(metric_cache, "observation", None)
    detection = getattr(observation, "detections_tracks", None)
    if detection is not None:
        return _tracked_objects_from_detection(detection, missing)
    missing["current_tracked_objects"] += 1
    return []


def _future_detection_frames(metric_cache: Any, missing: Counter) -> List[Any]:
    frames = getattr(metric_cache, "future_tracked_objects", None)
    if frames is None:
        missing["future_tracked_objects"] += 1
        return []
    return list(frames)


def _trajectory_poses(trajectory: Any, missing: Counter, field_name: str) -> np.ndarray:
    if trajectory is None:
        missing[field_name] += 1
        return np.zeros((0, 3), dtype=np.float64)
    poses = getattr(trajectory, "poses", None)
    if poses is not None:
        return np.asarray(poses, dtype=np.float64)
    states = getattr(trajectory, "_trajectory", None)
    if states is not None:
        return np.asarray([_xy_heading_from_state(state) for state in states], dtype=np.float64)
    missing[field_name] += 1
    return np.zeros((0, 3), dtype=np.float64)


def load_metric_cache_paths(metric_cache_root: Path) -> Dict[str, Path]:
    """Load MetricCache paths using repository metadata; fallback to recursive scan."""
    try:
        loader = MetricCacheLoader(metric_cache_root)
        metadata_paths = {token: Path(path) for token, path in loader.metric_cache_paths.items()}
    except Exception as exc:
        print(f"Warning: MetricCacheLoader failed ({type(exc).__name__}: {exc}); scanning for metric_cache.pkl files.")
        paths = sorted(metric_cache_root.rglob("metric_cache.pkl"))
        return {path.parent.name: path for path in paths}

    missing_tokens = [token for token, path in metadata_paths.items() if not path.exists()]
    if not missing_tokens:
        return metadata_paths

    print(
        f"Warning: {len(missing_tokens)}/{len(metadata_paths)} metadata cache paths do not exist; "
        f"scanning {metric_cache_root} for metric_cache.pkl files.",
        flush=True,
    )
    scanned_paths = sorted(metric_cache_root.rglob("metric_cache.pkl"))
    scanned_by_token = {path.parent.name: path for path in scanned_paths}
    if not scanned_by_token:
        return metadata_paths

    repaired = dict(metadata_paths)
    repaired_count = 0
    for token in missing_tokens:
        scanned_path = scanned_by_token.get(token)
        if scanned_path is None:
            continue
        repaired[token] = scanned_path
        repaired_count += 1

    print(
        f"Warning: repaired {repaired_count}/{len(missing_tokens)} missing metadata paths from current cache root.",
        flush=True,
    )
    return repaired


def load_metric_cache(cache_path: Path) -> Any:
    with lzma.open(cache_path, "rb") as f:
        return pickle.load(f)


def extract_static_features(metric_cache: Any, missing: Counter) -> Dict[str, float]:
    centerline = getattr(metric_cache, "centerline", None)
    discrete_path = getattr(centerline, "discrete_path", None)
    if discrete_path is None:
        missing["centerline.discrete_path"] += 1
        discrete_path = []
    points = list(discrete_path)
    segment = _first_centerline_segment(points, CENTERLINE_LOOKAHEAD_METERS)
    heading_changes = [
        abs(_wrap_angle(_as_float(curr.heading) - _as_float(prev.heading)))
        for prev, curr in zip(segment[:-1], segment[1:])
        if hasattr(prev, "heading") and hasattr(curr, "heading")
    ]

    if not heading_changes:
        missing["centerline_heading_changes"] += 1

    route_lane_ids = getattr(metric_cache, "route_lane_ids", None)
    route_complexity = float(len(route_lane_ids)) if route_lane_ids is not None else 0.0
    if route_lane_ids is None:
        missing["route_lane_ids"] += 1

    route_length = getattr(centerline, "length", None)
    if route_length is None:
        route_length = _path_length(points) if points else 0.0
        missing["centerline.length"] += 1

    curvature_mean = float(np.mean(heading_changes)) if heading_changes else 0.0
    curvature_max = float(np.max(heading_changes)) if heading_changes else 0.0
    return {
        "curvature_mean": curvature_mean,
        "curvature_max": curvature_max,
        "route_complexity": route_complexity,
        "route_length": _as_float(route_length),
    }


def _agent_counts_and_distances(objects: Iterable[Any], ego_state: Any) -> Dict[str, float]:
    ego_x, ego_y, ego_heading = _xy_heading_from_state(ego_state)
    forward_threshold = math.radians(FORWARD_CONE_DEGREES)

    n_vehicles = 0
    n_pedestrians = 0
    distances: List[float] = []
    vehicle_distances: List[float] = []
    speeds: List[float] = []
    forward_count = 0
    quadrant_front = 0
    quadrant_left = 0

    for obj in objects:
        obj_type = _object_type_name(obj)
        if "vehicle" in obj_type:
            n_vehicles += 1
        if "pedestrian" in obj_type:
            n_pedestrians += 1

        obj_x, obj_y, _ = _object_xy_heading(obj)
        dx = obj_x - ego_x
        dy = obj_y - ego_y
        distance = float(math.hypot(dx, dy))
        distances.append(distance)
        if "vehicle" in obj_type:
            vehicle_distances.append(distance)

        rel_x = math.cos(ego_heading) * dx + math.sin(ego_heading) * dy
        rel_y = -math.sin(ego_heading) * dx + math.cos(ego_heading) * dy
        quadrant_front += int(rel_x >= 0.0)
        quadrant_left += int(rel_y >= 0.0)

        if distance <= FORWARD_CONE_DISTANCE_METERS and _angle_to_agent(dx, dy, ego_heading) <= forward_threshold:
            forward_count += 1

        obj_vx, obj_vy = _object_velocity_xy(obj)
        speeds.append(float(math.hypot(obj_vx, obj_vy)))

    return {
        "n_vehicles": float(n_vehicles),
        "n_pedestrians": float(n_pedestrians),
        "n_total_agents": float(len(distances)),
        "min_agent_distance_to_ego": min(distances) if distances else NO_AGENT_DISTANCE_FALLBACK,
        "forward_agent_count": float(forward_count),
        "agent_speed_variance": float(np.var(speeds)) if speeds else 0.0,
        "quadrant_front_count": float(quadrant_front),
        "quadrant_left_count": float(quadrant_left),
        "nearest_vehicle_distance": min(vehicle_distances) if vehicle_distances else NO_AGENT_DISTANCE_FALLBACK,
    }


def extract_agent_features(metric_cache: Any, missing: Counter) -> Dict[str, float]:
    ego_state = getattr(metric_cache, "ego_state", None)
    if ego_state is None:
        missing["ego_state_for_agents"] += 1

    objects = _current_objects(metric_cache, missing)
    features = _agent_counts_and_distances(objects, ego_state)

    future_min_distance = NO_AGENT_DISTANCE_FALLBACK
    future_max_forward_count = 0.0
    for frame in _future_detection_frames(metric_cache, missing):
        frame_objects = _tracked_objects_from_detection(frame, missing)
        frame_features = _agent_counts_and_distances(frame_objects, ego_state)
        future_min_distance = min(future_min_distance, frame_features["min_agent_distance_to_ego"])
        future_max_forward_count = max(future_max_forward_count, frame_features["forward_agent_count"])

    features["future_min_agent_distance"] = future_min_distance
    features["future_max_forward_agent_count"] = future_max_forward_count
    return features


def extract_ego_features(metric_cache: Any, missing: Counter) -> Dict[str, float]:
    ego_state = getattr(metric_cache, "ego_state", None)
    if ego_state is None:
        missing["ego_state"] += 1

    dynamic_state = getattr(ego_state, "dynamic_car_state", None)
    current_speed = _as_float(getattr(dynamic_state, "speed", 0.0))

    accel = getattr(dynamic_state, "rear_axle_acceleration_2d", None)
    if accel is None:
        accel = getattr(dynamic_state, "center_acceleration_2d", None)
    if accel is None:
        missing["ego_acceleration"] += 1
        accel_mag = 0.0
    else:
        accel_mag = float(math.hypot(_as_float(getattr(accel, "x", 0.0)), _as_float(getattr(accel, "y", 0.0))))

    past_poses = _trajectory_poses(getattr(metric_cache, "past_human_trajectory", None), missing, "past_human_trajectory")
    if past_poses.shape[0] >= 2:
        deltas = np.diff(past_poses[:, :2], axis=0)
        past_speeds = np.linalg.norm(deltas, axis=1) / 0.5
        past_speed_variance = float(np.var(past_speeds))
        past_heading_rate_mean = float(np.mean(np.abs(np.diff(np.unwrap(past_poses[:, 2]))))) / 0.5
    else:
        missing["past_ego_states"] += 1
        past_speed_variance = 0.0
        past_heading_rate_mean = 0.0

    return {
        "current_speed": current_speed,
        "ego_acceleration_magnitude": accel_mag,
        "past_speed_variance": past_speed_variance,
        "past_heading_rate_mean": past_heading_rate_mean,
    }


def _constant_velocity_ttc(pos: np.ndarray, vel: np.ndarray, radius: float) -> float:
    dist_now = float(np.linalg.norm(pos))
    if dist_now <= radius:
        return 0.0
    a = float(np.dot(vel, vel))
    if a < EPS:
        return math.inf
    b = 2.0 * float(np.dot(pos, vel))
    c = float(np.dot(pos, pos)) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return math.inf
    sqrt_disc = math.sqrt(discriminant)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    positive_roots = [root for root in roots if root >= 0.0]
    return min(positive_roots) if positive_roots else math.inf


def extract_interaction_safety_features(metric_cache: Any, missing: Counter) -> Dict[str, float]:
    ego_state = getattr(metric_cache, "ego_state", None)
    ego_x, ego_y, _ = _xy_heading_from_state(ego_state)
    ego_vx, ego_vy = _ego_velocity_xy(ego_state)

    scene_min_ttc = math.inf
    scene_drac_max = 0.0

    for obj in _current_objects(metric_cache, missing):
        obj_type = _object_type_name(obj)
        if not _is_dynamic_agent_type(obj_type):
            continue

        obj_x, obj_y, _ = _object_xy_heading(obj)
        obj_vx, obj_vy = _object_velocity_xy(obj)
        pos = np.asarray([obj_x - ego_x, obj_y - ego_y], dtype=np.float64)
        rel_vel = np.asarray([obj_vx - ego_vx, obj_vy - ego_vy], dtype=np.float64)
        distance = float(np.linalg.norm(pos))
        collision_radius = EGO_COLLISION_RADIUS_METERS + _object_collision_radius(obj)
        scene_min_ttc = min(scene_min_ttc, _constant_velocity_ttc(pos, rel_vel, collision_radius))

        if distance > EPS:
            closing_speed = -float(np.dot(pos, rel_vel)) / distance
            if closing_speed > 0.0:
                scene_drac_max = max(scene_drac_max, (closing_speed * closing_speed) / max(2.0 * distance, EPS))

    return {
        "scene_min_ttc": min(scene_min_ttc, TTC_HORIZON_SECONDS),
        "scene_drac_max": scene_drac_max,
    }


def extract_one_scene(metric_cache: Any, missing: Counter) -> np.ndarray:
    feature_dict: Dict[str, float] = {}
    feature_dict.update(extract_static_features(metric_cache, missing))
    feature_dict.update(extract_agent_features(metric_cache, missing))
    feature_dict.update(extract_ego_features(metric_cache, missing))
    feature_dict.update(extract_interaction_safety_features(metric_cache, missing))

    feature = np.asarray([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)
    if feature.shape != (N_FEATURES,):
        raise ValueError(f"Expected {N_FEATURES} features, got shape {feature.shape}")
    return np.nan_to_num(feature, nan=0.0, posinf=1e6, neginf=-1e6)


def normalize_features(features: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(features).astype(np.float32)


def run_raw_pca(features_normalized: np.ndarray) -> np.ndarray:
    n_samples, n_features = features_normalized.shape
    pca_features = np.zeros((n_samples, RAW_PCA_COMPONENTS), dtype=np.float32)
    if n_samples < 2:
        return pca_features
    n_components = min(RAW_PCA_COMPONENTS, n_samples, n_features)
    pca_values = PCA(n_components=n_components, random_state=0).fit_transform(features_normalized).astype(np.float32)
    pca_features[:, :n_components] = pca_values
    return pca_features


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(round(width * done / total))
    filled = min(width, max(0, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_progress(done: int, total: int, start_time: float, skipped: int) -> None:
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = total - done
    eta = remaining / rate if rate > 0 else 0.0
    percent = 100.0 * done / total if total > 0 else 100.0
    print(
        f"{_progress_bar(done, total)} {done}/{total} ({percent:5.1f}%) "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
        f"rate={rate:.2f} scenes/s skipped={skipped}",
        flush=True,
    )


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TiB"


def _metadata_from_cache(token: str, cache_path: Path, metric_cache: Any) -> Dict[str, Any]:
    return {
        "token": token,
        "log_name": str(getattr(metric_cache, "log_name", "")),
        "scene_type": str(getattr(metric_cache, "scene_type", "")),
        "time_s": _as_float(getattr(getattr(metric_cache, "timepoint", None), "time_s", 0.0)),
        "cache_path": str(cache_path),
    }


def _extract_worker(payload: Tuple[int, str, str]) -> Dict[str, Any]:
    index, token, cache_path_str = payload
    cache_path = Path(cache_path_str)
    missing: Counter = Counter()
    try:
        metric_cache = load_metric_cache(cache_path)
        feature = extract_one_scene(metric_cache, missing)
        metadata = _metadata_from_cache(token, cache_path, metric_cache)
    except Exception as exc:
        missing[f"load_or_extract_failed:{type(exc).__name__}"] += 1
        return {
            "ok": False,
            "index": index,
            "token": token,
            "missing": dict(missing),
            "warning": f"Warning: skipping {token}: {type(exc).__name__}: {exc}",
        }

    return {
        "ok": True,
        "index": index,
        "feature": feature,
        "metadata": metadata,
        "missing": dict(missing),
    }


def save_npz(output_path: Path, features: np.ndarray, metadata: List[Dict[str, Any]], missing: Counter) -> None:
    features_normalized = normalize_features(features)
    pca_12d = run_raw_pca(features_normalized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features.astype(np.float32),
        features_normalized=features_normalized.astype(np.float32),
        pca_12d=pca_12d.astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
        scenario_tokens=np.asarray([row["token"] for row in metadata], dtype=str),
        log_names=np.asarray([row["log_name"] for row in metadata], dtype=str),
        cache_paths=np.asarray([row["cache_path"] for row in metadata], dtype=str),
        metadata_json=np.asarray(json.dumps(metadata), dtype=str),
        missing_counts_json=np.asarray(json.dumps(dict(missing), sort_keys=True), dtype=str),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 21D CBS scene features from NAVSIM MetricCache.")
    parser.add_argument("--metric-cache-root", type=Path, required=True, help="Root directory containing MetricCache metadata/log folders.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Progress print interval.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel scene extraction workers.")
    parser.add_argument("--preflight-scenes", type=int, default=1, help="Number of initial scenes to extract synchronously before parallel workers.")
    parser.add_argument("--watchdog-interval", type=float, default=60.0, help="Seconds between no-progress watchdog messages in parallel mode.")
    args = parser.parse_args()

    if len(FEATURE_NAMES) != N_FEATURES:
        raise RuntimeError(f"FEATURE_NAMES must contain exactly {N_FEATURES} names, got {len(FEATURE_NAMES)}")

    metric_cache_paths = load_metric_cache_paths(args.metric_cache_root)
    items = list(metric_cache_paths.items())
    if args.max_scenes is not None:
        items = items[: args.max_scenes]

    num_workers = max(1, int(args.num_workers))

    print(f"MetricCache root: {args.metric_cache_root}", flush=True)
    print(f"Scenes to process: {len(items)}", flush=True)
    print(f"Workers: {num_workers}", flush=True)

    missing: Counter = Counter()
    results: List[Optional[Tuple[np.ndarray, Dict[str, Any]]]] = [None] * len(items)
    start_time = time.time()
    skipped = 0
    done = 0
    _print_progress(0, len(items), start_time, skipped)

    def collect_result(row: Dict[str, Any]) -> None:
        nonlocal skipped
        missing.update(row["missing"])
        if row["ok"]:
            results[row["index"]] = (row["feature"], row["metadata"])
        else:
            skipped += 1
            print(row["warning"], flush=True)

    worker_payloads = [(index, token, str(cache_path)) for index, (token, cache_path) in enumerate(items)]
    preflight_count = min(max(0, int(args.preflight_scenes)), len(worker_payloads))

    for payload in worker_payloads[:preflight_count]:
        _, token, cache_path_str = payload
        cache_path = Path(cache_path_str)
        size_text = _format_bytes(cache_path.stat().st_size) if cache_path.exists() else "missing"
        print(f"[PREFLIGHT] Extracting {token}", flush=True)
        print(f"[PREFLIGHT] Cache path: {cache_path}", flush=True)
        print(f"[PREFLIGHT] Cache size: {size_text}", flush=True)
        scene_start = time.time()
        collect_result(_extract_worker(payload))
        done += 1
        print(f"[PREFLIGHT] Finished {token} in {_format_duration(time.time() - scene_start)}", flush=True)

    worker_payloads = worker_payloads[preflight_count:]

    if num_workers == 1:
        for payload in worker_payloads:
            collect_result(_extract_worker(payload))
            done += 1
            if args.progress_interval > 0 and done % args.progress_interval == 0:
                _print_progress(done, len(items), start_time, skipped)
    else:
        max_pending = max(num_workers * 4, num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            payload_iter = iter(worker_payloads)
            future_to_payload = {}

            for _ in range(min(max_pending, len(worker_payloads))):
                payload = next(payload_iter)
                future_to_payload[executor.submit(_extract_worker, payload)] = payload

            while future_to_payload:
                completed, _ = wait(
                    future_to_payload,
                    timeout=max(1.0, float(args.watchdog_interval)),
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    examples = list(future_to_payload.values())[:3]
                    example_tokens = ", ".join(payload[1] for payload in examples)
                    print(
                        f"[WATCHDOG] No scene completed in {args.watchdog_interval:.0f}s; "
                        f"pending={len(future_to_payload)} examples={example_tokens}",
                        flush=True,
                    )
                    continue
                for future in completed:
                    payload = future_to_payload.pop(future)
                    try:
                        collect_result(future.result())
                    except Exception as exc:
                        _, token, _ = payload
                        skipped += 1
                        missing[f"worker_failed:{type(exc).__name__}"] += 1
                        print(f"Warning: skipping {token}: worker {type(exc).__name__}: {exc}", flush=True)
                    done += 1
                    if args.progress_interval > 0 and done % args.progress_interval == 0:
                        _print_progress(done, len(items), start_time, skipped)

                    try:
                        next_payload = next(payload_iter)
                    except StopIteration:
                        continue
                    future_to_payload[executor.submit(_extract_worker, next_payload)] = next_payload

    if items and args.progress_interval > 0 and done % args.progress_interval != 0:
        _print_progress(done, len(items), start_time, skipped)

    extracted = [row for row in results if row is not None]
    features = [row[0] for row in extracted]
    metadata = [row[1] for row in extracted]

    if not features:
        raise RuntimeError("No features extracted. Check --metric-cache-root and Python environment.")

    feature_matrix = np.vstack(features).astype(np.float32)
    if feature_matrix.shape[1] != N_FEATURES:
        raise RuntimeError(f"Expected [N, {N_FEATURES}] features, got {feature_matrix.shape}")
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    save_npz(args.output, feature_matrix, metadata, missing)
    print(f"Saved: {args.output}", flush=True)
    print(f"features shape: {feature_matrix.shape}", flush=True)
    print("Missing/fallback counts:", flush=True)
    for name, count in sorted(missing.items()):
        print(f"  {name}: {count}", flush=True)


if __name__ == "__main__":
    main()
