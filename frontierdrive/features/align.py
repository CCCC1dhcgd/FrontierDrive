#!/usr/bin/env python
"""
Align CBS features across NAVSIM splits.

First principle:
  Raw scene features are measurements. They must stay unchanged.
  Standardization and PCA define a coordinate system. To compare splits, that
  coordinate system must be learned from navtrain only and then reused for
  navtest/navhard.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


N_RAW_FEATURES = 21
N_PCA_COMPONENTS = 12


def load_raw_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Raw feature file does not exist: {path}")

    data = np.load(path)
    required = ["features", "scenario_tokens", "log_names", "feature_names"]
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")

    features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != N_RAW_FEATURES:
        raise ValueError(f"{path}: expected features shape [N, {N_RAW_FEATURES}], got {features.shape}")

    return {
        "features": np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6),
        "scenario_tokens": data["scenario_tokens"],
        "log_names": data["log_names"],
        "feature_names": data["feature_names"],
    }


def save_aligned_npz(
    output_path: Path,
    raw_data: dict,
    scaler: StandardScaler,
    pca: PCA,
    navtrain_feature_names: np.ndarray,
) -> None:
    features_raw = raw_data["features"]
    features_norm = scaler.transform(features_raw).astype(np.float32)
    features_pca = pca.transform(features_norm).astype(np.float32)

    np.savez_compressed(
        output_path,
        features_raw=features_raw.astype(np.float32),
        features_norm=features_norm,
        features_pca=features_pca,
        scenario_tokens=raw_data["scenario_tokens"],
        log_names=raw_data["log_names"],
        feature_names=navtrain_feature_names,
    )
    print(f"Saved {output_path}")
    print(f"  features_raw:  {features_raw.shape}")
    print(f"  features_norm: {features_norm.shape}")
    print(f"  features_pca:  {features_pca.shape}")


def maybe_load_optional(path: Optional[Path], split_name: str) -> Optional[dict]:
    if path is None:
        print(f"[INFO] Skip {split_name}: no path provided")
        return None
    if not path.exists():
        print(f"[WARN] Skip {split_name}: raw feature file does not exist: {path}")
        return None
    return load_raw_npz(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit navtrain-only StandardScaler + whitened PCA(12), then align all CBS splits.")
    parser.add_argument("--navtrain", type=Path, required=True, help="Raw navtrain .npz from extract_features.py.")
    parser.add_argument("--navtest", type=Path, default=None, help="Optional raw navtest .npz.")
    parser.add_argument("--navhard", type=Path, default=None, help="Optional raw navhard .npz.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for *_aligned.npz outputs.")
    parser.add_argument("--scaler_pca_path", type=Path, required=True, help="Pickle path for fitted scaler and PCA.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scaler_pca_path.parent.mkdir(parents=True, exist_ok=True)

    navtrain = load_raw_npz(args.navtrain)
    navtest = maybe_load_optional(args.navtest, "navtest")
    navhard = maybe_load_optional(args.navhard, "navhard")

    train_features = navtrain["features"]
    if train_features.shape[0] < N_PCA_COMPONENTS:
        raise ValueError(
            f"Need at least {N_PCA_COMPONENTS} navtrain scenes to fit PCA(12), "
            f"got {train_features.shape[0]}"
        )

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_features).astype(np.float32)

    pca = PCA(n_components=N_PCA_COMPONENTS, whiten=True, random_state=42)
    pca.fit(train_norm)

    payload = {
        "scaler": scaler,
        "pca": pca,
        "fit_split": "navtrain",
        "n_raw_features": N_RAW_FEATURES,
        "n_pca_components": N_PCA_COMPONENTS,
        "pca_whiten": True,
        "feature_names": navtrain["feature_names"],
    }
    with args.scaler_pca_path.open("wb") as handle:
        pickle.dump(payload, handle)
    print(f"Saved {args.scaler_pca_path}")

    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    print("\nPCA explained variance ratio:")
    print("  whiten=True")
    for index, (ratio, total) in enumerate(zip(ratios, cumulative), start=1):
        print(f"  PC{index:02d}: ratio={ratio:.6f}, cumulative={total:.6f}")

    save_aligned_npz(args.output_dir / "navtrain_aligned.npz", navtrain, scaler, pca, navtrain["feature_names"])
    if navtest is not None:
        save_aligned_npz(args.output_dir / "navtest_aligned.npz", navtest, scaler, pca, navtrain["feature_names"])
    if navhard is not None:
        save_aligned_npz(args.output_dir / "navhard_aligned.npz", navhard, scaler, pca, navtrain["feature_names"])


if __name__ == "__main__":
    main()
