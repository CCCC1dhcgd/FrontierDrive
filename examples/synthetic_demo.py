"""Run the numerical FrontierDrive pipeline on explicitly synthetic inputs."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from frontierdrive.capability_frontier import SUBMETRIC_COLUMNS


def run_module(module: str, *arguments: object) -> None:
    subprocess.run(
        [sys.executable, '-m', module, *(str(argument) for argument in arguments)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    generator = np.random.default_rng(42)
    feature_names = np.asarray([
        'route_length', 'min_agent_distance_to_ego', 'scene_drac_max',
        'n_total_agents', *(f'synthetic_feature_{index}' for index in range(17)),
    ])
    features = output / 'features'
    features.mkdir()
    for split, count in [('navtrain', 600), ('navtest', 120)]:
        raw = generator.normal(size=(count, 21)).astype(np.float32)
        raw[:, 0] = generator.uniform(10, 150, count)
        raw[:, 1] = generator.uniform(2, 25, count)
        raw[:, 2] = generator.uniform(0, 10, count)
        raw[:, 3] = generator.integers(1, 25, count)
        np.savez_compressed(
            features / f'{split}_raw.npz', features=raw,
            scenario_tokens=np.asarray([f'{split}_{index:04d}' for index in range(count)]),
            log_names=np.asarray([f'synthetic_{split}'] * count),
            feature_names=feature_names,
        )
    run_module(
        'frontierdrive.features.align',
        '--navtrain', features / 'navtrain_raw.npz',
        '--navtest', features / 'navtest_raw.npz',
        '--output_dir', features,
        '--scaler_pca_path', features / 'navtrain_scaler_pca.pkl',
    )
    run_root = output / 'run'
    difficulty = generator.uniform(0, 1, 120)
    for stage in range(3):
        evaluation = run_root / 'evaluate' / f'stage{stage}' / f'demo_eval_stage{stage}_navtest'
        evaluation.mkdir(parents=True)
        with (evaluation / 'metrics_results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['token', 'valid', 'score', *SUBMETRIC_COLUMNS])
            writer.writeheader()
            for index, value in enumerate(difficulty):
                score = 0.98 if value < 0.30 + 0.12 * stage else 0.55 + 0.30 * value
                writer.writerow({
                    'token': f'navtest_{index:04d}', 'valid': 'True', 'score': score,
                    **{metric: score for metric in SUBMETRIC_COLUMNS},
                })
    stage_root = run_root / 'evaluate' / 'stage3'
    run_module(
        'frontierdrive.evaluation_summary',
        '--eval-csv', run_root / 'evaluate/stage2/demo_eval_stage2_navtest/metrics_results.csv',
        '--output-dir', stage_root / 'eval',
    )
    run_module(
        'frontierdrive.capability_frontier',
        '--navtest-aligned', features / 'navtest_aligned.npz',
        '--eval-history-root', run_root, '--eval-name-prefix', 'demo_eval',
        '--stage', 3, '--n-min', 20, '--output-dir', stage_root / 'boundary',
    )
    run_module(
        'frontierdrive.frontier_retrieval',
        '--navtrain-aligned', features / 'navtrain_aligned.npz',
        '--boundary', stage_root / 'boundary/navtest_boundary.npz',
        '--output-dir', stage_root / 'retrieval',
        '--clusters', 4, '--min-train-per-sphere', 20,
    )
    with (stage_root / 'retrieval/train_manifest.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    weights = np.asarray([float(row['train_weight']) for row in rows])
    selected = np.asarray([int(row['selected']) == 1 for row in rows])
    assert len(rows) == 600
    assert len({row['token'] for row in rows}) == 600
    assert all(row['token'].startswith('navtrain_') for row in rows)
    assert np.all(np.isfinite(weights)) and np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1)
    assert selected.any() and (~selected).any()
    assert np.isclose(weights[selected].sum(), 0.5)
    print(f'Synthetic walkthrough passed. Artifacts: {output}')
    print('These synthetic scores are not experimental evidence or paper results.')


if __name__ == '__main__':
    main()
