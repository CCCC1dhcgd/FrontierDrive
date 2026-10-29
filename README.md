# FrontierDrive

**Capability-frontier-guided curriculum learning for autonomous driving.**

FrontierDrive uses sequential policy evaluations to estimate a capability
frontier, retrieves related training scenarios in a shared feature space, and
constructs a curriculum combining frontier-focused sampling with retention
replay. The repository also includes an agent-based curriculum controller,
cross-stage memory, and training adapters for external NAVSIM/SimScale planners.

This repository contains the **core method and integration code**, not a complete
simulator or a bundled training environment. Datasets, metric caches, planner
implementations, pretrained weights, and experimental checkpoints are not included.

> **Current status:** the deterministic synthetic workflow has been checked.
> The agent preparation pipeline has a known manifest-path type error and is
> not currently an end-to-end working entry point. See [Known limitations](#known-limitations).

## Repository structure

```text
FrontierDrive/
├── frontierdrive/
│   ├── features/
│   │   ├── extract.py             # Extract scene descriptors from metric caches
│   │   ├── align.py               # Train-fitted standardization and whitened PCA
│   │   └── diagnostics.py         # Feature diagnostics and visualization
│   ├── capability_frontier.py     # Estimate capability and identify frontier scenes
│   ├── frontier_retrieval.py      # Retrieve training support and build sampling weights
│   ├── curriculum_sampler.py      # Weighted sampling with per-epoch frontier rotation
│   ├── curriculum_controller.py   # Agent-based curriculum preparation
│   ├── schemas.py                 # Structured reports and curriculum contracts
│   ├── learning_memory.py         # Persistent cross-stage state
│   ├── execution_harness.py       # Apply plans and record execution status
│   ├── evaluation_summary.py      # Summarize evaluation CSVs
│   ├── read_curriculum_plan.py    # Read individual proposal fields
│   ├── reweight_manifest.py       # Adjust frontier/replay sampling mass
│   ├── visualization.py           # Visualize retrieved training support
│   └── integrations/navsim/
│       ├── train_diffusiondrive.py
│       ├── train_ltf.py
│       └── train_meanfuser.py
├── examples/synthetic_demo.py
├── scripts/prepare_stage.sh
├── docs/source_manifest.json
├── .env.example
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

Run the following commands from the repository root in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package declares Python `>=3.9`. The numerical workflow was checked using
Python 3.12, NumPy 1.26.4, scikit-learn 1.5.2, and SciPy 1.13.1. These checks
do not establish compatibility with the external planner training stack.

The core dependencies are listed in [requirements.txt](requirements.txt).
The numerical workflow does not require a GPU, PyTorch, NAVSIM, or an API key.
Optional dependencies can be installed separately:

```bash
python -m pip install -e '.[sampler]'
python -m pip install -e '.[diagnostics]'
```

`sampler` adds PyTorch; `diagnostics` adds UMAP. For GPU training, use a PyTorch
installation compatible with your external simulator environment and hardware.
Installing these extras does **not** install NAVSIM, planner models, or their
complete training dependencies.

## Quick start: offline synthetic example

```bash
python examples/synthetic_demo.py --output-dir outputs/synthetic-demo
```

The example generates 600 synthetic training scenes, 120 synthetic evaluation
scenes, and three evaluation rounds. It then:

1. Fits a shared feature transform using training scenes only.
2. Summarizes evaluation scores and estimates the capability frontier.
3. Retrieves training support and creates a weighted sampling manifest.
4. Checks token separation, nonnegative finite weights, normalized total mass,
   and the requested frontier/replay allocation.

Expected outputs include:

```text
outputs/synthetic-demo/
├── features/
│   ├── navtrain_aligned.npz
│   ├── navtest_aligned.npz
│   └── navtrain_scaler_pca.pkl
└── run/evaluate/stage3/
    ├── eval/eval_summary.json
    ├── boundary/
    │   ├── navtest_boundary.npz
    │   └── boundary_summary.json
    └── retrieval/
        ├── train_manifest.csv
        ├── train_selection.npz
        ├── selected_train_tokens.txt
        ├── cluster_centers.npy
        └── sphere_summary.json
```

The output directory must not already exist. To rerun the example, select a new
directory. This example does not train a driving policy or invoke an LLM, and
its synthetic scores are **not experimental results**.

## Deterministic workflow with real data

### 1. Extract and align scene features

Feature extraction requires a compatible external NAVSIM/SimScale installation
and its metric caches. Replace the cache paths with your own:

```bash
python -m frontierdrive.features.extract \
  --metric-cache-root /path/to/navtrain_metric_cache \
  --output outputs/features/navtrain_raw.npz

python -m frontierdrive.features.extract \
  --metric-cache-root /path/to/frontier_evaluation_metric_cache \
  --output outputs/features/navtest_raw.npz

python -m frontierdrive.features.align \
  --navtrain outputs/features/navtrain_raw.npz \
  --navtest outputs/features/navtest_raw.npz \
  --output_dir outputs/features/aligned \
  --scaler_pca_path outputs/features/aligned/navtrain_scaler_pca.pkl
```

The extractor produces 21-dimensional descriptors. Alignment fits a
`StandardScaler` and whitened 12-component PCA on training data, then applies
the same transform to the other splits. At least 12 training scenes are
required. Use the aligned features for retrieval, not the extractor's separate
per-split diagnostic PCA. Only load trusted metric-cache and scaler pickle files.

### 2. Provide sequential evaluation results

Preparing stage `t` requires evaluation results for stages `0` through `t-1`.
Stage 0 conventionally evaluates the initial/warmup policy. The structured
layout is:

```text
outputs/my_run/evaluate/
├── stage0/my_eval_stage0_navtest/metrics_results.csv
├── stage1/my_eval_stage1_navtest/metrics_results.csv
└── ...
```

Each CSV must contain `token`, `valid`, `score`, and these eight submetrics:

```text
no_at_fault_collisions, drivable_area_compliance, driving_direction_compliance,
traffic_light_compliance, ego_progress, time_to_collision_within_bound,
lane_keeping, history_comfort
```

Valid rows use the string `True`. Evaluation tokens used by the frontier
estimator must have valid records throughout the required history.
The `navtest` suffix is a legacy naming convention; use a designated
frontier-estimation split and keep final test data separate from curriculum
selection.

### 3. Identify the capability frontier

```bash
python -m frontierdrive.capability_frontier \
  --navtest-aligned outputs/features/aligned/navtest_aligned.npz \
  --eval-history-root outputs/my_run \
  --eval-name-prefix my_eval \
  --stage 1 \
  --output-dir outputs/my_run/evaluate/stage1/boundary
```

The estimator uses a feature/score-based prior when fewer than three evaluation
rounds are available, and fits a regularized Rasch model from three rounds onward.
The frontier probability band can expand to meet the requested minimum size.

### 4. Retrieve training support

```bash
python -m frontierdrive.frontier_retrieval \
  --navtrain-aligned outputs/features/aligned/navtrain_aligned.npz \
  --boundary outputs/my_run/evaluate/stage1/boundary/navtest_boundary.npz \
  --output-dir outputs/my_run/evaluate/stage1/retrieval
```

Retrieval groups frontier scenes, constructs adaptive support regions, and
assigns weights to selected training scenes and the remaining replay pool.
The resulting `train_manifest.csv` contains the fields `token`, `selected`, and
`train_weight`, together with retrieval metadata. Both selected and replay pools
must remain nonempty for the sampler.

### 5. Use the weighted sampler

With the PyTorch dependency installed:

```python
import csv
from frontierdrive.curriculum_sampler import TBATSampler

manifest_path = "outputs/my_run/evaluate/stage1/retrieval/train_manifest.csv"
with open(manifest_path, newline="") as handle:
    train_tokens = [row["token"] for row in csv.DictReader(handle)]

sampler = TBATSampler(
    tokens=train_tokens,
    manifest_path=manifest_path,
    rotation_fraction=0.85,
    epoch_size=len(train_tokens),
    seed=42,
)
sampled_indices = list(sampler)
```

For an actual dataset, pass tokens in **dataset index order**, rather than
assuming its order matches the CSV. Use `DataLoader(..., sampler=sampler,
shuffle=False)`. Sampling is with replacement; `epoch_size` counts draws, not
unique scenes. Frontier rotation preserves the selected/replay mass while
changing the active subset of selected scenes between iterations.

## Main defaults

| Setting | Default | Interface |
| --- | --- | --- |
| Binary pass threshold | `0.95` | `--tau` |
| Initial frontier probability interval | `[0.20, 0.60]` | `--p-lo`, `--p-hi` |
| Minimum requested frontier size | `300` | `--n-min` |
| Initial frontier sampling mass | `0.50` | `--boundary-mass` |
| Frontier rotation fraction | `0.85` | `TBATSampler(rotation_fraction=...)` |
| Preparation-wrapper sampler seed | `42` | `TBAT_SAMPLER_SEED` |
| LLM verifier in preparation wrapper | Disabled | `AUDIT_AGENT_ENABLED=0` |

The sampler class itself defaults to seed `0`; the preparation wrapper uses
`42`. Use `python -m frontierdrive.<module> --help` to inspect a module's CLI.
Legacy names such as `TBATSampler`, `tbat_*`, and `boundary` remain in interfaces
and saved artifacts for compatibility.

## Agent-based curriculum preparation

The controller contains four roles: Capability Analyst, Frontier Mapper,
Curriculum Planner, and Curriculum Verifier. It uses structured reports and
cross-stage memory to produce a proposal that the execution harness can apply.

**This entry point currently requires the manifest-path bug described below to
be fixed. The following is a configuration template, not a verified working
end-to-end command.**

```bash
cp .env.example .env
```

Edit `.env` locally with your API key and an enabled model ID, then export it:

```bash
set -a
source .env
set +a

export NAVSIM_EXP_ROOT="$PWD/outputs"
export TBAT_RUN_NAME=my_run
export TBAT_EVAL_NAME_PREFIX=my_eval
export FEATURE_DIR="$PWD/outputs/features/aligned"
export PREVIOUS_CHECKPOINT=/absolute/path/to/warmup/last.ckpt
export STAGE=1

bash scripts/prepare_stage.sh
```

The wrapper expects real checkpoint files and existing evaluation history. It
is designed to prepare artifacts including `agent/training_proposal.json` and
`retrieval/curriculum_manifest.csv`; it does not submit training or evaluation
jobs. `AUDIT_AGENT_ENABLED=1` enables the additional LLM verifier call, while
`0` uses deterministic checks without that call. `DEEPSEEK_MOCK=1` is for
interface debugging only; it does not resolve the known path-type error.

API calls can incur costs and transmit evaluation summaries, scenario metadata,
and learning-history information to the configured provider. Review the inputs
before enabling live calls, and never commit `.env` or API credentials.

## External planner integration

The following adapters connect the sampler to compatible external planners:

| Planner | Python module |
| --- | --- |
| DiffusionDrive | `frontierdrive.integrations.navsim.train_diffusiondrive` |
| LTF / TransFuser | `frontierdrive.integrations.navsim.train_ltf` |
| MeanFuser | `frontierdrive.integrations.navsim.train_meanfuser` |

These require the external `navsim` package, planner-specific Lightning modules,
Hydra configurations, datasets/caches, and checkpoint assets. Configurations
are located relative to the installed `navsim` package. Setting
`NAVSIM_DEVKIT_ROOT` alone does not install those dependencies. In particular,
the MeanFuser adapter does not include a MeanFuser model implementation.

The adapters preserve the configuration keys `tbat_manifest_path`,
`tbat_rotation_fraction`, `tbat_epoch_size`, and `tbat_sampler_seed`.
`read_curriculum_plan.py` extracts proposal fields, and `execution_harness.py`
records `RUNNING`, `SUCCEEDED`, or `FAILED` execution states. Actual training
and evaluation must be run with your compatible external environment.

## Known limitations

- **Agent preparation is currently blocked:** `FrontierReport.support_manifest`
  is a string, but `build_curriculum_planner_input()` passes it directly to
  `validate_manifest()`, which calls `path.is_file()`. This produces
  `AttributeError: 'str' object has no attribute 'is_file'`. It does not affect
  the deterministic synthetic example.
- **Distributed sampling needs integration work:** `TBATSampler` is not
  rank-sharded. Do not assume its per-instance draw count is a global multi-GPU
  budget or that ranks automatically receive different sample streams.
- **Validation is limited:** syntax checks, the synthetic numerical workflow,
  and numerical-output comparison against the source implementation passed
  during code organization. Real GPU training and live-agent execution were
  not validated. The full agent preparation smoke check exposed the issue above.
- **Research assets are external:** this snapshot does not provide a complete
  experiment reproduction bundle or guarantee compatibility with arbitrary
  NAVSIM/SimScale revisions.

## Provenance and license

[docs/source_manifest.json](docs/source_manifest.json) records the original
implementation paths and source/release SHA-256 hashes from the local
SimScale-based working tree.

The existing Apache-2.0 license text is retained in [LICENSE](LICENSE).
External simulator, planner, dataset, and model assets are not relicensed by
this repository; their respective terms and notices still apply.
