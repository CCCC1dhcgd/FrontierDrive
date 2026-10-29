<div align="center">

# FrontierDrive

### Capability-Frontier-Guided Curriculum Learning for Autonomous Driving

**Turn driving evaluations into targeted training curricula.**

[Highlights](#highlights) · [Method](#method) · [Quick Start](#quick-start) · [Code](#code) · [License](#license)

</div>

As a driving policy evolves, its training curriculum should evolve with it.
**FrontierDrive** identifies scenarios near the policy's current capability
frontier from sequential evaluations, retrieves related training scenes, and
combines focused sampling with retention replay. This connects capability
assessment to an explicit decision: **which scenarios to train on next.**

<a id="highlights"></a>

## ✨ Highlights

- **A curriculum that tracks capability.** Estimate the policy's capability
  frontier from evaluation history and update the training focus at each stage.
- **From evaluation to training data.** Match frontier scenarios to training
  scenes in a shared feature space, turning capability estimates into a weighted
  sampling manifest.
- **Focused learning with replay.** Allocate sampling mass between frontier
  support and the replay pool, with a configurable per-epoch draw budget and
  rotation of frontier-support scenes across epochs.

<a id="method"></a>

## 🎯 Method

1. **Locate the frontier.** Estimate scenario difficulty and policy capability
   using evaluation history, with a regularized Rasch model once three rounds
   are available.
2. **Retrieve training support.** Align scene features with a transform fitted
   on training data, then construct adaptive support regions around frontier
   scenarios to select related training scenes.
3. **Build the next curriculum.** Assign frontier and replay weights, then use
   the weighted sampler in the planner's training loop.

Use a dedicated split for frontier estimation and a separate held-out test split
for final evaluation.

<a id="quick-start"></a>

## 🚀 Quick Start

Run from the repository root with **Python 3.9+**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python examples/synthetic_demo.py --output-dir outputs/synthetic-demo
```

The CPU-only synthetic walkthrough runs feature alignment, frontier estimation,
and training-scene retrieval with **600 training scenes, 120 evaluation scenes,
and three evaluation rounds**. It checks sampling weights and frontier/replay
allocation without a GPU, simulator, or API key.

Inspect the generated curriculum at
`outputs/synthetic-demo/run/evaluate/stage3/retrieval/train_manifest.csv`.
Use a new output directory for each run. The demo illustrates curriculum
construction with synthetic inputs; driving-policy training uses the external
planner environment below.

### Planner Integration

Install the PyTorch sampler with `python -m pip install -e '.[sampler]'`.
Training adapters are provided for
[DiffusionDrive](frontierdrive/integrations/navsim/train_diffusiondrive.py),
[LTF / TransFuser](frontierdrive/integrations/navsim/train_ltf.py), and
[MeanFuser](frontierdrive/integrations/navsim/train_meanfuser.py).
Use them with a compatible NAVSIM/SimScale environment, planner implementation,
datasets, caches, and checkpoints.

<a id="code"></a>

## 🧩 Code

| Component | Entry point |
| --- | --- |
| Scene features and alignment | [Feature pipeline](frontierdrive/features) |
| Capability-frontier estimation | [Frontier estimator](frontierdrive/capability_frontier.py) |
| Training-scene retrieval | [Frontier retrieval](frontierdrive/frontier_retrieval.py) |
| Weighted sampling and replay | [Curriculum sampler](frontierdrive/curriculum_sampler.py) |
| Agent-based curriculum logic | [Controller](frontierdrive/curriculum_controller.py) · [Cross-stage memory](frontierdrive/learning_memory.py) |

<a id="license"></a>

## 📄 License

Released under the [Apache-2.0 License](LICENSE). External simulator, planner,
dataset, and model assets remain subject to their respective licenses.
