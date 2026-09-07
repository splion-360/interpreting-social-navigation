# Repository Guidelines

## Project Structure & Module Organization

`scripts/` contains the historical MABe mouse-trajectory implementation: entry points, Structural RNN, graph construction, losses, and data utilities. Treat it as code to audit and migrate, not as the final architecture. `socialAttention/` is the upstream reference submodule. `docs/PLAN.md` records the correctness audit and planned hierarchical model. `docs/` and `misc/` hold research references, experiment notes, and media.

New first-party implementation should live under `src/social_nav/`. Place tests in `tests/`, mirroring the module under test, such as `tests/graphs/test_flat_keypoint_graph.py`. Keep generated checkpoints, logs, downloaded datasets, and plots out of version control.

## Repository Architecture

Use this directory tree as the intended project shape:

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── docs/
│   ├── PLAN.md
│   ├── data/
│   │   ├── data__preparation.runme
│   │   └── data__visualize_mabe.runme
│   └── social-attention.pdf
├── scripts/
│   ├── train.py
│   ├── sample.py
│   ├── model.py
│   ├── st_graph.py
│   ├── criterion.py
│   ├── helper.py
│   └── utils.py
├── src/
│   └── social_nav/
│       ├── config/
│       ├── data/
│       ├── evaluation/
│       ├── experiments/
│       ├── geometry/
│       ├── graphs/
│       ├── logging/
│       ├── losses/
│       ├── models/
│       └── training/
├── tests/
│   ├── data/
│   ├── evaluation/
│   ├── geometry/
│   ├── graphs/
│   ├── losses/
│   ├── models/
│   └── training/
├── socialAttention/
└── data/                 # ignored local dataset storage
```

`scripts/` is the historical runnable implementation to audit and migrate. `src/social_nav/` is the new first-party package for reusable research code. `socialAttention/` is the upstream reference submodule; keep it read-only unless intentionally updating the submodule pointer. Keep `scripts/` runnable while migrating logic into `src/social_nav/`; over time, scripts should shrink to CLI adapters that parse arguments and call package interfaces.

Use Runme-compatible notebooks for ADRs, PRDs, design-decision records, workflow docs, and data/model/train/test walkthroughs. Store them under `docs/{category}/` and name them `{category}__{task}.runme`, for example `docs/data/data__preparation.runme`.

Create and manage project tickets in the relevant GitHub Project when available. Assign created tickets to `splion-360`; also assign to the acting agent only if a real GitHub identity is available. After completing implementation work for a ticket, move the project item to `In review`, not `Done`, so the user can review it.

Tag tickets with broad category labels so the project can be filtered quickly. Use labels such as `data`, `graph`, `model`, `training`, `evaluation`, and `docs-runme`; apply more than one label when a ticket spans categories.

Migration map:

```text
scripts/utils.py      -> src/social_nav/data/
scripts/st_graph.py   -> src/social_nav/graphs/
scripts/helper.py     -> src/social_nav/geometry/ or src/social_nav/models/
scripts/model.py      -> src/social_nav/models/
scripts/criterion.py  -> src/social_nav/losses/
scripts/train.py      -> src/social_nav/training/
scripts/sample.py     -> src/social_nav/evaluation/ or inference helpers
```

Do not move everything in one edit. Migrate one module at a time, add tests at the new seam, and preserve the historical script path until the replacement is verified.

## Target Codebase Design

Design deep modules: small interfaces with substantial behavior hidden behind them. Prefer this DAG:

```text
config -> data -> geometry -> graphs -> models -> losses -> training -> evaluation -> experiments/scripts
```

Lower layers must not import higher layers. For example, `geometry` must not know about graphs or training; models must not know where datasets live; losses must not log to W&B or write checkpoints.

Use these target modules:

- `src/social_nav/config/`: typed experiment and runtime configuration.
- `src/social_nav/data/`: MABe loading, splits, window sampling, masking, and normalization. Return tensors shaped `[batch, time, mice, keypoints, coordinates]`.
- `src/social_nav/geometry/`: coordinate transforms, velocities, centroid/reference positions, anatomical edge definitions, and bone-length metrics.
- `src/social_nav/graphs/`: flat sparse, flat full, and mouse-level graph builders with explicit node/edge-count contracts.
- `src/social_nav/models/`: flat Social Attention, hierarchical mouse encoder, mouse-level Social Attention, and keypoint decoder.
- `src/social_nav/losses/`: Gaussian NLL, geodesic/structural loss, and Gaussian parameter validation.
- `src/social_nav/training/`: training loops, device selection, seeds, checkpointing, tiny-batch overfit mode, and optional W&B adapters.
- `src/social_nav/evaluation/`: ADE, FDE, anatomical consistency, efficiency metrics, representation extraction, and behavior probes.
- `src/social_nav/experiments/`: named experiment definitions for flat baseline, hierarchical baseline, and ablations.

Preferred interfaces include `DatasetAdapter.load_split(config)`, `GraphBuilder.build(sequence)`, `TrajectoryModel.forward(batch_or_graph)`, `TrajectoryLoss(prediction, target, mask)`, `Trainer.run(config)`, and `Evaluator.evaluate(checkpoint, dataset)`. Add a seam only when behavior actually varies, such as graph variants, dataset adapters, model families, losses, or loggers.

## Build, Test, and Development Commands

Use Python 3.10+ in an isolated environment. Install the package in editable mode before working on the new `src/social_nav/` code.

- `git submodule update --init --recursive` checks out the upstream reference implementation.
- `python -m pip install -e ".[dev]"` installs runtime and development dependencies.
- `cd scripts && python train.py` trains the model; it expects root-level `data/MaBe/mouse_train.npy` and CUDA-capable PyTorch.
- `cd scripts && python train.py --wandb` also logs the run to Weights & Biases.
- `cd scripts && python sample.py --epoch 199` evaluates checkpoint epoch 199 from `scripts/save/save_attention/`.
- `python -m pytest tests` runs new first-party tests once pytest is installed.

The scripts expect log/save directories to exist. Submodule tests are legacy scripts with Python 2 syntax and dataset/GPU assumptions, not a reliable root suite.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8: `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep dataset and experiment constants in `src/social_nav/config/`, not mixed into data/model/loss implementations. Group standard-library, third-party, then local imports. Document non-obvious tensor shapes such as `[batch, time, mice, keypoints, coordinates]`.

Use Google-style docstrings for every non-trivial function, class, and method. Keep research code concise: avoid production-grade defensive layers unless they protect a known research invariant, prevent silent data leakage, or make tensor contracts clear. Prefer DRY, SOLID code with focused modules over broad utility files.

Start every new source and test file with a one-line module docstring in this exact format: `"""File description: {one line summary}"""`.

Run Ruff lint and formatting checks after code changes:

- `python -m ruff check src tests`
- `python -m ruff format --check src tests`

## Testing Guidelines

Add tests before correcting historical graph, loss, or sampling behavior. Cover node/edge counts, tensor shapes, finite Gaussian parameters, valid sigma/rho values, gradient flow, and autoregressive stability. Use `test_<behavior>.py` names and deterministic seeds. There is no coverage threshold; add a regression test for every correctness bug.

Use tiny synthetic fixtures for tests. Do not require the real MABe dataset, GPUs, W&B, or external downloads in the default test suite.

## Security & Data Hygiene

Never commit MABe data, credentials, W&B keys, large checkpoints, generated runs, or large NumPy arrays. Keep real datasets under ignored `data/`, model artifacts under ignored `checkpoints/` or `outputs/`, and local secrets in ignored `.env` files. Track only small synthetic fixtures needed for tests.

## Commit & Pull Request Guidelines

History is sparse, but uses short imperative subjects and Conventional Commit prefixes where applicable (for example, `chore: restore original social navigation content`). Keep commits narrow and separate submodule-pointer updates from first-party code. Pull requests should explain the research question, data split, commands run, and observed metrics; link the relevant issue or plan section and attach plots when model behavior changes. Never commit MABe data, credentials, W&B keys, or large checkpoints.
