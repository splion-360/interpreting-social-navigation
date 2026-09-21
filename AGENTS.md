# Repository Guidelines

## Project Structure & Module Organization

`scripts/` contains the historical MABe mouse-trajectory implementation: entry points, Structural RNN, graph construction, losses, and data utilities. Treat it as code to audit and migrate, not as the final architecture. `socialAttention/` is the upstream reference submodule. `docs/PLAN.md` records the correctness audit and planned hierarchical model. `docs/` and `misc/` hold research references, experiment notes, and media.

New first-party implementation should live directly under `src/`. Keep folders for concerns that already have multiple files or a near-term reason to grow, such as `data/`, `config/`, and `models/`. Place tests in `tests/`, mirroring the module under test, such as `tests/test_st_graph.py`. Keep generated checkpoints, logs, downloaded datasets, and plots out of version control.

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
│   ├── findings/
│   │   └── findings__YYYY-MM-DD.html
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
│   ├── config/
│   │   ├── train/
│   │   ├── benchmark/
│   │   └── test/
│   ├── baselines.py
│   ├── benchmark.py
│   ├── benchmark_report.py
│   ├── data/
│   │   ├── mabe.py
│   │   ├── motion_sampling.py
│   │   ├── schema.py
│   │   ├── temporal_sampling.py
│   │   └── visualization.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── loss.py
│   ├── metrics.py
│   ├── models/
│   │   └── flat.py
│   ├── st_graph.py
│   └── train.py
├── tests/
│   ├── data/
│   ├── test_loss.py
│   ├── test_models_flat.py
│   ├── test_st_graph.py
│   └── test_train.py
├── socialAttention/
└── data/
    └── mabe/
        └── raw/          # ignored local dataset storage
```

`scripts/` is the historical runnable implementation to audit and migrate. `src/` is the new first-party root for reusable research code. `socialAttention/` is the upstream reference submodule; keep it read-only unless intentionally updating the submodule pointer. Keep `scripts/` runnable while migrating logic into `src/`; over time, scripts should shrink to CLI adapters that parse arguments and call package interfaces.

Use Runme-compatible notebooks for ADRs, PRDs, design-decision records, workflow docs, and data/model/train/test walkthroughs. Store them under `docs/{category}/` and name them `{category}__{task}.runme`, for example `docs/data/data__preparation.runme`.
Store immutable research checkpoints under `docs/findings/` and name them `findings__YYYY-MM-DD.html`. Use the `findings-report` skill to create or revise these reports.

Create and manage project tickets in the relevant GitHub Project when available. Prefer high-level tickets with concrete subtasks and acceptance criteria over many granular tickets that duplicate one roadmap. Assign created tickets to `splion-360`; also assign to the acting agent only if a real GitHub identity is available. Update ticket checklists as work progresses so completed and remaining subtasks are visible. After completing implementation work for a ticket, move the project item to `In review`, not `Done`, so the user can review it. Move cancelled or superseded tickets to `Cancelled`, not `Done`; if the project has no `Cancelled` status, close them as not planned and report that the status column is missing.

Tag tickets with broad category labels so the project can be filtered quickly. Use labels such as `data`, `graph`, `model`, `training`, `evaluation`, and `docs-runme`; apply more than one label when a ticket spans categories.

Migration map:

```text
scripts/utils.py      -> src/data/
scripts/st_graph.py   -> src/st_graph.py
scripts/helper.py     -> src/geometry.py or src/models/
scripts/model.py      -> src/models/
scripts/criterion.py  -> src/loss.py
scripts/train.py      -> src/train.py
scripts/sample.py     -> src/evaluate.py or inference helpers
```

Do not move everything in one edit. Migrate one module at a time, add tests at the new seam, and preserve the historical script path until the replacement is verified.

## Target Codebase Design

Design deep modules: small interfaces with substantial behavior hidden behind them. Prefer this DAG:

```text
config -> data -> geometry -> st_graph -> models -> loss -> train -> evaluate -> experiments/scripts
```

Lower layers must not import higher layers. For example, geometry helpers must not know about graph builders or training; models must not know where datasets live; losses must not log to W&B or write checkpoints.

Use folders only when a concern has multiple files or a stable internal API. Start with a single module for thin orchestration paths such as training and evaluation, then promote to a package only after the file becomes crowded.

Use these current modules:

- `src/config/`: YAML configuration grouped by purpose. Store training, benchmark, and test files under `train/`, `benchmark/`, and `test/`, respectively. Use descriptive filenames without repeating the directory name, for example `train/flat_dense_triplet_30fps.yml`, `test/mabe.yml`, and `benchmark/flat_dense_triplet_5fps.yml`.
- `src/data/`: MABe loading, splits, window sampling, masking, and normalization. Return tensors shaped `[batch, time, mice, keypoints, coordinates]`.
- `src/st_graph.py`: graph dataclasses and flat/mouse-level graph builders with explicit node/edge-count contracts.
- `src/models/`: flat model now, hierarchical mouse/keypoint models next.
- `src/loss.py`: Gaussian NLL, structural losses, and Gaussian parameter validation until losses grow enough to split.
- `src/train.py`: training CLI, device selection, seeds, warm-up smoke runs, and later real experiment entry points.
- `src/evaluate.py`: checkpoint and baseline evaluation on validation or held-out test windows.
- `src/benchmark.py` and `src/benchmark_report.py`: fair model-versus-baseline benchmark orchestration and table/report construction.

Add future modules only when needed:

- `src/geometry.py` or `src/geometry/`: coordinate transforms, velocities, anatomical edge definitions, and bone-length metrics.
- `src/experiments.py` or `src/experiments/`: named experiment definitions for flat baseline, hierarchical baseline, and ablations.

Preferred interfaces include `DatasetAdapter.load_split(config)`, `GraphBuilder.build(sequence)`, `TrajectoryModel.forward(batch_or_graph)`, `TrajectoryLoss(prediction, target, mask)`, `Trainer.run(config)`, and `Evaluator.evaluate(checkpoint, dataset)`. Add a seam only when behavior actually varies, such as graph variants, dataset adapters, model families, losses, or loggers.

## Build, Test, and Development Commands

Use Python 3.10+ in an isolated environment. Install the package in editable mode before working on the new `src/` code.

- `git submodule update --init --recursive` checks out the upstream reference implementation.
- `python -m pip install -e ".[dev]"` installs runtime and development dependencies.
- `cd scripts && python train.py` trains the model; it expects root-level `data/MaBe/mouse_train.npy` and CUDA-capable PyTorch.
- `cd scripts && python train.py --wandb` also logs the run to Weights & Biases.
- `cd scripts && python sample.py --epoch 199` evaluates checkpoint epoch 199 from `scripts/save/save_attention/`.
- `python src/train.py warmup --data data/mabe/raw/mouse_triplet_train.npy --device cpu --steps 5` runs a short training smoke test.
- `python src/train.py fit --show-config` prints the resolved training setup from `src/config/train/flat_dense_triplet_30fps.yml` without training.
- `python src/train.py fit --wandb` runs flat-model training with W&B logging using `src/config/train/flat_dense_triplet_30fps.yml`.
- `python src/train.py fit --wandb --resume-wandb-artifact flat-best-checkpoint:best` resumes from the best W&B model artifact.
- `python -m pytest tests` runs new first-party tests once pytest is installed.

The scripts expect log/save directories to exist. Submodule tests are legacy scripts with Python 2 syntax and dataset/GPU assumptions, not a reliable root suite.

Training commands should run locally with visible CLI progress. Load default training parameters from variant-specific YAML files under `src/config/train/`, named `{variant}_{fps}fps.yml` when frame rate matters, then use argparse only for `--config`, `--show-config`, and explicit overrides. Use `tqdm` for batch progress and print epoch-level train/validation losses. W&B is the monitoring platform, but it must remain opt-in through a `--wandb` boolean flag; never require W&B for tests, warm-up runs, or local debugging. When W&B logging and checkpointing are enabled, upload the best checkpoint as a W&B model artifact so model versions are preserved outside the local workspace.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8: `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep `src/config/` YAML-only; MABe schema constants belong in `src/data/schema.py`, not mixed into data/model/loss implementations. Group standard-library, third-party, then local imports. Document non-obvious tensor shapes such as `[batch, time, mice, keypoints, coordinates]`.

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
