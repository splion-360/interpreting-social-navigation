# Interpreting Social Navigation

Trajectory forecasting experiments for MABe mouse triplets using a Social Attention-style spatio-temporal graph model.

## Setup

```bash
python -m pip install -e ".[dev]"
git submodule update --init --recursive
```

## Data

Place local MABe files under:

```text
data/MaBe/mouse_triplet_train.npy
data/MaBe/mouse_triplet_test.npy
```

`data/`, `checkpoints/`, generated outputs, and secrets should stay out of git.

## Train

```bash
python src/train.py fit --config src/config/dense_keypoint__train.yml --wandb
```

Inspect without training:

```bash
python src/train.py fit --config src/config/dense_keypoint__train.yml --show-config
```

## Evaluate

Validation split from training data:

```bash
python src/evaluate.py --config src/config/dense_keypoint__train.yml --checkpoint checkpoints/dense_keypoint/flat_best.pt
```

Held-out MABe test file:

```bash
python src/evaluate.py --config src/config/dense_keypoint__train.yml --test-config src/config/test.yml --checkpoint checkpoints/dense_keypoint/flat_best.pt --split test
```

Add `--mean` for deterministic Gaussian-mean rollout instead of sampling.
