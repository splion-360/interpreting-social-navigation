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
data/mabe/raw/mouse_triplet_train.npy
data/mabe/raw/mouse_triplet_test.npy
```

`data/`, `checkpoints/`, generated outputs, and secrets should stay out of git.

## Train

```bash
python src/train.py fit --config src/config/train__dense_keypoint_30fps.yml --wandb
```

Inspect without training:

```bash
python src/train.py fit --config src/config/train__dense_keypoint_30fps.yml --show-config
```

## Evaluate

Validation split from training data:

```bash
python src/evaluate.py --config src/config/train__dense_keypoint_30fps.yml --checkpoint checkpoints/dense_keypoint/flat_best.pt
```

Held-out MABe test file:

```bash
python src/evaluate.py --config src/config/train__dense_keypoint_30fps.yml --test-config src/config/test__mabe.yml --checkpoint checkpoints/dense_keypoint/flat_best.pt --split test
```

Test evaluation uses `src/config/test__mabe.yml`, samples future coordinates from the predicted bivariate Gaussian, shows per-window progress, defaults to 100 windows, and reports pixel-space trajectory/pose metrics.
