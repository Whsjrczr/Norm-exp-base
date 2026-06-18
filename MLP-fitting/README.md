# MLP-fitting

`MLP-fitting` is the synthetic regression / fitting task in this repo.

The main training entry is:

- `MLP-fitting/fitting.py`

The legacy name:

- `MLP-fitting/mlpfitting.py`

now forwards to the same unified pipeline.

## Task

This task trains a one-hidden-layer MLP on synthetic random regression data:

- input: random vectors in `[-1, 1]`
- target: random scalar values in `[-1, 1]`

It is mainly used for optimization-geometry experiments rather than benchmark accuracy.

## Training

Example:

```bash
python MLP-fitting/fitting.py \
  --in-dim 8 \
  --num-samples 512 \
  --width 512 \
  --lr 1e-3 \
  --epochs 200 \
  --batch-size "64,256,256"
```

## Geometry Tracking

Enable geometry tracking with:

```bash
python MLP-fitting/fitting.py \
  --track-geometry \
  --track-geometry-every 1
```

Recorded geometry scalars:

- `distance_from_init`
- `update_rate`
- `curve_rate`
- `cosine_similarity`

Outputs:

- `geometry_stats.csv`
- `geometry_stats.json`

## Landscape

Use the post-hoc landscape tool after training:

```bash
python MLP-fitting/landscape.py \
  --resume results/.../checkpoint.pth \
  --output results/mlp-fitting-landscape \
  --grid-size 41 \
  --batch-size 256 \
  --plot-3d
```

Outputs:

- `loss_landscape.npz`
- `loss_landscape_2d.png`
- optional `loss_landscape_3d.png`
