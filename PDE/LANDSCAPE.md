# PDE Landscape

`PDE/landscape.py` is the post-hoc landscape entry for PDE runs.

It supports two different scalar targets:

- `train_loss`
- `val_error`

## Why two modes

`train_loss`

- Uses the actual DeepXDE training objective.
- Good for studying optimization geometry.

`val_error`

- Uses analytical-reference error when a reference solution exists.
- Good for studying the geometry of the final evaluation target.

These two surfaces are not the same, so both are useful.

## Example

```bash
python PDE/landscape.py \
  --resume results/.../checkpoint.pth \
  --output results/pde-landscape \
  --modes train_loss,val_error \
  --grid-size 41
```

## Main options

- `--resume`
- `--best-model`
- `--output`
- `--modes`
- `--grid-size`
- `--s-range`
- `--t-range`
- `--direction-seed`
- `--plot-3d`

## Outputs

For `train_loss`:

- `landscape_train_loss.npz`
- `landscape_train_loss_2d.png`
- optional `landscape_train_loss_3d.png`

For `val_error`:

- `landscape_val_error.npz`
- `landscape_val_error_2d.png`
- optional `landscape_val_error_3d.png`

## Notes

- `val_error` is only available for PDEs with an analytical reference solution.
- `checkpoint.pth` is now saved by `pde.py`, `pde_taiyi.py`, and `pde_ntk.py` so the landscape tool can recover the exact run config.
