# Norm Layout Update Summary

Date: 2026-04-01

## Problem

`ext.normalization.Norm(...)` used to rely on an implicit CNN-style assumption.
That broke multiple model families:

- MLP / KAN hidden states are 2D `(N, C)`.
- ViT token states are 3D token-last `(B, N, C)`.
- CNN feature maps are 4D channel-first `(N, C, H, W)`.

Without explicit layout information, `BN/GN/IN` factories could instantiate the wrong variant or normalize along the wrong axis.

## What Changed

### 1. Normalization factory now supports explicit layout

Updated [`extension/normalization.py`](E:\norm-exp\extension\normalization.py):

- Added explicit `dim=2/3/4`
- Added explicit `layout="first"|"last"`
- Added automatic adapters for channel-first norms on token-last inputs
- Added `ext.make_norm_factory(...)` for binding layout once at model entrypoints

Recommended usage:

```python
norm_2d = ext.make_norm_factory(dim=2)
norm_vit = ext.make_norm_factory(dim=3, layout="last")
norm_4d = ext.make_norm_factory(dim=4)
```

### 2. Model entrypoints now bind the correct layout

- [`MLP/model/MLP.py`](E:\norm-exp\MLP\model\MLP.py)
  - fully-connected MLP paths now use `dim=2`
- [`KAN/model/select_kan.py`](E:\norm-exp\KAN\model\select_kan.py)
  - KAN/MLP selector now passes a 2D norm factory instead of raw `ext.normalization.Norm`
- [`ViT/model_vit/select_vit.py`](E:\norm-exp\ViT\model_vit\select_vit.py)
  - ViT now uses `dim=3, layout="last"`
- [`MLP/model/test_bn.py`](E:\norm-exp\MLP\model\test_bn.py)
- [`MLP/model/test_ln.py`](E:\norm-exp\MLP\model\test_ln.py)
  - conv-style test models now bind `dim=4`

### 3. Custom normalization module fixes included

- [`extension/my_modules/norm/bn1d_modules.py`](E:\norm-exp\extension\my_modules\norm\bn1d_modules.py)
  - supports both `[B, C, L]` and `[B, N, C]`
- [`extension/my_modules/norm/bn2d_modules.py`](E:\norm-exp\extension\my_modules\norm\bn2d_modules.py)
  - fixed running stats shape corruption
- [`extension/my_modules/norm/gn_modules.py`](E:\norm-exp\extension\my_modules\norm\gn_modules.py)
  - fixed `GroupNormScaling(..., bias=True)` initialization
- [`extension/my_modules/activation/pgn_modules.py`](E:\norm-exp\extension\my_modules\activation\pgn_modules.py)
  - fixed affine broadcasting for NCHW inputs
- [`extension/my_modules/norm/pln.py`](E:\norm-exp\extension\my_modules\norm\pln.py)
  - supports common 2D/3D/4D layouts

### 4. Training loop fix

- [`ViT/vit.py`](E:\norm-exp\ViT\vit.py)
  - moved non-`auto` scheduler stepping to after `optimizer.step()`
  - removed the PyTorch warning about scheduler order

## Review Finding Status

Resolved:

- KAN/MLP selector did not pass layout information into `Norm()`
- MLP 2D paths defaulted to 4D BN/IN factories
- ViT token-last paths were incompatible with BN/GN/IN factory assumptions
- Multiple custom norm implementation bugs found during follow-up review

## Validation

### Unit / regression tests

Added:

- [`tests/test_norm_modules.py`](E:\norm-exp\tests\test_norm_modules.py)

Current result:

- `9 passed`

### Smoke tests

ViT 1-epoch smoke tests completed successfully on CIFAR10 for:

- `bCRMS`
- `BN`
- `GN` with `num_groups=6`
- `PLN` with `num_per_group=8`

MLP/KAN forward smoke tests also passed for representative `BN/GN/LN/PLN/bCRMS` configurations.

## Known Constraint

`InstanceNorm` is intentionally rejected for `dim=2` pure `(N, C)` inputs.
This is not a bug: instance statistics require a spatial axis.

## Documentation

README updates were appended to:

- [`extension/README.md`](E:\norm-exp\extension\README.md)
- [`MLP/README.md`](E:\norm-exp\MLP\README.md)
- [`ViT/README.md`](E:\norm-exp\ViT\README.md)
- [`KAN/README.md`](E:\norm-exp\KAN\README.md)
