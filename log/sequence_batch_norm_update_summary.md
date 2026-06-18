# Sequence BatchNorm Update Summary

Date: 2026-04-12

## Scope

This update adds BatchNorm variants for Transformer sequence inputs that normalize along the sequence axis instead of the channel axis.

The implementation is split into two modules to match the repository's existing normalization style:

- centering only
- scaling only

It also now includes a combined module:

- centering -> scaling

## What Changed

### 1. Added sequence-axis BatchNorm modules

Created:

- [`extension/my_modules/norm/seq_bn.py`](/e:/norm-exp/extension/my_modules/norm/seq_bn.py)

Updated:

- [`extension/my_modules/norm/bn1d_modules.py`](/e:/norm-exp/extension/my_modules/norm/bn1d_modules.py)

Added:

- `SequenceBatchNorm1d`
- `SequenceBatchNorm1dCentering`
- `SequenceBatchNorm1dScaling`
- `DynamicSequenceBatchNorm1d`
- `DynamicSequenceBatchNorm1dCentering`
- `DynamicSequenceBatchNorm1dScaling`

Behavior:

- supports sequence inputs with `dim >= 3`
- supports both `(B, N, ..., C)` and `(B, C, ..., N)`
- computes statistics per sequence position
- for `(B, N, ..., C)`, each of the `N` sequence positions is normalized over all non-sequence axes
- for `(B, C, ..., N)`, each of the `N` sequence positions is normalized over all non-sequence axes

`SequenceBatchNorm1d` is implemented as:

```python
SequenceBatchNorm1dCentering -> SequenceBatchNorm1dScaling
```

This is different from the existing `BatchNorm1dCentering` / `BatchNorm1dScaling`, which normalize along the feature/channel axis.

Dynamic sequence BN variants differ from fixed-length `SeqBN*` in one key way:

- they do not require `num_features=seq_len`
- they do not keep per-position running statistics
- their optional affine parameters are shared scalars, not per-position vectors

### 2. Registered factory names in `Norm(...)`

Updated:

- [`extension/normalization.py`](/e:/norm-exp/extension/normalization.py)

Added factory names:

- `SeqBN`
- `SeqBNc`
- `SeqBNs`
- `DSeqBN`
- `DSeqBNc`
- `DSeqBNs`

Behavior:

- intended for sequence tensors with `dim >= 3`
- supports `layout="last"` for `(B, N, ..., C)`
- supports `layout="first"` for `(B, C, ..., N)`
- `num_features` means sequence length, not hidden size

Dynamic behavior:

- `DSeqBN*` accepts variable-length sequence inputs
- `DSeqBN*` can be constructed through `Norm(...)` without passing `num_features`

Example:

```python
ext.Norm(seq_len, dim=3, layout="last")
```

with:

```python
normalization._config.norm = "SeqBNc"
normalization._config.norm = "SeqBNs"
```

### 3. Exported the new modules

Updated:

- [`extension/my_modules/norm/__init__.py`](/e:/norm-exp/extension/my_modules/norm/__init__.py)
- [`extension/my_modules/__init__.py`](/e:/norm-exp/extension/my_modules/__init__.py)

Exports added:

- `SequenceBatchNorm1d`
- `SequenceBatchNorm1dCentering`
- `SequenceBatchNorm1dScaling`
- `DynamicSequenceBatchNorm1d`
- `DynamicSequenceBatchNorm1dCentering`
- `DynamicSequenceBatchNorm1dScaling`

### 4. Added tests and fixed running-stat behavior

Updated:

- [`tests/test_norm_modules.py`](/e:/norm-exp/tests/test_norm_modules.py)

Added checks for:

- shape support on both `(B, N, C)` and `(B, C, N)`
- shape support on higher-dimensional sequence tensors
- centering correctness on the sequence axis
- scaling correctness on the sequence axis
- combined `SequenceBatchNorm1d` equivalence to `centering -> scaling`
- dynamic sequence BN reuse across different sequence lengths
- factory path through `Norm(...)` using `SeqBN` / `SeqBNc` / `SeqBNs`
- factory path through `Norm(...)` using `DSeqBN` / `DSeqBNc` / `DSeqBNs`

Bug fixed during validation:

- `SequenceBatchNorm1dCentering` and `SequenceBatchNorm1dScaling` incorrectly tried to update running statistics when `track_running_stats=False`
- this is now guarded correctly

### 5. Updated README documentation

Updated:

- [`extension/README.md`](/e:/norm-exp/extension/README.md)

Added documentation for:

- `SeqBN`
- `SeqBNc`
- `SeqBNs`
- `DSeqBN`
- `DSeqBNc`
- `DSeqBNs`
- sequence-axis semantics
- the meaning of `num_features` for sequence BatchNorm
- higher-dimensional layout conventions
- dynamic variable-length sequence behavior
- CLI examples for ViT-style inputs

## Validation

Environment used:

- local project conda environment at `E:\norm-exp\.conda`

Targeted command:

```powershell
& E:\norm-exp\.conda\python.exe -m pytest tests\test_norm_modules.py -q
```

Result:

- `22 passed`

Observed warning:

- pytest cache path creation produced a Windows permission warning under `.pytest_cache`
- this did not affect test execution or outcomes

## Usage Notes

- Use `SeqBN` / `SeqBNc` / `SeqBNs` only for sequence tensors with `dim >= 3`.
- Pass sequence length as `num_features`.
- On `(B, N, ..., C)`, use `layout="last"`.
- On `(B, C, ..., N)`, use `layout="first"`.
- Use `DSeqBN` / `DSeqBNc` / `DSeqBNs` when sequence length varies across calls and you do not want per-position running stats.
- If you want channel-axis BatchNorm on sequence tensors, continue using the existing `BN` / `BNc` / `BNs`.

## Examples

### Direct module usage on `(B, N, C)`

```python
import torch
from extension.my_modules.norm.bn1d_modules import (
    SequenceBatchNorm1d,
    SequenceBatchNorm1dCentering,
    SequenceBatchNorm1dScaling,
)

x = torch.randn(2, 16, 64)  # batch=2, seq_len=16, hidden=64

centering = SequenceBatchNorm1dCentering(16, affine=False)
scaling = SequenceBatchNorm1dScaling(16, affine=True)

y = scaling(centering(x))
print(y.shape)  # (2, 16, 64)
```

Meaning:

- `16` is the sequence length
- each sequence position is normalized across batch and hidden dimensions

### Combined module usage

```python
import torch
from extension.my_modules.norm.bn1d_modules import SequenceBatchNorm1d

x = torch.randn(2, 16, 4, 64)  # (B, N, H, C)
norm = SequenceBatchNorm1d(16, layout="last")
y = norm(x)
print(y.shape)  # (2, 16, 4, 64)
```

### Factory usage through `ext.Norm(...)`

```python
import extension as ext
import extension.normalization as normalization

normalization._config.norm = "SeqBNc"
normalization._config.norm_cfg = {}
seq_center = ext.Norm(16, dim=3, layout="last")

normalization._config.norm = "SeqBNs"
normalization._config.norm_cfg = {}
seq_scale = ext.Norm(16, dim=3, layout="last")

normalization._config.norm = "SeqBN"
normalization._config.norm_cfg = {}
seq_norm = ext.Norm(16, dim=3, layout="last")

normalization._config.norm = "DSeqBN"
normalization._config.norm_cfg = {}
dyn_seq_norm = ext.Norm(dim=3, layout="last")
```

Equivalent CLI-style config:

```bash
--norm SeqBN
--norm-cfg "dim=3,layout=last"

--norm SeqBNc
--norm-cfg "dim=3,layout=last"

--norm SeqBNs
--norm-cfg "dim=3,layout=last"

--norm DSeqBN
--norm-cfg "dim=3,layout=last"
```
