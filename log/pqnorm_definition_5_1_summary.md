# PQNorm Definition 5.1 Summary

Date: 2026-04-01

## Source

Reference paper:

- `C:/Users/DELL/Downloads/ICML_2026_PLN_arxiv.pdf`

Used definition:

- Definition 5.1 `((p, q)-normalization)`

The extracted definition states that for a single sample
`h = [h1, h2, ..., hd]^T in R^d`,

```text
h~_i = h_i^(p/q) / |h^p|^(1/q)
```

with the paper's induced constraint equivalent to:

```text
mean(|h~|^q) = 1
```

In implementation, sign is preserved via:

```text
sign(h_i) * |h_i|^(p/q)
```

which is the numerically stable odd-extension consistent with the paper's later discussion of odd `p/q`.

## What Changed

### 1. Added `PQNorm` module

Created:

- [`extension/my_modules/norm/pq_norm.py`](/e:/norm-exp/extension/my_modules/norm/pq_norm.py)

Behavior:

- supports `2D`, `3D`, `4D` inputs
- supports both channel-first and channel-last layouts
- supports learnable affine `weight` and `bias`
- uses `eps` for denominator stability

Core implementation:

```python
ratio = p / q
numerator = sign(x) * abs(x) ** ratio
denominator = mean(abs(x) ** p, dim=-1, keepdim=True)
y = numerator / (denominator + eps) ** (1 / q)
```

### 2. Registered `PQN` in normalization factory

Updated:

- [`extension/normalization.py`](/e:/norm-exp/extension/normalization.py)

Changes:

- imported `PQNorm`
- added `_PQNorm(...)`
- registered `"PQN"` in `_config.norm_methods`
- added config flag rendering for `p` and `q`

You can now use:

```bash
--norm PQN --norm-cfg "p=4,q=2,dim=2"
```

### 3. Exported module

Updated:

- [`extension/my_modules/__init__.py`](/e:/norm-exp/extension/my_modules/__init__.py)

Change:

- exported `PQNorm`

### 4. Added tests

Updated:

- [`tests/test_norm_modules.py`](/e:/norm-exp/tests/test_norm_modules.py)

Added coverage for:

- common input layouts
- Definition 5.1 constraint `mean(|y|^q) ~= 1`
- factory integration through `normalization.Norm(...)`

## Validation

Targeted test command:

```bash
pytest -q tests/test_norm_modules.py -k "pq_norm or norm_factory_supports_mlp_2d_for_bn_and_gn or norm_factory_supports_vit_token_last_for_bn_in_gn"
```

Result:

- `4 passed, 7 deselected`

Observed environment warning:

- pytest cache path creation produced a Windows permission warning
- this did not affect the correctness of the implementation or the test result

## Notes

- The implementation is not grouped like `PLN`; it applies `(p, q)` normalization across the feature axis of each sample.
- Current implementation assumes the normalization axis is the feature/channel axis, matching the rest of this repository's normalization interface.

## Follow-up: centered PQN

A later update added optional `centering` to `PQN`.

Behavior:

- `centering=False`: keep the original Definition 5.1 style implementation in code
- `centering=True`: subtract the per-group or full-feature mean before `(p, q)` normalization

This makes centered `PQN` align with `PLN` semantics when `p=q=2`.
