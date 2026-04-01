# Grouped PQNorm and PQ Activation Update Summary

Date: 2026-04-01

## Scope

This update extends the initial `PQN` implementation in three directions:

1. grouped `(p, q)` normalization, analogous to `LN -> PLN`
2. the paper's `(p, q)` activation
3. tests and README synchronization

Reference paper:

- `C:/Users/DELL/Downloads/ICML_2026_PLN_arxiv.pdf`

Relevant paper items:

- Definition 5.1: `(p, q)-normalization`
- Eqn. (26): `(p, q)-activation`

## What Changed

### 1. `PQNorm` now supports grouping

Updated:

- [`extension/my_modules/norm/pq_norm.py`](/e:/norm-exp/extension/my_modules/norm/pq_norm.py)

Change:

- added `num_per_group`

Behavior:

- `num_per_group=None`:
  `PQNorm` applies `(p, q)` normalization across the full feature axis
- `num_per_group=k`:
  the feature axis is partitioned into groups of size `k`, and `(p, q)` normalization is applied independently per group

This makes `PQNorm` the direct grouped analogue of the original full-feature version, in the same way that `PLN` is the grouped analogue of `LN`.

For grouped inputs, the implementation reshapes features as:

```python
(-1, num_features // num_per_group, num_per_group)
```

then applies:

```python
sign(x) * |x|^(p/q) / (mean(|x|^p) + eps)^(1/q)
```

inside each group.

### 1b. `PQNorm` now supports optional centering

Updated:

- [`extension/my_modules/norm/pq_norm.py`](/e:/norm-exp/extension/my_modules/norm/pq_norm.py)

Change:

- added `centering=False`

Behavior:

- `centering=False`: normalize directly on the original grouped features
- `centering=True`: subtract the grouped mean before `(p, q)` normalization

At `p=q=2`, grouped centered `PQN` matches `ParallelLN(..., centering=True)` when affine is disabled.

### 2. Normalization factory now supports grouped `PQN`

Updated:

- [`extension/normalization.py`](/e:/norm-exp/extension/normalization.py)

Changes:

- `_PQNorm(...)` now accepts `num_per_group`
- `PQN` config flag rendering now includes:
  - `num_per_group`
  - `p`
  - `q`

Example:

```bash
--norm PQN --norm-cfg "num_per_group=8,p=4,q=2,dim=2"
--norm PQN --norm-cfg "num_per_group=8,p=2,q=2,centering=True,dim=2"
```

### 3. Added the paper's `(p, q)` activation

Created:

- [`extension/my_modules/activation/pq_activation.py`](/e:/norm-exp/extension/my_modules/activation/pq_activation.py)

Module:

- `PQActivation`

Implemented formula:

```python
phi_{p,q}(x) = sign(x) * |x|^(p/q) / (|x|^p + 1)^(1/q)
```

This is the odd-extension form used in code, matching the paper's discussion of odd `p/q`.

### 4. Registered `pqact` in activation factory

Updated:

- [`extension/activation.py`](/e:/norm-exp/extension/activation.py)

Changes:

- added activation name: `pqact`
- added support for `activation_cfg` keys:
  - `p`
  - `q`
- added flag formatting for `pqact`

Example:

```bash
--activation pqact --activation-cfg "p=4,q=2"
```

### 5. Exported new modules

Updated:

- [`extension/my_modules/__init__.py`](/e:/norm-exp/extension/my_modules/__init__.py)

Exports added:

- `PQNorm`
- `PQActivation`

## Tests Added

Updated:

- [`tests/test_norm_modules.py`](/e:/norm-exp/tests/test_norm_modules.py)

Added checks for:

- grouped `PQNorm` shape handling on common layouts
- Definition 5.1 constraint:
  `mean(|y|^q) ~= 1`
- grouped `PQN` factory path through `normalization.Norm(...)`
- centered grouped `PQN` factory path through `normalization.Norm(...)`
- equivalence at `p=q=2`:
  grouped `PQNorm(..., affine=False)` matches
  `ParallelLN(..., centering=False, affine=False)`
- centered equivalence at `p=q=2`:
  grouped `PQNorm(..., centering=True, affine=False)` matches
  `ParallelLN(..., centering=True, affine=False)`
- activation equivalence at `p=q=2`:
  `PQActivation(2,2)` matches `SinArctan`

## Validation

Targeted command:

```bash
pytest -q tests/test_norm_modules.py -k "pq_norm or pq_activation or norm_factory_supports_mlp_2d_for_bn_and_gn or norm_factory_supports_vit_token_last_for_bn_in_gn"
```

Result:

- `6 passed, 7 deselected`

Meaning of `7 deselected`:

- the file contains additional tests
- they were not selected by this `-k` filter
- they were not executed, but also did not fail

Observed warning:

- pytest cache path creation produced a Windows permission warning
- this did not affect the implementation or test outcomes

## Documentation Updated

Updated README files:

- [`extension/README.md`](/e:/norm-exp/extension/README.md)
- [`MLP/README.md`](/e:/norm-exp/MLP/README.md)
- [`KAN/README.md`](/e:/norm-exp/KAN/README.md)
- [`PDE/README.md`](/e:/norm-exp/PDE/README.md)
- [`ViT/README.md`](/e:/norm-exp/ViT/README.md)

Added documentation for:

- `PQN`
- grouped `PQN` via `num_per_group`
- `pqact`
- `p/q` config examples
- ViT token-last usage for grouped `PQN`

## Practical Examples

Grouped `PQN` on MLP/KAN/PDE:

```bash
--norm PQN --norm-cfg "num_per_group=8,p=4,q=2,dim=2"
--norm PQN --norm-cfg "num_per_group=8,p=2,q=2,centering=True,dim=2"
```

Grouped `PQN` on ViT:

```bash
--norm PQN --norm-cfg "num_per_group=8,p=4,q=2,dim=3,layout=last"
```

`(p, q)` activation:

```bash
--activation pqact --activation-cfg "p=4,q=2"
```

## Notes

- When `p=q=2`, grouped `PQN` reduces to grouped L2 normalization, which matches `PLN` with `centering=False`.
- When `p=q=2` and `centering=True`, grouped `PQN` matches `PLN` with `centering=True`.
- When `p=q=2`, `pqact` reduces to:

```python
x / sqrt(x^2 + 1)
```

which is exactly the repository's `SinArctan`.
