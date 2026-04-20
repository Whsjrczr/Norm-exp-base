# KAN

`KAN/KAN.py` 是当前 `KAN` 实验的统一训练入口。整理后已经和 `PDE/pde.py`、`MLP/cifar10.py`、`ViT/vit.py` 保持一致的整体风格：`cfg` 驱动、`Trainer` 类结构、统一日志/结果目录、checkpoint、resume、多阶段 optimizer 调度。

## 目录结构

- `KAN.py`: KAN 回归任务训练入口
- `kan_dataset.py`: 合成回归数据构造器 `KANDatasetBuilder`
- `extension/model/kan/__init__.py`: KAN 模型选择入口，统一为 `get_model(cfg)`
- `extension/model/kan/KAN.py`: KAN 网络封装 `KANNetwork` / `KAN_norm`
- `extension/model/kan/KAN_layer.py`: KAN 核心层 `KANLinear`
- `extension/model/kan/MLP.py`: 对照用 MLP 实现

## 当前整理结果

### 1. `KAN_layer` 合并

原先 `KAN_layer copy.py` 中存在大量分叉实现，当前已经收束为一个核心层：

- `KANLinear`: 标准 KAN 层
- `KANLinearReLU`: 在输出端增加 ReLU 的兼容变体
- `KANLinearNoRes`: 关闭 base branch 的兼容变体
- `KANLinearWN`: 兼容别名，统一走 `KANLinear`

核心保留能力：

- spline grid 初始化
- `update_grid`
- `regularization_loss`
- 权重归一化接口 `normalize_weights`

### 2. 统一模型入口

`extension/model/kan/__init__.py` 已改成和其他目录一致的风格：

```python
def get_model(cfg):
    ...
```

当前支持：

- `KAN`
- `MLP`

并统一从 `cfg` 推导：

- `layers_hidden`
- `input_dim`
- `output_dim`
- `depth`
- `width`

### 3. `KAN.py` 已补齐的能力

对照 `PDE.py`、`cifar10.py`、`vit.py`，当前 `KAN.py` 已补齐：

- `Trainer` 类结构
- `add_arguments`
- `train`
- `validate`
- `test`
- checkpoint / resume / load
- 多阶段 optimizer 调度
- wandb / visdom 接口
- 统一结果目录
- 自动保存预测图

### 4. `kan_dataset.py` 已整理

`kan_dataset.py` 现在模仿 `pde_dataset.py` 的组织方式，主要提供：

- `KANDatasetBuilder`
- `equation`
- `prepare_data`（兼容旧调用）

数据构造流程集中在 builder 中完成：

- 采样输入
- 构造真值
- 加噪
- 划分 train / val / test
- 生成 DataLoader

## `KAN.py` 对比说明

### 原来缺少的功能

- 统一的 `cfg` 配置入口
- 和其他任务一致的 `Trainer` 结构
- `validate/test` 分离
- checkpoint / resume
- 多阶段训练支持
- 标准化的结果保存和日志输出
- `get_model(cfg)` 风格模型选择

### 原来多余或应移除的部分

- `__main__` 中硬编码实验循环
- 模型构造、数据处理、训练逻辑耦合在一个脚本里
- 通配符导入
- 不稳定的包路径写法
- 重复的 KAN layer 变体文件

## 运行示例

### 1. 训练 KAN

```bash
python KAN/KAN.py \
  --arch KAN \
  --layers-hidden 3,32,32,32,1 \
  --batch-size 64,256 \
  --epochs 500 \
  --optimizer adam \
  --lr 1e-3 \
  --norm No \
  --activation relu \
  --error 0.05
```

### 2. 训练对照 MLP

```bash
python KAN/KAN.py \
  --arch MLP \
  --layers-hidden 3,32,32,32,1 \
  --batch-size 64,256 \
  --epochs 500 \
  --optimizer adam \
  --lr 1e-3 \
  --norm No \
  --activation relu
```

### 3. 开启 grid 更新

```bash
python KAN/KAN.py \
  --arch KAN \
  --layers-hidden 3,32,32,32,1 \
  --update-grid
```

### 4. 多阶段 optimizer

```bash
python KAN/KAN.py \
  --arch KAN \
  --layers-hidden 3,32,32,32,1 \
  --optimizer-stages "adam:lr=1e-3,epochs=100; adamw:lr=3e-4,weight_decay=1e-4,epochs=200"
```

## 主要参数

### 模型参数

- `--arch`: `KAN` 或 `MLP`
- `--layers-hidden`: 完整网络宽度列表，如 `3,32,32,1`
- `--width`: 当未显式给 `layers_hidden` 时使用
- `--depth`: 当未显式给 `layers_hidden` 时使用
- `--grid-size`
- `--spline-order`
- `--scale-noise`
- `--scale-base`
- `--scale-spline`
- `--grid-eps`
- `--grid-range`
- `--update-grid`
- `--weight-norm`
- `--kan-regularization`
- `--kan-init`: `origin` / `xavier` / `kaiming`
- `--residual-activation`: 单独控制 KAN residual/base branch 上的激活函数，`same` 表示跟随 `--activation`
- `--disable-residual-branch`: 关闭 KAN residual/base branch，仅保留 spline branch

### 数据与任务参数

- `--num-samples`
- `--train-ratio`
- `--val-ratio`
- `--function`
- `--error`
- `--curve-points`
- `--curve-min`
- `--curve-max`

### 训练参数

- `--batch-size`
- `--epochs`
- `--start-epoch`
- `--optimizer`
- `--optimizer-config`
- `--optimizer-stages`
- `--weight-decay`
- `--lr-method`
- `--lr`
- `--lr-step`
- `--lr-gamma`
- `--lr-steps`
- `--seed`
- `--output`
- `--test`
- `--resume`
- `--load`

### 归一化与激活

- `--norm`
- `--norm-cfg`
- `--activation`
  ????? `pqact`
- `--activation-cfg`
  `pqact` ?? `p=4,q=2`

### 可视化

- `--visualize`
- `--wandb-project`
- `--offline`
- `--vis`
- `--vis-port`
- `--vis-env`

## 输出目录

训练结果默认写入：

```text
results/
  KAN_<arch>_<function>_h<layers>_<norm>_<act>_lr<lr>_bs<train_bs>_wd<wd>_noise<error>_seed<seed>/
    <log_suffix>/
      log.txt
      checkpoint.pth
      best.pth
      prediction_curve.png
```

## 已验证内容

本次整理后已完成：

- `py_compile` 语法检查
- `python KAN/KAN.py --epochs 1 --num-samples 32 ...` smoke test

当前脚本可以完成：

- 训练
- 验证
- 测试
- 保存 checkpoint
- 保存预测图

## 说明

- 仓库根目录中的包名实际为 `KAN`，因此 README 和命令统一使用大写目录名。
- 当前任务是合成回归任务，不直接复用 `extension.dataset` 中的图像数据加载流程。
- 如果后续需要把 `KAN` 接入 `PDE` 或其他任务，只需要通过 `extension.model.get_model(cfg)` 接入即可。

## 2026-04 Normalization Update

KAN and the comparison MLP path now bind normalization as 2D feature normalization:

```python
norm_2d = ext.make_norm_factory(dim=2)
```

This means hidden activations are treated as `(N, C)` tensors instead of falling back to the old implicit `dim=4` behavior.

### Current effect

The following norm families can now be used on KAN/MLP fully-connected paths:

- `BN`, `BNc`, `BNs`
- `GN`, `GNc`, `GNs`
- `LN`, `LNc`, `LNs`, `RMS`
- `PLN`, `PLS`, `PQN`
- `bCLN`, `bCRMS`

`InstanceNorm` is still not supported for pure 2D `(N, C)` activations.

### Recommended CLI examples

```bash
python KAN/KAN.py --arch KAN --layers-hidden 3,32,32,1 --norm BN --norm-cfg "dim=2"
python KAN/KAN.py --arch MLP --layers-hidden 3,32,32,1 --norm GN --norm-cfg "dim=2,num_groups=8"
python KAN/KAN.py --arch KAN --layers-hidden 3,32,32,1 --norm PLN --norm-cfg "dim=2,num_per_group=8"
python KAN/KAN.py --arch KAN --layers-hidden 3,32,32,1 --norm PQN --norm-cfg "dim=2,p=4,q=2"
python KAN/KAN.py --arch KAN --layers-hidden 3,32,32,1 --activation pqact --activation-cfg "p=4,q=2"
```
