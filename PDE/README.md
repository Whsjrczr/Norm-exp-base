# PDE

本目录用于基于 `DeepXDE + PyTorch` 训练 PDE/PINN 模型，统一入口为 `pde.py`。`pde_multistage.py` 保留为兼容入口，内部直接转发到 `pde.py`。

## 入口文件

- `pde.py`：单阶段、多阶段、可视化统一入口
- `pde_multistage.py`：兼容入口
- `pde_dataset.py`：PDE 定义、几何区域、边界条件、采样点数与损失权重

## 快速开始

```bash
python PDE/pde.py \
  -a MLP \
  --width 32 \
  --depth 4 \
  --pde_type allen_cahn \
  --lr 1e-3 \
  --epochs 6000 \
  --norm LN \
  --activation relu \
  --optimizer adam \
  --metrics "l2 relative error,MSE"
```

多阶段优化示例：

```bash
python PDE/pde.py \
  -a MLP \
  --width 64 \
  --depth 4 \
  --pde_type helmholtz \
  --optimizer-stages "adam:lr=1e-3,iterations=3000; lbfgs:optimizer_config=max_iter=5000"
```

启用 Visdom 与 WandB：

```bash
python PDE/pde.py \
  -a MLP \
  --pde_type poisson \
  --visualize \
  --vis \
  --subject_name pde-exp
```

## 实验参数

### 模型参数

- `-a, --arch`：`MLP`、`PreNormMLP`、`CenDropScalingMLP`、`CenDropScalingPreNormMLP`、`ResCenDropScalingMLP`
- `--width`：隐藏层宽度，默认 `50`
- `--depth`：隐藏层深度，默认 `3`
- `--dropout`：dropout 概率，默认 `0`

### PDE 参数

- `--pde_type`：`poisson`、`poisson_new`、`helmholtz`、`helmholtz_new`、`helmholtz_learnable_2`、`helmholtz2d`、`allen_cahn`、`allen_cahn_new`、`wave`、`klein_gordon`、`convdiff`、`cavity`
- `--loss-weights`：损失权重列表，默认 `"1.0,1.0"`
- `--metrics`：评估指标列表，默认 `"l2 relative error"`
- `--batch_size`：训练 batch size，默认 `128`
- `--display_every`：每多少次迭代打印和记录一次，默认 `1000`
- `--float64`：启用 `float64`

### 训练参数

- `-n, --epochs`：训练总迭代数，默认 `10000`
- `--seed`：随机种子
- `-o, --output`：输出目录，默认 `./results`
- `-t, --test`：仅验证
- `--offline`：WandB 离线模式
- `--no_save_best`：不保存最佳模型
- `--subject_name`：WandB project 名

### 优化器参数

- `-oo, --optimizer`：`sgd`、`adam`、`adamw`、`adamax`、`RMSprop`、`lbfgs`
- `-oc, --optimizer-config`：优化器附加参数
- `-os, --optimizer-stages`：多阶段优化配置
- `-wd, --weight-decay`：权重衰减

多阶段常用字段：

- `optimizer`
- `lr`
- `weight_decay`
- `optimizer_config`
- `iterations`
- `batch_size`
- `loss_weights`
- `metrics`

说明：

- 如果 `--optimizer-stages` 中每个 stage 都显式提供了 `iterations`、`iters` 或兼容写法 `epochs`，总训练迭代数会自动由 stage 配置推导，`--epochs` 仅在 stage 未写完整时作为回退值。

### 学习率调度参数

- `--lr-method`：`fix`、`step`、`steps`、`ploy`、`auto`、`exp`、`user`、`cos`、`1cycle`
- `--lr`
- `--lr-step`
- `--lr-gamma`
- `--lr-steps`

### 归一化与激活

- `--norm`?`BN`?`GN`?`LN`?`IN`?`LNc`?`LNs`?`RMS`?`CDS`?`BNc`?`BNs`?`bCDS`?`bClCDS`?`bCLN`?`bCRMS`?`GNc`?`GNs`?`PLN`?`PLS`?`PQN`?`No`?`no`
- `--norm-cfg`：归一化配置
- `--activation`：`relu`、`sigmoid`、`tanh`、`gn`、`pgn`、`sinarctan`、`no`
- `--activation-cfg`：激活附加配置

### 日志与可视化

- `--visualize`：启用 WandB
- `--taiyi`：启用 Taiyi 监控
- `--vis`：启用 Visdom
- `--vis-port`：Visdom 端口
- `--vis-env`：Visdom 环境名
- `--resume`：从 checkpoint 恢复训练
- `--load`：仅加载模型参数
- `--load-no-strict`：关闭严格匹配
- `--log-suffix`：输出子目录后缀

当前 `--vis` 会记录：

- `train total loss`
- `test total loss`
- `train loss i`
- `test loss i`
- `metric <name>`
- `val error`

对于 `poisson`、`poisson_new`、`helmholtz`、`helmholtz_new`、`helmholtz_learnable_2`、`allen_cahn`、`allen_cahn_new`，训练结束后还会在结果目录额外保存 `solution_curve.png`。

## 参数格式

### 1. 列表参数

```bash
--metrics "l2 relative error,MSE"
--loss-weights "10.0,1.0"
--lr-steps "30,60,90"
```

### 2. 字典参数

```bash
--norm-cfg "num_per_group=8,dim=2"
--norm-cfg "p=4,q=2,dim=2"
--activation-cfg "inplace=True"
--activation pqact
--activation-cfg "p=4,q=2"
--optimizer-config "momentum=0.9"
```

### 3. 多阶段参数

Python/JSON 风格：

```bash
--optimizer-stages "[{'optimizer':'adam','lr':1e-3,'iterations':3000},{'optimizer':'lbfgs','optimizer_config':'max_iter=5000'}]"
```

简写风格：

```bash
--optimizer-stages "adam:lr=1e-3,iterations=3000; lbfgs:optimizer_config=max_iter=5000"
```

## 数据与任务格式

PDE 任务不依赖外部图像分类数据集，训练样本由 `pde_dataset.py` 在线构造。当前内置任务包括：

- 1D 定常问题：`poisson`、`helmholtz`、`allen_cahn`
- 2D 定常问题：`helmholtz2d`
- 时空问题：`wave`、`klein_gordon`、`convdiff`、`cavity`

验证阶段会对部分 1D PDE 使用解析解计算误差。

## 输出格式

默认输出目录：

```text
results/
  PDE_<pde_type>_<arch>_d<depth>_w<width>_<norm>_<act>_lr<lr>_epochs<epochs>_seed<seed>/
    <log_suffix>/
      log.txt
      checkpoint.pth
      best.pth
      model.pth
      solution_curve.png
```

其中 `solution_curve.png` 仅在存在解析参考解时生成。

## 参考脚本

- `bash.sh`：单次运行示例
- `generate.sh`：批量生成实验脚本
- `new.sh`、`new_generate.sh`：新版实验组合示例
