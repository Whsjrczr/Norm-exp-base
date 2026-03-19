# MLP

本目录用于图像分类实验，统一入口为 `cifar10.py`。虽然文件名是 `cifar10`，但代码实际支持 `mnist`、`fashion-mnist`、`cifar10`、`ImageNet` 和通用 `folder` 数据集。`cifar10_multistage.py` 保留为兼容入口，内部直接转发到 `cifar10.py`。

## 入口文件

- `cifar10.py`：单阶段与多阶段训练统一入口
- `cifar10_multistage.py`：兼容入口
- `model/selection_tool.py`：模型选择逻辑
- `model/MLP.py`、`model/resnet.py`：模型定义

## 快速开始

```bash
python MLP/cifar10.py \
  -a MLP \
  --dataset cifar10 \
  --dataset-root ./dataset \
  --batch-size 64,1000 \
  --depth 4 \
  --width 100 \
  --epochs 500 \
  --optimizer adam \
  --lr 1e-3 \
  --norm No \
  --activation relu
```

多阶段优化示例：

```bash
python MLP/cifar10.py \
  -a ConvBNPre \
  --dataset cifar10 \
  --optimizer-stages "adam:lr=1e-3,epochs=50; sgd:lr=1e-2,weight_decay=1e-4,epochs=150"
```

## 实验参数

### 模型参数

- `-a, --arch`：`MLP`、`ResCenDropScalingMLP`、`CenDropScalingMLP`、`CenDropScalingPreNormMLP`、`LinearModel`、`Linear`、`resnet18`、`resnet34`、`resnet50`、`MLPReLU`、`PreNormMLP`、`ConvBN`、`ConvBNPre`、`ConvBNRes`、`ConvBNResPre`、`ConvLN`、`ConvLNPre`、`ConvLNRes`、`ConvLNResPre`
- `--width`：隐藏层或通道宽度，默认 `100`
- `--depth`：网络深度，默认 `4`
- `--dropout`：dropout 概率，默认 `0`

### 数据集参数

- `--dataset`：`mnist`、`fashion-mnist`、`cifar10`、`ImageNet`、`folder`
- `--dataset-root`：数据根目录
- `--dataset-cfg`：数据集附加配置
- `-b, --batch-size`：batch size，支持 `train,val`
- `-j, --workers`：DataLoader worker 数
- `--im-size`：图像尺寸，例如 `"32,32"`
- `--dataset-classes`：类别数，默认自动推断

### 训练参数

- `-n, --epochs`：训练 epoch 数，默认 `500`
- `--start-epoch`：恢复训练时的起始 epoch
- `-o, --output`：输出目录，默认 `./results`
- `-t, --test`：仅验证
- `--seed`：随机种子
- `--offline`：WandB 离线模式

### 优化器与调度器

- `--optimizer`：`sgd`、`adam`、`adamw`、`adamax`、`RMSprop`、`lbfgs`
- `--optimizer-config`：优化器额外参数
- `--optimizer-stages`：多阶段训练配置
- `--weight-decay`：权重衰减
- `--lr-method`：`fix`、`step`、`steps`、`ploy`、`auto`、`exp`、`user`、`cos`、`1cycle`
- `--lr`
- `--lr-step`
- `--lr-gamma`
- `--lr-steps`

多阶段常用字段：

- `optimizer`
- `lr`
- `weight_decay`
- `optimizer_config`
- `epochs`
- `end_epoch`
- `lr_method`
- `lr_step`
- `lr_gamma`

说明：

- 如果 `--optimizer-stages` 中每个 stage 都显式提供了 `epochs` 或 `end_epoch`，总训练 epoch 会自动由 stage 配置推导，`--epochs` 不再额外截断或补长最后一个 stage。

### 归一化与激活

- `--norm`：`BN`、`GN`、`LN`、`IN`、`LNc`、`LNs`、`RMS`、`CDS`、`BNc`、`BNs`、`bCDS`、`bClCDS`、`bCLN`、`bCRMS`、`GNc`、`GNs`、`PLN`、`PLS`、`No`、`no`
- `--norm-cfg`：归一化层附加参数
- `--activation`：`relu`、`sigmoid`、`tanh`、`gn`、`pgn`、`sinarctan`、`no`
- `--activation-cfg`：激活附加参数

### 日志、恢复与可视化

- `--visualize`：启用 WandB
- `--wandb-project`：WandB project 名
- `--taiyi`：启用 Taiyi 监控
- `--vis`：启用 Visdom
- `--vis-port`：Visdom 端口
- `--vis-env`：Visdom 环境名
- `--resume`：从 checkpoint 恢复
- `--load`：加载模型参数
- `--load-no-strict`：关闭严格匹配
- `--log-suffix`：输出目录后缀

## 参数格式

### 1. 列表

```bash
--batch-size "64,1000"
--lr-steps "50,100,150"
--im-size "32,32"
```

说明：

- 单个值如 `--batch-size 128` 会自动扩展成训练和验证相同 batch size
- 两个值时分别表示 `train_batch_size,val_batch_size`

### 2. 字典

```bash
--dataset-cfg "grey=True"
--norm-cfg "num_groups=8,dim=4"
--activation-cfg "inplace=True"
--optimizer-config "momentum=0.9"
```

### 3. 多阶段优化

```bash
--optimizer-stages "adam:lr=1e-3,epochs=50; sgd:lr=1e-2,weight_decay=1e-4,epochs=150"
```

或：

```bash
--optimizer-stages "[{'optimizer':'adam','lr':1e-3,'epochs':50},{'optimizer':'sgd','lr':1e-2,'epochs':150}]"
```

## 数据格式

### 1. 内置数据集

当 `--dataset` 为 `mnist`、`fashion-mnist`、`cifar10` 时，代码会在 `dataset-root/<dataset_name>/` 下自动下载或读取数据。

### 2. ImageFolder 格式

当 `--dataset ImageNet` 或 `--dataset folder` 时，目录格式为：

```text
<dataset-root>/
  ImageNet/
    train/
      class_a/
      class_b/
    val/
      class_a/
      class_b/
```

或：

```text
<dataset-root>/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
```

其中：

- `ImageNet` 模式会自动拼接 `dataset-root/ImageNet`
- `folder` 模式直接把 `dataset-root` 视为数据集目录

### 3. 数据集配置选项

常用 `dataset-cfg`：

- `nogrey=True`：保留 RGB，不转灰度
- `random_label=True`：随机标签实验
- `loader=vit`：使用 ViT 风格增强与预处理
- `mean=...`、`std=...`：自定义归一化

默认情况下：

- `mnist` / `fashion-mnist`：单通道
- `cifar10`：默认会在归一化后附加灰度化，若需要 RGB，请设置 `--dataset-cfg "nogrey=True"`

## 输出格式

默认输出目录：

```text
results/
  <arch>_<datasetflag>_d<depth>_w<width>_<norm>_<act>_lr<lr>_bs<train_bs>_dropout<dropout>_wd<wd>_seed<seed>/
    <log_suffix>/
      log.txt
      checkpoint.pth
      best.pth
```

## 参考脚本

- `bash.sh`：单次实验示例
## Update

- 颜色开关语义已更新为“默认保留 RGB，只有 `grey=True` 时才灰度化”
- `nogrey` 仅保留兼容，不建议继续使用；`nogrey=True` 等价于 `grey=False`
- `--batch-size 128` 会自动扩成 train/val 都使用 `128`
