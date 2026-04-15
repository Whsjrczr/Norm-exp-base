# ViT

`ViT/vit.py` 现在已经按项目公共训练框架重构，训练、数据集、优化器、调度器、checkpoint、WandB、Taiyi、Visdom 都统一走 `extension`。

## 目录

- `vit.py`: ViT 训练入口
- `model_vit/select_vit.py`: 模型选择
- `model_vit/vision_transformer.py`: ViT 结构定义

## 支持模型

- `vit_tiny`
- `vit_small`
- `vit_base`

## 常用示例

```bash
python ViT/vit.py \
  --arch vit_small \
  --dataset folder \
  --dataset-root ./dataset/ImageNet \
  --im-size 224,224 \
  --patch-size 16 \
  --batch-size 256 \
  --epochs 100 \
  --optimizer adamw \
  --lr 1e-4 \
  --weight-decay 0.1 \
  --norm LN \
  --activation relu

# pq activation example
python ViT/vit.py \
  --arch vit_small \
  --dataset folder \
  --dataset-root ./dataset/ImageNet \
  --im-size 224,224 \
  --patch-size 16 \
  --norm PQN \
  --norm-cfg "num_per_group=8,p=4,q=2,dim=3,layout=last" \
  --activation pqact \
  --activation-cfg "p=4,q=2"
```

如果数据目录本身就是：

```text
./dataset/ImageNet/
  train/
  val/
```

也可以直接写：

```bash
python ViT/vit.py \
  --arch vit_small \
  --data_path ./dataset/ImageNet \
  --num_classes 1000
```

## 数据集相关参数

- `--dataset`: 默认 `ImageNet`，也支持 `folder`
- `--dataset-root`: 数据集根目录
- `--data_path`: `--dataset-root` 的别名；如果设置且 `dataset=ImageNet`，脚本会自动切到 `folder`
- `--dataset-cfg`: 数据集附加配置
- `--im-size`: 输入尺寸，ViT 默认 `224,224`
- `--batch-size`, `--batch_size`: 完全等价，支持单值或双值
- `--val-batch-size`: 单独覆盖验证 batch size
- `--disable-train-shuffle`: 关闭训练集 shuffle
- `--workers`: DataLoader worker 数

## batch size 规则

```bash
--batch-size 256
--batch_size 256
--batch-size 256,128
--val-batch-size 512
```

- 单个 `--batch-size 256` 会自动扩成 train/val 都使用 `256`
- `--batch_size` 只是下划线写法，和 `--batch-size` 没有功能差异
- 两个值表示 `train_batch_size,val_batch_size`
- `--val-batch-size` 会覆盖第二个值
- 最终 batch size 的补齐逻辑统一由 `extension/dataset.py` 处理

## 颜色与预处理

当前默认行为是保留 RGB，不再默认灰度化。

如果需要灰度输入：

```bash
--dataset-cfg "grey=True"
```

ViT 风格 `loader=vit` 下：

- train: `RandomResizedCrop -> RandomHorizontalFlip -> ToTensor -> Normalize`
- val: `Resize -> CenterCrop -> ToTensor -> Normalize`

默认归一化参数：

- `mean=[0.485, 0.456, 0.406]`
- `std=[0.229, 0.224, 0.225]`

灰度模式下会先做 `Grayscale(num_output_channels=1)`。

## 训练相关参数

- `--epochs`
- `--start-epoch`
- `--output`
- `--test`
- `--seed`
- `--offline`
- `--resume`
- `--load`
- `--load-no-strict`

## 优化器与调度器

- `--optimizer`
- `--optimizer-config`
- `--optimizer-stages`
- `--weight-decay`
- `--lr-method`
- `--lr`
- `--lr-step`
- `--lr-gamma`
- `--lr-steps`

默认设置：

- optimizer: `adamw`
- lr: `1e-4`
- weight decay: `0.1`
- lr method: `cos`

## 可视化与监控

- `--visualize`: 启用 WandB
- `--wandb-project`: WandB project 名称
- `--taiyi`: 启用 Taiyi monitor
- `--vis`: 启用 Visdom

## 数据目录格式

`--dataset folder` 时：

```text
<dataset-root>/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
```

`--dataset ImageNet` 时，实际读取：

```text
<dataset-root>/ImageNet/train
<dataset-root>/ImageNet/val
```

## 输出目录

```text
results/
  ViT_<arch>_<datasetflag>_img<im_size>_patch<patch_size>_<norm>_<act>_lr<lr>_bs<train_bs>_dropout<dropout>_droppath<drop_path_rate>_wd<wd>_seed<seed>/
    <log_suffix>/
      log.txt
      checkpoint.pth
      best.pth
```

## 2026-04 Normalization Update

ViT now binds normalization through:

```python
norm_layer = ext.make_norm_factory(dim=3, layout="last")
```

This matches ViT token tensors shaped `(B, N, C)`.

### What this enables

The following norm families can now be used directly in ViT without custom patching:

- `LN`, `LNc`, `LNs`, `RMS`
- `BN`, `BNc`, `BNs`
- `GN`, `GNc`, `GNs`
- `IN`
- `PLN`, `PLS`, `PQN`
- `bCLN`, `bCRMS`

For `BN/GN/IN`, the factory automatically adapts token-last `(B, N, C)` to the underlying channel-first implementation and then restores the original layout.

### Recommended CLI examples

```bash
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 16 --norm BN
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 16 --norm GN --norm-cfg "num_groups=6"
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 16 --norm PLN --norm-cfg "num_per_group=8"
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 16 --norm PQN --norm-cfg "num_per_group=8,p=4,q=2,dim=3,layout=last"
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 16 --norm bCRMS
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 16 --activation pqact --activation-cfg "p=4,q=2"
```

### Sequence BatchNorm on ViT

ViT now supports sequence-axis BatchNorm families directly through `select_vit.py`.

For ViT token tensors `(B, N, C)`:

- `SeqBN`, `SeqBNc`, `SeqBNs` automatically bind `num_features` to `N = num_patches + 1`
- `DSeqBN`, `DSeqBNc`, `DSeqBNs` work without a fixed token count

Minimal examples:

```bash
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 4 --norm SeqBN
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 4 --norm SeqBNc
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 4 --norm SeqBNs
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 4 --norm DSeqBN
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 4 --norm DSeqBLS
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 4 --norm DSeqBCLN
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 4 --norm DSeqBCRMS
python ViT/vit.py --arch vit_small --dataset cifar10 --im-size 32,32 --patch-size 4 --norm DSeqBCDS
```

Notes:

- You do not need to pass `num_features` manually for ViT.
- You do not need `--norm-cfg "dim=3,layout=last"` for the ViT path.
- Fixed-length `SeqBN*` is the better match when image size and patch size are fixed.
- Dynamic `DSeqBN*` is safer if token length may vary across calls.
- `DSeqBLS` means `DSeqBNc + LayerScaling`.
- `DSeqBCLN` means `DSeqBNc + LayerNorm`.
- `DSeqBCRMS` means `DSeqBNc + RMSNorm`.
- `DSeqBCDS` means `DSeqBNc + Dropout + LayerScaling`.

Batch script:

- [`run_vit_sequence_bn_batch.sh`](/e:/norm-exp/ViT/run_vit_sequence_bn_batch.sh)
