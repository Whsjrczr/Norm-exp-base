# Unified Model Selector

`extension/model/` 是仓库里统一的模型选择入口。

现在 `MLP/`、`KAN/`、`ViT/` 的训练脚本都统一走 `extension.model`，不再各自维护单独的模型选择入口：

```python
import extension as ext

ext.model.add_model_arguments(parser, task="classification", default_family="mlp")
model = ext.model.get_model(cfg)
```

## 目录结构

```text
extension/model/
  __init__.py      # 统一入口，负责 family 解析和分发
  mlp/__init__.py  # MLP / ResNet / ConvBN / ConvLN / MultiChannelMLP
  kan/__init__.py  # KAN / KAN-style MLP
  vit/__init__.py  # ViT 及其 norm layer 构造
```

## 设计目标

- 所有模型共用一个选择入口，统一挂在 `extension` 下。
- 内部仍按模型家族拆分，避免一个大文件堆所有逻辑。
- 保留旧路径兼容层，避免已有脚本或测试立刻失效。

## 对外接口

### `add_model_arguments(parser, task="classification", default_family="mlp")`

给 `argparse.ArgumentParser` 注入统一模型参数。

核心参数：

- `--model-family`：模型家族，当前支持 `mlp | kan | vit`
- `--arch`：具体结构名，例如 `MLP`、`KAN`、`vit_small`

该接口会合并三类模型所需参数：

- MLP 相关：`--width`、`--depth`、`--dropout`、multichannel 参数
- KAN 相关：`--layers-hidden`、`--grid-size`、`--spline-order` 等
- ViT 相关：`--patch-size`、`--in-chans`、`--drop-path-rate` 等

`default_family` 用来控制默认 `arch`：

- `mlp -> MLP`
- `kan -> KAN`
- `vit -> vit_small`

### `get_model(cfg)`

根据 `cfg.model_family` 和 `cfg.arch` 返回对应模型实例。

family 解析顺序：

1. 优先使用 `cfg.model_family`
2. 若未显式提供，则根据 `cfg.arch` 自动推断
3. 若仍无法判断，则回退到调用方给定的默认 family

## Family 说明

### `mlp/`

负责 MLP 家族模型，包括：

- `MLP`
- `PreNormMLP`
- `CenDropScalingMLP`
- `CenDropScalingPreNormMLP`
- `ResMLP`
- `ResCenDropScalingMLP`
- `resnet18 | resnet34 | resnet50`
- `ConvBN*`
- `ConvLN*`
- `MultiChannelMLP`

### `kan/`

负责 KAN 家族模型，包括：

- `KAN`
- `MLP`

保留了 KAN 特有参数，例如：

- `--grid-size`
- `--spline-order`
- `--residual-activation`
- `--disable-residual-branch`

### `vit/`

负责 ViT 家族模型，包括：

- `vit_tiny`
- `vit_small`
- `vit_base`

同时保留：

- `build_vit_norm_layer(cfg)`

用于 ViT 序列维度相关的 norm 构造，现有测试可以继续复用。

## 训练脚本接入方式

当前已接入统一入口的脚本包括：

- `MLP/cifar10.py`
- `MLP/cifar10_ntk.py`
- `PDE/pde.py`
- `PDE/pde_ntk.py`
- `PDE/pde_taiyi.py`
- `KAN/KAN.py`
- `ViT/vit.py`

推荐写法：

```python
parser = argparse.ArgumentParser(...)
ext.model.add_model_arguments(parser, task="classification", default_family="vit")
cfg = parser.parse_args()
model = ext.model.get_model(cfg)
```

## 示例

### MLP

```bash
python MLP/cifar10.py --model-family mlp --arch ResMLP --width 256 --depth 6
```

### KAN

```bash
python KAN/KAN.py --model-family kan --arch KAN --width 32 --depth 3 --grid-size 8
```

### ViT

```bash
python ViT/vit.py --model-family vit --arch vit_small --patch-size 16 --drop-path-rate 0.1
```

## 测试说明

本目录相关测试建议优先使用仓库内环境：

```powershell
.\.conda\python.exe -m py_compile extension\model\__init__.py extension\model\mlp\__init__.py extension\model\kan\__init__.py extension\model\vit\__init__.py
```

如果需要做实际导入或实例化测试，也建议显式使用：

```powershell
.\.conda\python.exe ...
```

这样可以避免系统 Python 缺少 `torch`、`torchvision` 等依赖。
