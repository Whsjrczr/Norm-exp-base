# extension 参数手册

`extension` 是仓库里的公共基础模块。`MLP/`、`PDE/`、`ViT/` 的训练脚本都会通过 `import extension as ext` 复用这里的参数、工具和组件。

这份 README 重点说明：

- 每个参数的具体作用
- 参数的格式
- 默认值
- 典型写法
- 实际代码行为

## 目录作用

- `dataset.py`：数据集与 DataLoader
- `trainer.py`：训练通用参数、种子、环境初始化
- `optimizer.py`：优化器与多阶段优化
- `scheduler.py`：学习率调度器
- `checkpoint.py`：模型保存、加载、恢复
- `logger.py`：日志输出
- `visualization.py`：Visdom 可视化
- `vis_taiyi.py`：WandB / Taiyi 参数
- `normalization.py`：归一化层工厂
- `activation.py`：激活函数工厂
- `utils.py`：命令行字符串解析工具
- `modules/`：基础层
- `my_modules/`：自定义归一化和激活层

## 参数格式总规则

整个仓库大量参数都依赖 `extension/utils.py` 里的解析函数。

### 1. 标量

普通整型、浮点型、字符串参数直接传：

```bash
--epochs 100
--lr 1e-3
--dataset cifar10
```

### 2. 列表格式 `str2list`

格式：

```text
v1,v2,v3
```

示例：

```bash
--batch-size "64,256"
--lr-steps "30,60,90"
--metrics "l2 relative error,MSE"
```

说明：

- 逗号分隔
- 自动尝试转成 `int`、`float`、`True`、`False`、`None`
- 其余保留为字符串

### 3. 字典格式 `str2dict`

格式：

```text
key=value,key=value
```

示例：

```bash
--optimizer-config "momentum=0.9"
--norm-cfg "num_groups=8,affine=True"
--dataset-cfg "loader=vit,drop_last_train=True"
```

说明：

- 每项必须是 `key=value`
- 逗号分隔多个键值对
- 值会自动解析成 `int`、`float`、`bool`、`None` 或字符串
- 不适合复杂嵌套结构

### 4. 元组格式 `str2tuple`

格式与列表相同，但最终转成 tuple：

```bash
--im-size "32,32"
```

### 5. 路径格式 `utils.path`

所有带 `type=utils.path` 的路径参数都会先执行 `expanduser`。

示例：

```bash
--dataset-root ~/dataset
--resume ./results/exp1/checkpoint.pth
```

## `trainer.py`

训练通用参数注册函数：`ext.trainer.add_arguments(parser)`

### `-n, --epochs`

- 类型：`int`
- 默认值：`90`
- 作用：设置总训练轮数
- 影响：训练脚本通常会遍历 `range(start_epoch + 1, epochs)`
- 示例：

```bash
--epochs 200
```

### `--start-epoch`

- 类型：`int`
- 默认值：`-1`
- 作用：手动指定恢复训练后的起始 epoch
- 影响：若从 checkpoint 恢复，通常会被恢复值覆盖
- 示例：

```bash
--start-epoch 49
```

### `-o, --output`

- 类型：`str`
- 默认值：`./results`
- 作用：所有实验输出的根目录
- 影响：日志、模型、checkpoint 都会写到这个目录下
- 示例：

```bash
--output ./results/exp1
```

### `-t, --test`

- 类型：`flag`
- 默认值：关闭
- 作用：只做验证，不进入训练循环
- 影响：训练脚本通常会直接调用 `validate()`
- 示例：

```bash
--test
```

### `--seed`

- 类型：`int`
- 默认值：`-1`
- 作用：设置随机种子
- 实际行为：
  如果值小于 `0`，代码会自动用当前时间戳生成种子
- 示例：

```bash
--seed 1
```

### `--gpu`

- 类型：`int`
- 默认值：`None`
- 作用：保留的 GPU id 参数
- 说明：
  当前多数训练脚本实际通过 `torch.cuda.is_available()` 或环境变量 `CUDA_VISIBLE_DEVICES` 决定设备，这个参数本身未被统一强制使用

## `dataset.py`

数据集参数注册函数：`ext.dataset.add_arguments(parser)`

### `--dataset`

- 类型：`str`
- 默认值：`cifar10`
- 可选值：`mnist`、`fashion-mnist`、`cifar10`、`ImageNet`、`folder`
- 作用：选择数据集类型
- 实际行为：
  `ImageNet` 会在 `dataset-root/ImageNet/` 下查找数据；
  `folder` 会直接把 `dataset-root` 当作数据集根目录
- 示例：

```bash
--dataset cifar10
--dataset folder
```

### `--dataset-cfg`

- 类型：`dict`
- 默认值：`{}`
- 格式：`key=value,key=value`
- 作用：给数据集加载和预处理补充附加选项
- 常用键：
  `grey`、`nogrey`、`random_label`、`loader`、`mean`、`std`、`val_resize_size`、`drop_last_train`
- 示例：

```bash
--dataset-cfg "grey=True"
--dataset-cfg "loader=vit,drop_last_train=True"
```

#### `dataset-cfg` 常见字段说明

`grey`

- 类型：`bool`
- 作用：强制转成灰度图
- 常见于 `cifar10` 或 `folder` 数据集

`nogrey`

- 类型：`bool`
- 作用：显式关闭灰度化
- 与 `grey` 相反

`random_label`

- 类型：`bool`
- 作用：将数据集标签随机打乱并保存为 `.npy`
- 实际行为：
  会在数据集目录下生成或复用 `train_<dataset>_random_labels.npy`、`val_<dataset>_random_labels.npy`

`loader`

- 类型：`str`
- 常见值：`vit`
- 作用：启用 ViT 风格的图像增强和归一化逻辑

`mean`

- 类型：通常期望 list，但受 `str2dict` 限制，命令行不适合直接写复杂列表
- 作用：覆盖默认归一化均值

`std`

- 类型：通常期望 list，但同样不适合直接在 CLI 里写复杂列表
- 作用：覆盖默认归一化方差

`val_resize_size`

- 类型：`int`
- 作用：ViT 风格验证集 resize 尺寸

`drop_last_train`

- 类型：`bool`
- 作用：ViT 风格训练集 DataLoader 是否丢弃最后一个不完整 batch

### `--dataset-root`

- 类型：`path`
- 默认值：`E:\norm-exp\dataset`
- 作用：数据根目录
- 实际行为：
  程序会先检查该路径是否存在；不存在会直接报错
- 示例：

```bash
--dataset-root ./dataset
```

### `-b, --batch-size`

- 类型：`list[int]`
- 默认值：`[64]`
- 格式：`train_bs` 或 `train_bs,val_bs`
- 作用：设置 DataLoader 的 batch size
- 实际行为：
  如果只传一个值，会自动补成 `[train_bs, train_bs]`
- 示例：

```bash
--batch-size 128
--batch-size "64,256"
```

### `-j, --workers`

- 类型：`int`
- 默认值：`4`
- 作用：DataLoader worker 数量
- 示例：

```bash
--workers 8
```

### `--im-size`

- 类型：`tuple`
- 默认值：`(32, 32)`
- 格式：`"h,w"` 或更长 tuple
- 作用：保存图像尺寸信息
- 注意：
  当前 `dataset.py` 里对内置数据集会重设 `args.im_size`，不完全等同于“强制 resize”
- 示例：

```bash
--im-size "224,224"
```

### `--dataset-classes`

- 类型：`int`
- 默认值：`None`
- 作用：数据集类别数
- 实际行为：
  内置数据集和 `ImageFolder` 通常会自动推断或覆盖这个值
- 示例：

```bash
--dataset-classes 1000
```

## 数据目录格式

### 1. `mnist` / `fashion-mnist` / `cifar10`

一般目录如下：

```text
<dataset-root>/
  mnist/
  fashion-mnist/
  cifar10/
```

代码调用 torchvision 数据集接口，必要时会自动下载。

### 2. `ImageNet`

当 `--dataset ImageNet` 时，实际读取路径是：

```text
<dataset-root>/ImageNet/train
<dataset-root>/ImageNet/val
```

目录格式：

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

### 3. `folder`

当 `--dataset folder` 时，直接读取：

```text
<dataset-root>/train
<dataset-root>/val
```

目录格式：

```text
<dataset-root>/
  train/
    class_a/
      xxx.jpg
    class_b/
      yyy.jpg
  val/
    class_a/
    class_b/
```

## 数据预处理行为

### `mnist` / `fashion-mnist`

- `ToTensor()`
- `Normalize((0.5,), (0.5,))`

### `cifar10`

当 `grey=True` 时：

- `Grayscale(num_output_channels=1)`
- `ToTensor()`
- `Normalize((0.5,), (0.5,))`

当不灰度化时：

- `ToTensor()`
- `Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))`

### `folder` / `ImageNet`

普通模式：

- 灰度版：`Grayscale -> ToTensor -> Normalize`
- 彩色版：`ToTensor -> Normalize`

ViT 模式 `loader=vit`：

- 训练：`RandomResizedCrop -> RandomHorizontalFlip -> ToTensor -> Normalize`
- 验证：`Resize -> CenterCrop -> ToTensor -> Normalize`

## `optimizer.py`

优化器参数注册函数：`ext.optimizer.add_arguments(parser)`

### `-oo, --optimizer`

- 类型：`str`
- 默认值：`sgd`
- 可选值：`sgd`、`adam`、`adamw`、`adamax`、`RMSprop`、`lbfgs`
- 作用：指定单阶段训练默认优化器
- 实际行为：
  `build_optimizer()` 会根据名字创建 PyTorch 优化器对象
- 示例：

```bash
--optimizer adam
```

### `-oc, --optimizer-config`

- 类型：`dict`
- 默认值：`{}`
- 格式：`key=value,key=value`
- 作用：传递优化器专用额外参数
- 典型用途：
  `momentum`、`betas`、`max_iter` 等
- 示例：

```bash
--optimizer-config "momentum=0.9"
```

说明：

- `sgd` 即使不传 `momentum`，代码也会自动默认加 `momentum=0.9`
- 复杂嵌套参数不建议走 CLI 直接传

### `-os, --optimizer-stages`

- 类型：`list[dict]`
- 默认值：`None`
- 作用：定义多阶段训练计划
- 支持格式 1：分号分隔字符串

```bash
--optimizer-stages "adam:lr=1e-3,epochs=50; sgd:lr=1e-2,epochs=150,weight_decay=1e-4"
```

- 支持格式 2：Python 字面量列表

```bash
--optimizer-stages "[{'optimizer':'adam','lr':1e-3,'epochs':50},{'optimizer':'sgd','lr':1e-2,'epochs':150}]"
```

- 支持格式 3：从文件读取

```bash
--optimizer-stages @stages.txt
```

常用字段说明：

`optimizer`

- 阶段使用的优化器名

`lr`

- 该阶段学习率

`weight_decay`

- 该阶段权重衰减

`optimizer_config`

- 该阶段额外优化器参数

`epochs`

- epoch-based 训练中该阶段持续多少个 epoch

`iterations`

- PDE/DeepXDE 场景中该阶段持续多少次迭代

`end_epoch`

- 直接指定阶段结束 epoch

`lr_method`

- 可覆盖该阶段使用的调度器类型

`lr_step`

- 可覆盖该阶段的 step 参数

`lr_gamma`

- 可覆盖该阶段衰减系数

### `-wd, --weight-decay`

- 类型：`float`
- 默认值：`0`
- 作用：设置优化器权重衰减
- 示例：

```bash
--weight-decay 1e-4
```

## `scheduler.py`

学习率调度参数注册函数：`ext.scheduler.add_arguments(parser)`

### `--lr-method`

- 类型：`str`
- 默认值：`step`
- 可选值：`fix`、`step`、`steps`、`ploy`、`auto`、`exp`、`user`、`cos`、`1cycle`
- 作用：指定学习率调度策略

具体行为：

`fix`

- 用 `StepLR(optimizer, epochs, lr_gamma)`
- 通常可视为整个训练期间基本不变，到末尾才走一次 step

`step`

- 用 `StepLR(optimizer, lr_step, lr_gamma)`
- 每 `lr_step` 个 epoch 衰减一次

`steps`

- 用 `MultiStepLR(optimizer, lr_steps, lr_gamma)`
- 在指定 epoch 列表处衰减

`ploy`

- 用 `LambdaLR`
- 衰减公式是 `(1 - epoch / epochs) ** lr_gamma`

`auto`

- 用 `ReduceLROnPlateau`
- 监控验证指标长期不下降时自动降低学习率

`exp`

- 用 `ExponentialLR`
- 通过 `lr_gamma / lr` 计算指数底数

`user`

- 用 `LambdaLR`
- 需要调用方自己传 `lr_func`

`cos`

- 用 `CosineAnnealingLR(optimizer, lr_step, lr_gamma)`

`1cycle`

- 用自定义 `LambdaLR`
- 先升后降

### `--lr`

- 类型：`float`
- 默认值：`0.1`
- 作用：初始学习率
- 示例：

```bash
--lr 1e-3
```

### `--lr-step`

- 类型：`int`
- 默认值：`30`
- 作用：step/cos/auto/1cycle 等策略的核心阶段参数
- 示例：

```bash
--lr-step 50
```

### `--lr-gamma`

- 类型：`float`
- 默认值：`0.1`
- 作用：学习率衰减比例或调度器附加参数
- 示例：

```bash
--lr-gamma 0.9
```

### `--lr-steps`

- 类型：`list[int]`
- 默认值：`[]`
- 作用：供 `steps` 调度器使用的多阶段衰减节点
- 示例：

```bash
--lr-steps "30,60,90"
```

## `checkpoint.py`

checkpoint 参数注册函数：`ext.checkpoint.add_arguments(parser)`

### `--resume`

- 类型：`path`
- 默认值：空字符串
- 作用：恢复完整训练状态
- 实际行为：
  会恢复模型参数，并尽量恢复优化器、scheduler，以及 checkpoint 里保存的其他字段
- 示例：

```bash
--resume ./results/exp1/checkpoint.pth
```

### `--load`

- 类型：`path`
- 默认值：空字符串
- 作用：只加载模型权重，不恢复训练状态
- 适合场景：
  预训练权重、finetune 初始化
- 示例：

```bash
--load ./pretrained/model.pth
```

### `--load-no-strict`

- 类型：`flag`
- 默认值：逻辑上为 `True`
- 作用：这个参数名比较反直觉
- 实际行为：
  代码写法是 `action='store_false'`
  也就是说：
  不传时 `cfg.load_no_strict == True`
  传了以后 `cfg.load_no_strict == False`
- 在 `load()` 里实际作为 `strict=self.cfg.load_no_strict` 使用
- 结果：
  不传时严格加载
  传了以后变成非严格加载
- 示例：

```bash
--load-no-strict
```

## `logger.py`

日志参数注册函数：`ext.logger.add_arguments(parser)`

### `--log-suffix`

- 类型：`str`
- 默认值：空字符串
- 作用：结果目录后缀
- 实际行为：
  多数训练脚本会把输出目录组织成：
  `<output>/<model_name>/<log_suffix>/`
- 示例：

```bash
--log-suffix base
```

### `--print-f`

- 类型：`int`
- 默认值：`100`
- 作用：打印频率占位参数
- 说明：
  该参数已注册，但并不是所有训练脚本都实际使用它控制输出频率

## `visualization.py`

Visdom 参数注册函数：`ext.visualization.add_arguments(parser)`

### `--vis`

- 类型：`flag`
- 默认值：关闭
- 作用：启用 Visdom 可视化
- 实际行为：
  如果本机没有安装 `visdom`，代码会自动关闭该功能

### `--vis-port`

- 类型：`int`
- 默认值：`6006`
- 作用：Visdom 服务端口

### `--vis-env`

- 类型：`str`
- 默认值：`None`
- 作用：Visdom 环境名
- 实际行为：
  上层脚本通常会在 `setting(cfg, env_name, names)` 时传入模型名作为环境名

## `vis_taiyi.py`

### `--visualize`

- 类型：`flag`
- 默认值：关闭
- 作用：启用 WandB 记录
- 实际行为：
  多数训练脚本会在该开关打开后调用 `wandb.init()`

### `--wandb-project`

- 类型：`str`
- 默认值：`test`
- 作用：设置 WandB project 名

### `--taiyi`

- 类型：`flag`
- 默认值：关闭
- 作用：启用 Taiyi 监控
- 实际行为：
  打开后，训练脚本通常会构建 `Monitor(...)` 并把统计量推给 WandB

## `normalization.py`

归一化参数注册函数：`ext.normalization.add_arguments(parser)`

### `--norm`

- 类型：`str`
- 默认值：`No`
- 作用：选择归一化层类型
- 可选值：
  `BN`?`GN`?`LN`?`IN`?`LNc`?`LNs`?`RMS`?`CDS`?`BNc`?`BNs`?`bCDS`?`bClCDS`?`bCLN`?`bCRMS`?`GNc`?`GNs`?`PLN`?`PLS`?`PQN`?`No`?`no`

具体含义：

`BN`

- BatchNorm

`GN`

- GroupNorm

`LN`

- 标准 LayerNorm

`IN`

- InstanceNorm

`LNc`

- LayerNormCentering

`LNs`

- LayerNormScaling

`RMS`

- RMSNorm 风格实现

`CDS`

- Centering -> Dropout -> Scaling 的组合模块

`BNc`

- BatchNormCentering

`BNs`

- BatchNormScaling

`SeqBN`

- SequenceBatchNorm = SequenceBatchNormCentering -> SequenceBatchNormScaling
- Supports sequence-axis normalization for inputs with `dim >= 3`

`SeqBNc`

- SequenceBatchNormCentering
- For inputs with `dim >= 3`, normalize along the sequence axis instead of the channel axis

`SeqBNs`

- SequenceBatchNormScaling
- For inputs with `dim >= 3`, normalize along the sequence axis instead of the channel axis

`DSeqBN`

- DynamicSequenceBatchNorm = DynamicSequenceBatchNormCentering -> DynamicSequenceBatchNormScaling
- Supports variable-length sequence inputs with `dim >= 3`
- Does not keep per-position running statistics

`DSeqBNc`

- DynamicSequenceBatchNormCentering
- Variable-length sequence centering without fixed `seq_len`

`DSeqBNs`

- DynamicSequenceBatchNormScaling
- Variable-length sequence scaling without fixed `seq_len`

`bCDS`

- BN Centering + Dropout + Layer Scaling

`bClCDS`

- BN Centering + LN Centering + Dropout + Scaling

`bCLN`

- BN Centering + LayerNorm

`bCRMS`

- BN Centering + RMSNorm

`GNc`

- GroupNormCentering

`GNs`

- GroupNormScaling

`PLN`

- Parallel LayerNorm，默认 `centering=True`

`PLS`

- Parallel LayerScaling，默认 `centering=False`

`PQN`

- `(p, q)-normalization`????? Definition 5.1
- ??????????? `sign(x) * |x|^(p/q) / (mean(|x|^p) + eps)^(1/q)`
- ?? `centering=True` ?????????????? `(p, q)` normalization
- ? `affine=True` ?????????

`No` / `no`

- 恒等映射，不做归一化

### `--norm-cfg`

- 类型：`dict`
- 默认值：`{}`
- 格式：`key=value,key=value`
- 作用：传给归一化层构造函数的附加参数

常见字段：

`num_groups`

- 用于 `GN`、`GNc`、`GNs`
- 表示 group 数

`num_per_group`

- 用于 `PLN`、`PLS`
- 表示每组特征数
`p` / `q`

- ?? `PQN`
- ?? `p=4,q=2`


`p` / `q`

- ?? `pqact`
- ?? `p=4,q=2`

`eps`

- 数值稳定项

`momentum`

- 用于 BatchNorm 相关层

`affine`

- 是否启用可学习仿射参数

`dim`

- 用于 BN/IN 相关层，区分 1D 或 2D
- 代码中通常 `dim=4` 代表 2D，其他值走 1D

`dropout_prob`

- 用于 `CDS` 等组合层

示例：

```bash
--norm LN
--norm-cfg "affine=False"

--norm GN
--norm-cfg "num_groups=8,affine=True"

--norm PLN
--norm-cfg "num_per_group=8,affine=False"

--norm PQN
--norm-cfg "p=4,q=2,affine=False"

--norm PQN
--norm-cfg "num_per_group=8,p=4,q=2,affine=False"

--norm PQN
--norm-cfg "num_per_group=8,p=2,q=2,centering=True,affine=False"
```

## `activation.py`

激活参数注册函数：`ext.activation.add_arguments(parser)`

### `--activation`

- 类型：`str`
- 默认值：`relu`
- ????`relu`?`sigmoid`?`tanh`?`gn`?`pgn`?`sinarctan`?`pqact`?`no`
- 作用：指定激活函数或激活式模块

具体含义：

`relu`

- `nn.ReLU`

`sigmoid`

- `nn.Sigmoid`

`tanh`

- `nn.Tanh`

`gn`

- 这里不是普通激活函数，而是把 `GroupNorm` 当作激活模块插入

`pgn`

- PointwiseGroupNorm

`sinarctan`

- 自定义 `SinArctan`

`pqact`

- ???? `(p, q)` activation
- ???`sign(x) * |x|^(p/q) / (|x|^p + 1)^(1/q)`
- ? `p=q=2` ???? `sinarctan`

`no`

- `Identity`

### `--activation-cfg`

- 类型：`dict`
- 默认值：`{}`
- 格式：`key=value,key=value`
- 作用：传给激活模块构造函数的附加参数

常见字段：

`inplace`

- 用于 `relu`

`num_groups`

- 用于 `gn`、`pgn`

`eps`

- 用于 group norm 风格模块

示例：

```bash
--activation relu
--activation-cfg "inplace=True"

--activation pgn
--activation-cfg "num_groups=16"

--activation pqact
--activation-cfg "p=4,q=2"
```

## `logger` / `checkpoint` / `scheduler` 的典型组合

```bash
--output ./results/exp1 \
--log-suffix base \
--optimizer adam \
--optimizer-config "betas=0.9" \
--weight-decay 1e-4 \
--lr-method step \
--lr 1e-3 \
--lr-step 30 \
--lr-gamma 0.1 \
--resume ./results/exp1/checkpoint.pth
```

## `modules/` 和 `my_modules/`

### `modules/`

`View`

- 作用：reshape / flatten 输入

`Scale`

- 作用：可学习缩放因子

`Sequential`

- 作用：顺序容器扩展

### `my_modules/`

????????????? `normalization.py` ? `activation.py` ?????
???????

`norm/`

- `ln_modules.py`
- `bn1d_modules.py`
- `bn2d_modules.py`
- `gn_modules.py`
- `pln.py`
- `pq_norm.py`

`activation/`

- `pgn_modules.py`
- `pq_activation.py`
- `sinarctan.py`
- `dyt.py`

??????

- `copy.py`

## `my_modules/`

这里是自定义层实现，主要被 `normalization.py` 和 `activation.py` 注册调用：

- `ln_modules.py`
- `bn1d_modules.py`
- `bn2d_modules.py`
- `gn_modules.py`
- `pln.py`
- `pgn_modules.py`
- `sinarctan.py`

## 常见命令示例

### 1. CIFAR10 + MLP

```bash
python MLP/cifar10.py \
  --dataset cifar10 \
  --dataset-root ./dataset \
  --batch-size "64,1000" \
  --epochs 200 \
  --optimizer adam \
  --lr 1e-3 \
  --lr-method step \
  --lr-step 50 \
  --lr-gamma 0.5 \
  --norm LN \
  --activation relu \
  --seed 1
```

### 2. ImageFolder + ViT 风格加载

```bash
python ViT/vit.py \
  --dataset folder \
  --dataset-root ./dataset/ImageNet \
  --dataset-cfg "loader=vit,drop_last_train=True" \
  --batch-size "256,256" \
  --workers 8
```

### 3. 多阶段优化

```bash
--optimizer-stages "adam:lr=1e-3,epochs=50; sgd:lr=1e-2,weight_decay=1e-4,epochs=150"
```

## 实际使用注意

- `--load-no-strict` 的命名和行为相反，传入后会关闭严格匹配
- `--im-size` 当前更像“记录尺寸信息”，不是所有数据集都会真的执行 resize
- `--dataset-cfg` / `--optimizer-config` 适合简单键值对，不适合复杂嵌套
- `--gpu` 参数已注册，但很多训练脚本主要还是靠 `CUDA_VISIBLE_DEVICES` 和自动设备检测
- `--print-f` 已注册，但不是所有脚本都真正按它控制打印频率
## Update

- `dataset.py` 里颜色开关语义已更新为“默认保留 RGB，只有 `grey=True` 时才灰度化”
- `nogrey` 仅保留兼容，不建议继续使用；`nogrey=True` 等价于 `grey=False`
- `-b, --batch-size` 传单个值时会自动扩成 `[train_bs, val_bs]`
- 对 `folder/ImageNet` 且 `dataset_cfg.loader=vit` 的加载方式，`--im-size` 会直接决定 ViT 风格的 crop / resize 尺寸
## Visualization Update

### Module Naming

- 新的统一入口是 `tracking.py`
- 新的 Visdom 专用入口是 `visdom.py`
- 新的 Taiyi 专用入口是 `taiyi.py`
- `visualize.py` 现在只是 `tracking.py` 的兼容别名
- `visualization.py` 现在只是 `visdom.py` 的兼容别名
- `vis_taiyi.py` 暂时保留为旧参数兼容入口

### Recommended Usage

- 推荐在训练脚本中使用 `ext.tracking.add_arguments(parser)`
- 推荐在需要 Taiyi 时显式使用 `ext.taiyi.add_arguments(parser)`
- 推荐用 `ext.tracking.normalize_config(args)` 做参数归一化
- 推荐用 `ext.tracking.setting(...)` 创建统一追踪器
- 推荐用 `ext.taiyi.setting(...)` 创建 Taiyi 追踪器
- 如果只需要 Visdom，可以直接使用 `ext.visdom.add_arguments(parser)` 和 `ext.visdom.setting(...)`

### Visualization Flags

`tracking.py` 现在统一管理 WandB、Visdom、Taiyi，但三者开关已经分离。

`--wandb`

- 类型：`flag`
- 作用：启用 WandB 记录

`--visualize`

- 类型：`flag`
- 作用：`--wandb` 的兼容别名

`--wandb-project`

- 类型：`str`
- 默认值：`test`
- 作用：设置 WandB project 名称

`--visdom`

- 类型：`flag`
- 作用：启用 Visdom 曲线可视化

`--vis`

- 类型：`flag`
- 作用：`--visdom` 的兼容别名

`--visdom-port` / `--vis-port`

- 类型：`int`
- 默认值：`6006`
- 作用：设置 Visdom 服务端口

`--visdom-env` / `--vis-env`

- 类型：`str`
- 默认值：`None`
- 作用：设置 Visdom 环境名；如果不传，通常会回退到 `model_name`

`--taiyi`

- 类型：`flag`
- 作用：启用 Taiyi 监控桥接
- 当前行为：Taiyi 开关已和 WandB / Visdom 分离，但仍通过 `tracking.py` 统一接入
- 后续方向：可以继续把 Taiyi 抽成 `extension` 下的独立工具模块

## 2026-04 Normalization Update

### Explicit dim/layout in `ext.normalization.Norm`

`Norm` now supports explicit tensor layout control and should no longer rely on the old implicit CNN-only assumption.

Recommended usage:

```python
ext.Norm(num_features, dim=2)
ext.Norm(num_features, dim=3, layout="last")
ext.Norm(num_features, dim=4)
```

Meaning:

- `dim=2`: `(N, C)`, used by MLP/KAN fully-connected paths.
- `dim=3, layout="last"`: `(B, N, C)`, used by ViT token sequences.
- `dim=3, layout="first"`: `(B, C, L)`.
- `dim=4`: `(N, C, H, W)`, used by CNN paths.

For channel-first norm families such as `BN`, `BNc`, `BNs`, `GN`, `GNc`, `GNs`, and `IN`, the factory now automatically adapts token-last inputs when `layout="last"` is given.

For sequence-axis BatchNorm families `SeqBN`, `SeqBNc`, and `SeqBNs`, `num_features` means sequence length rather than hidden size.

Layout convention:

- `layout="last"`: `(B, N, ..., C)`
- `layout="first"`: `(B, C, ..., N)`

For dynamic sequence-axis BatchNorm families `DSeqBN`, `DSeqBNc`, and `DSeqBNs`, no fixed `num_features` is required. They are intended for variable-length sequence inputs and use current-batch statistics only.

### `make_norm_factory`

Preferred wiring pattern:

```python
norm_2d = ext.make_norm_factory(dim=2)
norm_vit = ext.make_norm_factory(dim=3, layout="last")
norm_4d = ext.make_norm_factory(dim=4)
```

### Notes

- `InstanceNorm` is intentionally rejected for `dim=2` because pure `(N, C)` tensors have no spatial axis for instance statistics.
- `bCLN`, `bCRMS`, `PLN`, `PLS`, and `PQN` now work on both 2D MLP inputs and 3D ViT token inputs.
- `BN/GN/IN` can now be used on ViT through `dim=3, layout="last"`.
- `SeqBN/SeqBNc/SeqBNs` support sequence inputs with `dim >= 3` and compute statistics per sequence position, not per channel.
- `DSeqBN/DSeqBNc/DSeqBNs` support variable-length sequence inputs, but they do not maintain per-position running statistics and their affine parameters are shared scalars rather than per-position vectors.

### CLI examples

```bash
# MLP / KAN
--norm BN
--norm-cfg "dim=2"

# ViT
--norm GN
--norm-cfg "num_groups=6,dim=3,layout=last"

# ViT with sequence-axis BN centering
--norm SeqBNc
--norm-cfg "dim=3,layout=last"

# ViT with sequence-axis BN scaling
--norm SeqBNs
--norm-cfg "dim=3,layout=last"

# ViT with sequence-axis BN (centering -> scaling)
--norm SeqBN
--norm-cfg "dim=3,layout=last"

# Higher-dimensional sequence input with sequence axis at the end
--norm SeqBN
--norm-cfg "dim=4,layout=first"

# Variable-length sequence BN
--norm DSeqBN
--norm-cfg "dim=3,layout=last"

# CNN
--norm BN
--norm-cfg "dim=4"

# ViT with PLN
--norm PLN
--norm-cfg "num_per_group=8,dim=3,layout=last"

# ViT with PQN
--norm PQN
--norm-cfg "p=4,q=2,dim=3,layout=last"
```
