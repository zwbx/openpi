# 数据载入流程分析

本文档详细分析 OpenPI 项目中数据读取部分如何读取配置文件以及整个数据加载 pipeline。

## 目录

1. [配置系统架构](#配置系统架构)
2. [完整数据加载 Pipeline](#完整数据加载-pipeline)
3. [配置文件解析流程](#配置文件解析流程)
4. [数据集创建流程](#数据集创建流程)
5. [数据转换 Pipeline](#数据转换-pipeline)
6. [关键代码路径](#关键代码路径)

---

## 配置系统架构

### 1. 配置层级结构

```
TrainConfig (顶层训练配置)
├── model: BaseModelConfig (模型配置)
├── data: DataConfigFactory (数据配置工厂)
│   ├── repo_id: str (数据集ID)
│   ├── assets: AssetsConfig (资产配置)
│   ├── repack_transforms: Group (重打包转换)
│   ├── data_transforms: Group (数据转换)
│   └── model_transforms: Group (模型转换)
├── optimizer: OptimizerConfig
├── lr_schedule: LRScheduleConfig
└── ... (其他训练超参数)
```

### 2. 配置定义位置

**文件**: `src/openpi/training/config.py`

所有可用的配置定义在 `_CONFIGS` 列表中 (line 590-1148):

```python
_CONFIGS = [
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        ...
    ),
    TrainConfig(
        name="pi05_simpler_zscore",
        model=pi0_config.Pi0Config(pi05=True, ...),
        data=LeRobotSimplerDataConfig(
            repo_id="lerobot-pi0-bridge",
            base_config=DataConfig(
                prompt_from_task=True,
                dataset_root="/dev/shm/lerobot-pi0-bridge",
                use_quantile_norm=False,
            ),
        ),
        ...
    ),
    ...
]
```

---

## 完整数据加载 Pipeline

### Pipeline 概览图

```
1. 命令行参数
   ↓
2. Config 解析 (tyro.cli)
   ↓
3. DataConfigFactory.create()
   ↓
4. 数据集创建 (create_torch_dataset / create_rlds_dataset)
   ↓
5. 数据转换 (transform_dataset)
   ↓
6. DataLoader 包装 (TorchDataLoader / RLDSDataLoader)
   ↓
7. 训练循环迭代
```

---

## 配置文件解析流程

### Step 1: 命令行启动

**文件**: `scripts/train_pytorch.py` (line 639-646)

```python
def main():
    init_logging()
    config = _config.cli()  # ← 核心入口：解析配置
    train_loop(config)
```

### Step 2: tyro CLI 解析

**文件**: `src/openpi/training/config.py` (line 1155-1156)

```python
def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )
```

**工作原理**:
- tyro 是一个基于类型注解的命令行参数解析库
- `overridable_config_cli` 允许用户选择预定义的配置，并通过命令行参数覆盖任意字段

**示例命令**:
```bash
# 使用预定义配置
python scripts/train_pytorch.py pi05_simpler_zscore --exp_name my_experiment

# 覆盖配置字段
python scripts/train_pytorch.py pi05_simpler_zscore \
    --exp_name my_experiment \
    --batch_size 512 \
    --num_train_steps 50000 \
    --data.base_config.dataset_root /custom/path
```

### Step 3: DataConfigFactory 解析

**文件**: `src/openpi/training/config.py` (line 169-203)

每个 `DataConfigFactory` 子类定义了如何创建 `DataConfig`:

```python
@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path,
               model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""
```

**关键子类**:
- `FakeDataConfig`: 生成假数据 (用于调试)
- `SimpleDataConfig`: 简单配置
- `LeRobotAlohaDataConfig`: Aloha 机器人数据
- `LeRobotLiberoDataConfig`: Libero 仿真数据
- `LeRobotSimplerDataConfig`: SimplerEnv 数据
- `RLDSDroidDataConfig`: DROID RLDS 格式数据

---

## 数据集创建流程

### Step 1: 创建 DataLoader

**文件**: `scripts/train_pytorch.py` (line 125-128)

```python
def build_datasets(config: _config.TrainConfig):
    # 使用统一的数据加载器，指定 PyTorch 框架
    data_loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True
    )
    return data_loader, data_loader.data_config()
```

### Step 2: create_data_loader 路由

**文件**: `src/openpi/training/data_loader.py` (line 227-272)

```python
def create_data_loader(
    config: _config.TrainConfig,
    *,
    framework: Literal["jax", "pytorch"] = "jax",
    ...
) -> DataLoader:
    # 1. 创建 DataConfig
    data_config = config.data.create(config.assets_dirs, config.model)

    # 2. 路由到具体的数据加载器
    if data_config.rlds_data_dir is not None:
        # RLDS 格式 (用于大规模数据集如 DROID)
        return create_rlds_data_loader(...)
    else:
        # LeRobot 格式 (默认)
        return create_torch_data_loader(...)
```

### Step 3: 创建原始数据集

#### 3.1 LeRobot 数据集

**文件**: `src/openpi/training/data_loader.py` (line 130-155)

```python
def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig
) -> Dataset:
    repo_id = data_config.repo_id

    if repo_id == "fake":
        # 生成假数据 (用于测试)
        return FakeDataset(model_config, num_samples=1024)

    # 加载 LeRobot 数据集
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        repo_id,
        root=data_config.dataset_root
    )

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=data_config.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    # 可选：从任务字段提取 prompt
    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    return dataset
```

**关键参数**:
- `repo_id`: HuggingFace 数据集 ID 或本地数据集名称
- `dataset_root`: 本地数据集根目录 (如果为 None，则从 HuggingFace 下载)
- `delta_timestamps`: 动作序列的时间偏移量 (基于 FPS 和 action_horizon)

#### 3.2 RLDS 数据集 (DROID)

**文件**: `src/openpi/training/data_loader.py` (line 158-173)

```python
def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # RLDS 格式主要用于 DROID 数据集
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )
```

### Step 4: 应用数据转换

**文件**: `src/openpi/training/data_loader.py` (line 176-195)

```python
def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False
) -> Dataset:
    # 1. 加载归一化统计信息
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found.")
        norm_stats = data_config.norm_stats

    # 2. 组合所有转换
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,    # 重打包转换
            *data_config.data_transforms.inputs,      # 数据转换
            _transforms.Normalize(norm_stats, ...),   # 归一化
            *data_config.model_transforms.inputs,     # 模型转换
        ],
    )
```

### Step 5: 创建 PyTorch DataLoader

**文件**: `src/openpi/training/data_loader.py` (line 275-341)

```python
def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader:
    # 1. 创建并转换数据集
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=False)

    # 2. 处理分布式训练 (DDP)
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    # 3. 创建 TorchDataLoader 包装器
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)
```

---

## 数据转换 Pipeline

### 转换阶段

数据在加载过程中经过**四个阶段**的转换，**数据增强在模型 forward 时动态应用**：

```
原始数据 → Repack → Data Transform → Normalize → Model Transform → 模型输入
                                                          ↓
                                              (训练时) Data Augmentation
```

### 1. Repack Transform (重打包转换)

**目的**: 将数据集的原始字段映射到统一的字段名

**示例**: `LeRobotLiberoDataConfig` (config.py line 339-351)

```python
repack_transform = _transforms.Group(
    inputs=[
        _transforms.RepackTransform({
            "observation/image": "image",              # 映射: 数据集字段 -> 统一字段
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        })
    ]
)
```

**应用场景**:
- 数据集字段名与推理环境不一致时使用
- 仅在训练时应用，推理时不需要

### 2. Data Transform (数据转换)

**目的**: 执行特定机器人平台的数据预处理

**示例**: `LiberoInputs` 和 `LiberoOutputs`

```python
data_transforms = _transforms.Group(
    inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
    outputs=[libero_policy.LiberoOutputs()],
)
```

**常见操作**:
- 坐标系转换
- 动作空间转换 (绝对位置 ↔ 相对位置)
- 图像预处理
- 状态向量重组

**可选**: Delta Actions 转换

某些数据集使用绝对动作，需要转换为相对动作：

```python
# 前 6 个维度 (关节) 转换为 delta，最后一个 (夹爪) 保持绝对
delta_action_mask = _transforms.make_bool_mask(6, -1)
data_transforms = data_transforms.push(
    inputs=[_transforms.DeltaActions(delta_action_mask)],
    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
)
```

### 3. Normalize (归一化)

**文件**: `src/openpi/training/data_loader.py` (line 192)

```python
_transforms.Normalize(
    norm_stats,
    use_quantiles=data_config.use_quantile_norm
)
```

**归一化统计信息来源**:

```python
# 从 assets 目录加载
def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None):
    if asset_id is None:
        return None
    data_assets_dir = str(assets_dir / asset_id)
    norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
    return norm_stats
```

**位置**: `assets/{config_name}/{asset_id}/norm_stats.json`

**生成方法**:
```bash
python scripts/compute_norm_stats.py --config-name=<your-config>
```

### 4. Model Transform (模型转换)

**目的**: 将数据转换为模型期望的格式

**文件**: `src/openpi/training/config.py` (line 108-165)

```python
class ModelTransformFactory(GroupFactory):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    ),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ])
            case _model.ModelType.PI05:
                return _transforms.Group(inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ])
```

**操作**:
1. 注入默认 prompt
2. 调整图像尺寸 (224x224)
3. Tokenize prompt 和状态
4. Padding 动作和状态到固定维度

---

## 关键代码路径

### 文件结构

```
src/openpi/training/
├── config.py                  # 配置定义和解析
├── data_loader.py             # 数据加载器实现
├── droid_rlds_dataset.py      # DROID RLDS 数据集
└── ...

src/openpi/transforms.py       # 数据转换函数
src/openpi/policies/
├── aloha_policy.py            # Aloha 数据转换
├── libero_policy.py           # Libero 数据转换
├── droid_policy.py            # DROID 数据转换
└── simpler_policy.py          # SimplerEnv 数据转换

scripts/
├── train_pytorch.py           # PyTorch 训练入口
├── train.py                   # JAX 训练入口
└── compute_norm_stats.py      # 计算归一化统计
```

### 调用链总结

```
训练脚本 (train_pytorch.py)
  main() [line 639]
    ↓
  _config.cli() [line 641]
    ↓
  train_loop(config) [line 642]
    ↓
  build_datasets(config) [line 125]
    ↓
  _data.create_data_loader(config, framework="pytorch") [line 127]
    ↓
  data_config = config.data.create(assets_dirs, model_config) [line 246]
    ↓
  create_torch_data_loader(...) [line 275]
    ↓
  dataset = create_torch_dataset(...) [line 306]
    ↓
  dataset = transform_dataset(dataset, data_config) [line 307]
    ↓
  TorchDataLoader(dataset, ...) [line 329]
    ↓
  DataLoaderImpl(data_config, data_loader) [line 341]
```

---

## 配置示例解析

### 示例：pi05_simpler_zscore

```python
TrainConfig(
    name="pi05_simpler_zscore",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=4,
        discrete_state_input=True
    ),
    data=LeRobotSimplerDataConfig(
        repo_id="lerobot-pi0-bridge",
        base_config=DataConfig(
            prompt_from_task=True,
            dataset_root="/dev/shm/lerobot-pi0-bridge",
            use_quantile_norm=False,  # 使用 z-score 归一化
        ),
    ),
    batch_size=1024,
    num_train_steps=100_000,
    pytorch_weight_path="/dev/shm/pi05_base_pytorch",
    num_workers=32,
)
```

**执行流程**:

1. **配置解析**:
   ```bash
   python scripts/train_pytorch.py pi05_simpler_zscore --exp_name my_exp
   ```

2. **数据配置创建** (`LeRobotSimplerDataConfig.create()`):
   - 加载 norm_stats: `assets/pi05_simpler_zscore/lerobot-pi0-bridge/norm_stats.json`
   - 创建 repack_transforms: 映射数据集字段
   - 创建 data_transforms: SimplerInputs + SimplerOutputs
   - 创建 model_transforms: Pi05 tokenization + padding

3. **数据集加载**:
   - 从 `/dev/shm/lerobot-pi0-bridge` 加载 LeRobot 数据集
   - repo_id: `lerobot-pi0-bridge`
   - action_horizon: 4 (每个样本包含 4 步动作序列)

4. **数据转换链**:
   ```
   原始样本
     ↓ RepackTransform
   {"image": ..., "state": ..., "actions": ..., "prompt": ...}
     ↓ SimplerInputs
   {坐标系转换、图像预处理}
     ↓ Normalize (z-score)
   {归一化状态和动作}
     ↓ ModelTransform
   {Resize(224,224), Tokenize, Pad}
     ↓
   模型输入
   ```

5. **训练循环**:
   ```python
   for observation, actions in loader:
       losses = model(observation, actions)
       loss = losses.mean()
       loss.backward()
       ...
   ```

---

## 常见问题

### Q1: 如何添加自定义数据集？

**步骤**:

1. **定义 DataConfigFactory**:
   ```python
   @dataclasses.dataclass(frozen=True)
   class MyCustomDataConfig(DataConfigFactory):
       @override
       def create(self, assets_dirs, model_config) -> DataConfig:
           repack_transform = _transforms.Group(inputs=[...])
           data_transforms = _transforms.Group(inputs=[...], outputs=[...])
           model_transforms = ModelTransformFactory()(model_config)

           return dataclasses.replace(
               self.create_base_config(assets_dirs, model_config),
               repack_transforms=repack_transform,
               data_transforms=data_transforms,
               model_transforms=model_transforms,
           )
   ```

2. **添加到 _CONFIGS**:
   ```python
   TrainConfig(
       name="my_custom_config",
       model=pi0_config.Pi0Config(),
       data=MyCustomDataConfig(repo_id="my/dataset"),
       ...
   )
   ```

3. **计算 norm_stats**:
   ```bash
   python scripts/compute_norm_stats.py --config-name=my_custom_config
   ```

### Q2: 归一化统计信息存储在哪里？

**位置**: `{assets_base_dir}/{config_name}/{asset_id}/norm_stats.json`

**示例**:
- config_name: `pi05_simpler_zscore`
- asset_id: `lerobot-pi0-bridge`
- 路径: `./assets/pi05_simpler_zscore/lerobot-pi0-bridge/norm_stats.json`

**加载代码**: `config.py` line 192-202

### Q3: 如何在 Fine-tuning 时复用 Base Model 的 norm_stats？

**使用 AssetsConfig**:

```python
data=LeRobotAlohaDataConfig(
    repo_id="my-custom-aloha-dataset",
    assets=AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",  # 复用 trossen 的 norm_stats
    ),
)
```

### Q4: RLDS 和 LeRobot 格式有什么区别？

| 特性 | LeRobot | RLDS (DROID) |
|------|---------|--------------|
| 数据规模 | 小到中型 (<10小时) | 大规模 (100+ 小时) |
| 加载方式 | PyTorch Dataset | TensorFlow RLDS |
| Batching | PyTorch DataLoader | 内部处理 |
| num_workers | 支持多进程 | 必须为 0 |
| 使用场景 | Fine-tuning, 小规模训练 | 大规模预训练 |

---

## 数据增强 (Data Augmentation)

### ⚠️ 关键发现：数据增强不在数据加载时应用

与大多数深度学习框架不同，OpenPI 的**数据增强不在 DataLoader 中应用**，而是在**模型的 forward pass 中动态应用**。

### 增强应用位置

**文件**: `src/openpi/models_pytorch/preprocessing_pytorch.py`

**函数**: `preprocess_observation_pytorch(observation, *, train=False)`

**调用时机**:
1. **训练时** (`pi0_pytorch.py:708`):
   ```python
   def forward(self, observation, actions, ...):
       images, img_masks, lang_tokens, lang_masks, state = \
           self._preprocess_observation(observation, train=True)  # ← 启用增强
   ```

2. **推理时** (`pi0_pytorch.py:1044`):
   ```python
   def sample_actions(self, observation, ...):
       images, img_masks, lang_tokens, lang_masks, state = \
           self._preprocess_observation(observation, train=False)  # ← 禁用增强
   ```

### 增强类型

#### 1. 几何增强 (仅非手腕相机)

**Random Crop + Resize**:
```python
# preprocessing_pytorch.py line 65-84
crop_height = int(height * 0.95)  # 裁剪到 95%
crop_width = int(width * 0.95)

# 随机裁剪起始位置
start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
start_w = torch.randint(0, max_w + 1, (1,), device=image.device)

# 裁剪并 resize 回原始尺寸
image = image[:, start_h:start_h+crop_height, start_w:start_w+crop_width, :]
image = F.interpolate(image, size=(height, width), mode='bilinear')
```

**Random Rotation**:
```python
# preprocessing_pytorch.py line 86-121
angle = torch.rand(1, device=image.device) * 10 - 5  # -5° 到 +5°

# 使用 grid_sample 实现旋转
grid_x_rot = grid_x * cos_a - grid_y * sin_a
grid_y_rot = grid_x * sin_a + grid_y * cos_a
image = F.grid_sample(image, grid, mode='bilinear')
```

#### 2. 颜色增强 (所有相机)

**Random Brightness**:
```python
# preprocessing_pytorch.py line 124-127
brightness_factor = 0.7 + torch.rand(1) * 0.6  # 0.7-1.3
image = image * brightness_factor
```

**Random Contrast**:
```python
# preprocessing_pytorch.py line 129-133
contrast_factor = 0.6 + torch.rand(1) * 0.8  # 0.6-1.4
mean = image.mean(dim=[1, 2, 3], keepdim=True)
image = (image - mean) * contrast_factor + mean
```

**Random Saturation**:
```python
# preprocessing_pytorch.py line 135-140
saturation_factor = 0.5 + torch.rand(1) * 1.0  # 0.5-1.5
gray = image.mean(dim=-1, keepdim=True)
image = gray + (image - gray) * saturation_factor
```

### 设计原则

根据 `src/openpi/shared/embodiment_config.py` 的注释：

**影响 embodiment context 的因素** (需要不同的 TTT W 参数):
- 图像的几何变换 (crop, rotation, flip)
- 机器人配置 (DOF, 动作空间)
- 坐标系选择

**不影响 embodiment context 的因素** (共享 TTT W 参数):
- ✅ **外观增强** (color jitter, brightness, contrast)
- 这些只改变外观，不改变 observation-action-state 的对应关系

### 为什么在模型 forward 中应用？

**优点**:
1. **灵活性**: 可以根据模型状态 (`self.training`) 自动切换
2. **一致性**: 训练和推理使用同一套预处理代码
3. **GPU 加速**: 增强操作在 GPU 上执行，利用 PyTorch 的自动微分
4. **torch.compile 兼容**: 所有操作都是 tensor 操作，可以被 JIT 编译

**缺点**:
1. **占用 GPU 内存**: 增强在 GPU 上进行，增加显存占用
2. **无法在 DataLoader worker 中并行**: 传统方法可以用多个 CPU worker 并行增强

### 如何修改增强策略？

**修改增强参数**:
```python
# 编辑 src/openpi/models_pytorch/preprocessing_pytorch.py

# 修改裁剪比例 (line 66-67)
crop_height = int(height * 0.90)  # 改为 90%
crop_width = int(width * 0.90)

# 修改旋转角度 (line 88)
angle = torch.rand(1) * 20 - 10  # 改为 -10° 到 +10°

# 修改亮度范围 (line 126)
brightness_factor = 0.5 + torch.rand(1) * 1.0  # 改为 0.5-1.5
```

**禁用特定增强**:
```python
# 禁用旋转：注释掉 line 86-121

# 禁用颜色增强：注释掉 line 124-146

# 只对特定相机应用增强
if "wrist" not in key and "top" not in key:  # 只对非手腕、非顶部相机增强
    # ... 增强代码
```

### 增强流程总结

```python
# 完整的数据增强流程

1. DataLoader 返回原始数据
   ↓
2. 模型 forward() 被调用
   ↓
3. _preprocess_observation(observation, train=True)
   ↓
4. preprocess_observation_pytorch(observation, train=True)
   ↓
5. if train:
       # 几何增强 (非手腕相机)
       - Random Crop (95%)
       - Random Rotation (-5° to +5°)

       # 颜色增强 (所有相机)
       - Random Brightness (0.7-1.3)
       - Random Contrast (0.6-1.4)
       - Random Saturation (0.5-1.5)
   ↓
6. 增强后的图像进入模型
```

### 调试数据增强

**查看增强效果**:
```python
# 在 preprocessing_pytorch.py 中添加可视化代码
import matplotlib.pyplot as plt

def visualize_augmentation(original_image, augmented_image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_image)
    ax1.set_title("Original")
    ax2.imshow(augmented_image)
    ax2.set_title("Augmented")
    plt.show()

# 在 preprocess_observation_pytorch 中调用
if train and DEBUG:
    visualize_augmentation(image_before, image_after)
```

**禁用增强进行消融实验**:
```python
# 方法 1: 修改代码强制 train=False
observation = _preprocessing.preprocess_observation_pytorch(
    observation,
    train=False  # 即使在训练时也禁用增强
)

# 方法 2: 添加配置选项
if self.config.use_augmentation and train:
    # 应用增强
```

---

## 总结

OpenPI 的数据加载系统采用了**工厂模式 + 转换链 + 动态增强**的设计：

1. **配置驱动**: 所有配置通过 tyro CLI 解析，支持命令行覆盖
2. **模块化转换**: 数据经过 Repack → Data → Normalize → Model 四阶段转换
3. **动态增强**: 数据增强在模型 forward 中应用，而非数据加载时
4. **灵活扩展**: 通过继承 DataConfigFactory 可轻松添加新数据集
5. **统一接口**: JAX 和 PyTorch 共享同一套数据加载逻辑

**核心文件**:
- `src/openpi/training/config.py`: 配置定义
- `src/openpi/training/data_loader.py`: 数据加载实现
- `src/openpi/models_pytorch/preprocessing_pytorch.py`: **数据增强实现** ⭐
- `scripts/train_pytorch.py`: 训练入口

**关键设计原则**:
- 数据集无关的转换 (Normalize, Model Transform) 在 data_loader.py 中实现
- 数据集特定的转换 (Repack, Data Transform) 在各自的 policy 文件中实现
- **数据增强在模型 forward 中应用，利用 GPU 加速和 torch.compile 优化**
- Norm stats 存储在 assets 目录，可跨配置复用
