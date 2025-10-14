# Self-Alignment VLA Implementation Plan

**创建时间**: 2025-10-14
**目标**: 实现基于自对齐的零样本视觉-语言-动作模型

## 核心思想

通过解耦两种表征实现零样本学习：
1. **Embodiment-agnostic（与具身无关）表征**: 通过大规模预训练VLA学习
2. **Embodiment-relevant（与具身相关）表征**: 编码在TTT层参数W中，通过play data自对齐

关键创新：
- TTT层的参数 (W1, b1) 就是 embodiment context W
- Alignment experts 提供自监督信号，不需要标注
- 对比学习分离两种表征

---

## 已完成工作 ✅

### 1. 基础架构搭建 (2025-10-14)

#### 1.1 修改 `PaliGemmaWithExpertModel` (gemma_pytorch.py)
- ✅ 添加 `alignment_expert_config` 参数
- ✅ 添加 `use_alignment_expert` 参数
- ✅ 实例化 `self.alignment_expert = GemmaForCausalLM(config=alignment_expert_config_hf)`
- ✅ 使用 `gemma_300m` 作为轻量级架构（18层，311M参数）
- ✅ Alignment expert 不使用 TTT（接收 Action Expert 的 TTT 输出）
- ✅ 扩展 `use_adarms` 为 `[VLM, Action Expert, Alignment Expert]`

#### 1.2 修改 `PI0Pytorch` (pi0_pytorch.py)
- ✅ 添加 `alignment_expert_config` 加载逻辑
- ✅ 修复 `use_adarms` bug (line 107)
- ✅ 传递配置到 `PaliGemmaWithExpertModel`
- ✅ 添加三个 prediction heads:
  ```python
  self.inverse_dynamics_head = nn.Linear(config.width, 32)  # -> action_dim
  self.dynamics_head = nn.Linear(config.width, config.width)  # -> obs features
  self.perception_head = nn.Linear(config.width, 32)  # -> state_dim
  ```

#### 1.3 修改配置 (pi0_config.py)
- ✅ 添加 `alignment_expert_variant: _gemma.Variant = "gemma_300m"`
- ✅ 已有 `use_alignment_expert: bool = False`

### 当前架构

```
PaliGemmaWithExpertModel:
├── self.paligemma          # VLM (gemma_2b, 18层)
├── self.gemma_expert       # Action Expert (gemma_300m, 18层 + TTT)
└── self.alignment_expert   # Alignment Expert (gemma_300m, 18层, no TTT)

PI0Pytorch:
├── self.paligemma_with_expert
├── self.action_in_proj / self.action_out_proj
└── Alignment heads (when use_alignment_expert=True):
    ├── self.inverse_dynamics_head  # (obs_t, obs_t+1) -> action_t
    ├── self.dynamics_head          # (obs_t, action_t) -> obs_t+1
    └── self.perception_head        # obs_t -> state_t
```

---

## Phase 1: 完成基础架构 [P0] (1-2天)

### 1.1 测试模型初始化 ⏳
**文件**: 创建 `tests/test_alignment_expert_init.py`

**任务**:
- [ ] 测试能否正确加载 alignment expert
- [ ] 验证三个 expert 的参数量
  - VLM: ~2B params
  - Action Expert: ~311M params + TTT
  - Alignment Expert: ~311M params
- [ ] 测试内存占用
- [ ] 验证 gradient checkpointing 兼容性

**测试代码**:
```python
config = Pi0Config(
    use_alignment_expert=True,
    alignment_expert_variant="gemma_300m",
    paligemma_variant="dummy",  # 用 dummy 快速测试
    action_expert_variant="dummy",
)
model = Pi0Pytorch(config)
print(f"Total params: {sum(p.numel() for p in model.parameters())}")
```

### 1.2 修改 PaliGemmaWithExpertModel.forward() ⏳
**文件**: `src/openpi/models_pytorch/gemma_pytorch.py`

**当前返回**: `[prefix_output, suffix_output], past_key_values`

**需要改为**:
```python
if self.alignment_expert is not None and return_alignment_hidden:
    # 运行 alignment expert forward
    alignment_hidden = self.alignment_expert.model.forward(
        inputs_embeds=suffix_embs,  # 或者从 Action Expert 获取？
        attention_mask=...,
        ...
    )
    return [prefix_output, suffix_output], past_key_values, alignment_hidden
else:
    return [prefix_output, suffix_output], past_key_values
```

**关键问题**:
- Alignment expert 的输入应该是什么？
  - 选项A: 与 Action Expert 相同的输入（suffix_embs）
  - 选项B: Action Expert 的输出（suffix_output）
  - **建议选项B**: 因为 Action Expert 的输出已经是 TTT-conditioned

### 1.3 修改 PI0Pytorch.forward() ⏳
**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

**当前代码** (line 334-391):
```python
def forward(self, observation, actions, noise=None, time=None) -> Tensor:
    # ... 预处理 ...

    # Forward through backbone
    (_, suffix_out), _ = self.paligemma_with_expert.forward(...)

    # Action prediction
    v_t = self.action_out_proj(suffix_out)
    return F.mse_loss(u_t, v_t, reduction="none")
```

**需要改为**:
```python
def forward(self, observation, actions, noise=None, time=None, obs_next=None) -> Tensor:
    # ... 预处理 ...

    # Forward through backbone
    if self.config.use_alignment_expert and self.training:
        (_, suffix_out), _, alignment_hidden = self.paligemma_with_expert.forward(
            ..., return_alignment_hidden=True
        )
    else:
        (_, suffix_out), _ = self.paligemma_with_expert.forward(...)

    # Action prediction
    v_t = self.action_out_proj(suffix_out)
    action_loss = F.mse_loss(u_t, v_t, reduction="none")

    # Alignment losses (if enabled)
    if self.config.use_alignment_expert and self.training:
        alignment_losses = self._compute_alignment_losses(
            alignment_hidden, actions, obs_next, observation.state
        )
        return action_loss, alignment_losses

    return action_loss
```

---

## Phase 2: 数据流改造 [P0] (2-3天)

### 2.1 实现 _compute_alignment_losses() 方法 ⏳
**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

```python
def _compute_alignment_losses(self, alignment_hidden, actions, obs_next, state):
    """
    计算所有 alignment expert 的损失

    Args:
        alignment_hidden: [B, L, hidden_dim] from alignment expert
        actions: [B, action_horizon, action_dim] ground truth actions
        obs_next: Observation (下一帧)
        state: [B, state_dim] proprioceptive state

    Returns:
        dict with keys: 'inverse_dynamics', 'dynamics', 'perception'
    """
    losses = {}

    # 提取特征 (使用最后一个 token)
    feat = alignment_hidden[:, -1, :]  # [B, hidden_dim]

    # Inverse Dynamics Loss
    if obs_next is not None:
        # 需要获取 obs_t+1 的 hidden states
        # TODO: 这需要再次 forward，或者在主 forward 中一次性处理
        pred_action = self.inverse_dynamics_head(feat)  # [B, action_dim]
        losses['inverse_dynamics'] = F.mse_loss(pred_action, actions[:, 0, :])

    # Dynamics Loss
    pred_obs_next = self.dynamics_head(feat)  # [B, hidden_dim]
    # TODO: 需要 obs_next 的 target features

    # Perception Loss
    pred_state = self.perception_head(feat)  # [B, state_dim]
    losses['perception'] = F.mse_loss(pred_state, state)

    return losses
```

**问题**:
- Inverse dynamics 需要 (obs_t, obs_t+1) 的联合表征
  - 当前只有 obs_t 的 hidden states
  - 需要修改架构支持处理两帧
- Dynamics 需要 obs_t+1 的 target features
  - 需要通过 vision encoder 获取

### 2.2 修改数据加载器支持连续帧 ⏳
**文件**: `src/openpi/training/data/*.py`

**需要修改**:
- 加载连续的两帧: `(obs_t, obs_t+1)`
- 确保对应的 `(action_t, state_t)` 也正确加载

**修改位置**:
- `LeRobotDataset` 或相关的 data transform
- 在 `__getitem__` 中返回额外的 `obs_next` 字段

### 2.3 创建 AlignmentLossComputer ⏳
**文件**: 创建 `src/openpi/models_pytorch/alignment_loss.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class AlignmentLossComputer(nn.Module):
    """计算 alignment expert 的所有损失"""

    def __init__(
        self,
        lambda_inverse_dynamics: float = 1.0,
        lambda_dynamics: float = 1.0,
        lambda_perception: float = 1.0,
    ):
        super().__init__()
        self.lambda_inv = lambda_inverse_dynamics
        self.lambda_dyn = lambda_dynamics
        self.lambda_per = lambda_perception

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: {
                'inverse_dynamics': [B, action_dim],
                'dynamics': [B, hidden_dim],
                'perception': [B, state_dim],
            }
            targets: {
                'action': [B, action_dim],
                'next_obs_features': [B, hidden_dim],
                'state': [B, state_dim],
            }
        """
        loss_dict = {}
        total_loss = 0.0

        # Inverse Dynamics Loss
        if 'inverse_dynamics' in predictions:
            inv_loss = F.mse_loss(
                predictions['inverse_dynamics'],
                targets['action']
            )
            loss_dict['alignment/inverse_dynamics'] = inv_loss.item()
            total_loss += self.lambda_inv * inv_loss

        # Dynamics Loss
        if 'dynamics' in predictions:
            dyn_loss = F.mse_loss(
                predictions['dynamics'],
                targets['next_obs_features']
            )
            loss_dict['alignment/dynamics'] = dyn_loss.item()
            total_loss += self.lambda_dyn * dyn_loss

        # Perception Loss
        if 'perception' in predictions:
            per_loss = F.mse_loss(
                predictions['perception'],
                targets['state']
            )
            loss_dict['alignment/perception'] = per_loss.item()
            total_loss += self.lambda_per * per_loss

        loss_dict['alignment/total'] = total_loss.item()
        return total_loss, loss_dict
```

---

## Phase 3: 对比学习 [P1] (3-5天)

### 3.1 理解 W 的提取方式 🔍
**关键问题**: W 在哪里？如何访问？

TTT 层位置: `src/openpi/models_pytorch/transformers_replace/models/gemma/ttt_with_gate.py`

TTT 参数:
```python
class TTTLinear:
    def __init__(self):
        self.W1 = nn.Parameter(...)  # [num_heads, head_dim, head_dim]
        self.b1 = nn.Parameter(...)  # [num_heads, 1, head_dim]
```

访问方式:
```python
# 从 PI0Pytorch 访问
for layer in self.paligemma_with_expert.gemma_expert.model.layers:
    if hasattr(layer, 'ttt_layer'):
        W1 = layer.ttt_layer.W1  # [num_heads, head_dim, head_dim]
        b1 = layer.ttt_layer.b1  # [num_heads, 1, head_dim]
```

### 3.2 设计对比学习策略 🎯

#### 正样本构造
**目标**: 让相同 embodiment 的 W 保持接近

**方法**: Observation appearance augmentation
- 颜色抖动 (color jitter)
- 亮度、对比度变化
- 高斯噪声
- **不改变 layout**: 不翻转、不裁剪、不旋转

**为什么**: 外观变化不改变 embodiment context（动作空间、动力学、坐标系都没变）

#### 负样本构造
**目标**: 让不同 embodiment 的 W 距离拉远

**方法**: Embodiment configuration perturbation
1. **Action space 转换**:
   - Cartesian space (x,y,z,roll,pitch,yaw) ↔ Joint space (θ1,...,θ7)
   - Delta actions ↔ Absolute actions

2. **坐标系转换**:
   - Base frame ↔ World frame
   - Camera frame ↔ Robot frame

3. **DOF 变化**:
   - 7-DOF (Franka) ↔ 6-DOF (UR5)
   - 添加/移除某些维度

**为什么**: 这些变化直接影响 action prediction 的方式，应该被编码在 W 中

### 3.3 实现 EmbodimentContrastiveLoss ⏳
**文件**: 创建 `src/openpi/models_pytorch/contrastive_loss.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbodimentContrastiveLoss(nn.Module):
    """
    对比学习 loss，用于分离 embodiment-agnostic 和 embodiment-relevant 表征

    核心思想:
    - W (TTT 参数) 应该编码 embodiment context
    - 相同 embodiment (appearance augmentation) 的 W 应该接近
    - 不同 embodiment (configuration perturbation) 的 W 应该远离
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def extract_W_embedding(self, model) -> torch.Tensor:
        """
        从 TTT 层提取 W 并 flatten 成 embedding

        Returns:
            W_emb: [B, W_dim] 如果支持 batch-level W
                   或 [W_dim] 如果是 global W
        """
        W_list = []
        for layer in model.paligemma_with_expert.gemma_expert.model.layers:
            if hasattr(layer, 'ttt_layer'):
                W1 = layer.ttt_layer.W1  # [num_heads, head_dim, head_dim]
                b1 = layer.ttt_layer.b1  # [num_heads, 1, head_dim]
                # Flatten and concatenate
                W_list.append(W1.flatten())
                W_list.append(b1.flatten())

        W_emb = torch.cat(W_list, dim=0)  # [W_dim]
        return W_emb

    def forward(
        self,
        model_anchor,
        model_positive,
        model_negative,
    ) -> torch.Tensor:
        """
        InfoNCE loss for embodiment context

        Args:
            model_anchor: 原始模型
            model_positive: 经过 appearance augmentation 的模型
            model_negative: 经过 configuration perturbation 的模型
        """
        W_anchor = self.extract_W_embedding(model_anchor)
        W_pos = self.extract_W_embedding(model_positive)
        W_neg = self.extract_W_embedding(model_negative)

        # Normalize embeddings
        W_anchor = F.normalize(W_anchor, dim=-1)
        W_pos = F.normalize(W_pos, dim=-1)
        W_neg = F.normalize(W_neg, dim=-1)

        # Compute similarities
        pos_sim = torch.sum(W_anchor * W_pos, dim=-1) / self.temperature
        neg_sim = torch.sum(W_anchor * W_neg, dim=-1) / self.temperature

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim.unsqueeze(0)], dim=0)
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits.unsqueeze(0), labels)

        return loss
```

**问题**:
- 当前 TTT 实现是否支持 batch-level 的不同 W？
  - 查看 `ttt_with_gate.py` 的实现
  - 可能需要修改为支持 per-sample W

### 3.4 实现数据增强 Pipeline ⏳
**文件**: 创建 `src/openpi/training/data/embodiment_augmentation.py`

```python
import torch
import torchvision.transforms as T

class AppearanceAugmentation:
    """外观增强（正样本）"""
    def __init__(self):
        self.transform = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def __call__(self, image):
        return self.transform(image)

class EmbodimentConfigurationAugmentation:
    """配置增强（负样本）"""
    def __init__(self):
        pass

    def transform_action_space(self, action, state, mode='cartesian_to_joint'):
        """
        转换 action space

        Args:
            action: [action_horizon, action_dim]
            state: [state_dim] current robot state
            mode: 'cartesian_to_joint' or 'joint_to_cartesian'
        """
        if mode == 'cartesian_to_joint':
            # 使用逆运动学转换
            # 需要机器人的运动学模型
            pass
        elif mode == 'joint_to_cartesian':
            # 使用正运动学转换
            pass
        return transformed_action

    def transform_frame(self, action, state, from_frame='base', to_frame='world'):
        """
        转换坐标系

        需要知道两个坐标系之间的变换矩阵
        """
        # Apply transformation matrix
        pass
        return transformed_action
```

---

## Phase 4: 训练配置 [P0] (1-2天)

### 4.1 添加 AlignmentConfig ⏳
**文件**: `src/openpi/training/config.py`

```python
@dataclasses.dataclass
class AlignmentConfig:
    """Alignment expert 训练配置"""

    # Loss weights
    lambda_inverse_dynamics: float = 1.0
    lambda_dynamics: float = 1.0
    lambda_perception: float = 1.0
    lambda_contrastive: float = 0.1

    # Contrastive learning
    use_contrastive: bool = False
    contrastive_temperature: float = 0.07
    num_negative_samples: int = 4

    # Data augmentation
    use_appearance_aug: bool = True
    use_embodiment_aug: bool = False  # 需要运动学模型支持

# 在 TrainConfig 中添加
@dataclasses.dataclass
class TrainConfig:
    # ... 现有字段 ...

    # Alignment expert config
    alignment: AlignmentConfig = dataclasses.field(default_factory=AlignmentConfig)
```

### 4.2 创建训练配置示例 ⏳
**文件**: `src/openpi/training/config.py`

添加新的 TrainConfig:
```python
TrainConfig(
    name="pi05_alignment_debug",
    model=pi0_config.Pi0Config(
        pi05=True,
        use_alignment_expert=True,
        alignment_expert_variant="gemma_300m",
        use_ttt=True,
        ttt_layer_positions="all",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    ),
    data=FakeDataConfig(),  # 先用 fake data 测试
    alignment=AlignmentConfig(
        lambda_inverse_dynamics=1.0,
        lambda_dynamics=1.0,
        lambda_perception=1.0,
        use_contrastive=False,  # 先不用对比学习
    ),
    batch_size=2,
    num_train_steps=100,
    overwrite=True,
    exp_name="alignment_debug",
    wandb_enabled=False,
)
```

### 4.3 修改训练循环 ⏳
**文件**: `src/openpi/training/train.py`

**当前代码** (需要找到具体位置):
```python
# 训练循环
loss = model.forward(batch)
loss.backward()
optimizer.step()
```

**需要改为**:
```python
if config.model.use_alignment_expert:
    # Forward with alignment
    action_loss, alignment_losses = model.forward(
        observation=batch['observation'],
        actions=batch['actions'],
        obs_next=batch['obs_next'],  # 新增
    )

    # 计算总 loss
    total_loss = action_loss.mean()

    # 添加 alignment losses
    if alignment_losses is not None:
        alignment_loss_computer = AlignmentLossComputer(
            lambda_inverse_dynamics=config.alignment.lambda_inverse_dynamics,
            lambda_dynamics=config.alignment.lambda_dynamics,
            lambda_perception=config.alignment.lambda_perception,
        )
        alignment_total, alignment_metrics = alignment_loss_computer(
            predictions=alignment_losses['predictions'],
            targets=alignment_losses['targets'],
        )
        total_loss = total_loss + alignment_total

        # Log alignment metrics
        for key, value in alignment_metrics.items():
            wandb.log({key: value})
else:
    # 原有流程
    total_loss = model.forward(batch)

total_loss.backward()
optimizer.step()
```

---

## Phase 5: 自对齐训练流程 [P0] (3-4天)

### 5.1 理解两阶段训练 🎯

#### Stage 1: 预训练（现有流程）
**目标**: 在大规模数据上学习 embodiment-agnostic 表征

**训练内容**:
- VLM: 视觉-语言对齐
- Action Expert + TTT: 学会如何利用 TTT 进行 adaptation
- Alignment Experts: 学会预测 alignment 信号

**数据**: 大规模多机器人数据集

**优化**: 所有参数

#### Stage 2: 自对齐（新流程）
**目标**: 在目标环境使用 play data 快速适应

**训练内容**:
- **只优化 W (TTT 参数)**
- 冻结所有其他参数
- 使用 alignment expert 的自监督信号

**数据**: 目标环境的 play data（无标注！）

**优化**: 只有 TTT 层的 W1, b1

### 5.2 实现 Self-Alignment 脚本 ⏳
**文件**: 创建 `src/openpi/training/self_alignment.py`

```python
"""
Self-Alignment Script

使用 play data 在新环境中对齐 embodiment context (W)
"""

import torch
from torch import nn
from torch.optim import Adam
from openpi.models_pytorch.pi0_pytorch import Pi0Pytorch
from openpi.models_pytorch.alignment_loss import AlignmentLossComputer

class SelfAlignmentTrainer:
    """自对齐训练器"""

    def __init__(
        self,
        model: Pi0Pytorch,
        learning_rate: float = 1e-4,
        num_steps: int = 1000,
    ):
        self.model = model
        self.num_steps = num_steps

        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 只解冻 TTT 参数
        ttt_params = []
        for layer in model.paligemma_with_expert.gemma_expert.model.layers:
            if hasattr(layer, 'ttt_layer'):
                for param in layer.ttt_layer.parameters():
                    param.requires_grad = True
                    ttt_params.append(param)

        print(f"Optimizing {len(ttt_params)} TTT parameters")

        # 只优化 W
        self.optimizer = Adam(ttt_params, lr=learning_rate)

        # Alignment loss computer
        self.loss_computer = AlignmentLossComputer(
            lambda_inverse_dynamics=1.0,
            lambda_dynamics=1.0,
            lambda_perception=1.0,
        )

    def alignment_step(self, batch):
        """
        单步自对齐

        Args:
            batch: play data (无标注)
                - observation: 当前观测
                - obs_next: 下一帧观测
                - action: 执行的动作（从 play data 中记录）
                - state: proprioceptive state
        """
        self.model.train()

        # Forward (只使用 alignment experts，不需要 action prediction)
        _, alignment_outputs = self.model.forward_alignment_only(
            observation=batch['observation'],
            obs_next=batch['obs_next'],
        )

        # 计算 alignment losses
        alignment_loss, metrics = self.loss_computer(
            predictions=alignment_outputs['predictions'],
            targets={
                'action': batch['action'],
                'next_obs_features': alignment_outputs['targets']['next_obs_features'],
                'state': batch['state'],
            }
        )

        # Backward (只更新 W)
        self.optimizer.zero_grad()
        alignment_loss.backward()
        self.optimizer.step()

        return metrics

    def train(self, play_dataloader):
        """
        完整的自对齐训练流程

        Args:
            play_dataloader: 目标环境的 play data
        """
        print("Starting self-alignment training...")
        print(f"Total steps: {self.num_steps}")

        step = 0
        while step < self.num_steps:
            for batch in play_dataloader:
                metrics = self.alignment_step(batch)

                if step % 10 == 0:
                    print(f"Step {step}: {metrics}")

                step += 1
                if step >= self.num_steps:
                    break

        print("Self-alignment training completed!")

    def save_adapted_model(self, path):
        """保存适应后的模型（只需要保存 W）"""
        ttt_state = {}
        for name, layer in enumerate(self.model.paligemma_with_expert.gemma_expert.model.layers):
            if hasattr(layer, 'ttt_layer'):
                ttt_state[f'layer_{name}'] = {
                    'W1': layer.ttt_layer.W1.data.clone(),
                    'b1': layer.ttt_layer.b1.data.clone(),
                }
        torch.save(ttt_state, path)
        print(f"Saved adapted TTT parameters to {path}")

# 使用示例
if __name__ == "__main__":
    # 1. 加载预训练模型
    config = Pi0Config(
        use_alignment_expert=True,
        use_ttt=True,
        # ...
    )
    model = Pi0Pytorch(config)
    model.load_state_dict(torch.load("pretrained_model.pth"))

    # 2. 准备 play data
    play_dataset = PlayDataset(data_dir="./play_data")
    play_dataloader = DataLoader(play_dataset, batch_size=8)

    # 3. 自对齐训练
    trainer = SelfAlignmentTrainer(model, learning_rate=1e-4, num_steps=1000)
    trainer.train(play_dataloader)

    # 4. 保存适应后的 W
    trainer.save_adapted_model("adapted_ttt_params.pth")
```

### 5.3 添加 forward_alignment_only() 方法 ⏳
**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

```python
def forward_alignment_only(self, observation, obs_next):
    """
    只使用 alignment experts 进行前向传播（用于自对齐）

    不需要 action labels，只使用自监督信号

    Args:
        observation: 当前观测
        obs_next: 下一帧观测

    Returns:
        alignment_outputs: {
            'predictions': {...},
            'targets': {...},
        }
    """
    # 1. 获取 obs_t 的 hidden states
    # ... (类似 forward 的处理)

    # 2. 获取 obs_t+1 的 hidden states
    # ... (需要再次 forward)

    # 3. 使用 alignment expert
    alignment_hidden_t = ...
    alignment_hidden_t1 = ...

    # 4. 预测
    predictions = {
        'inverse_dynamics': self.inverse_dynamics_head(...),
        'dynamics': self.dynamics_head(...),
        'perception': self.perception_head(...),
    }

    # 5. 构造 targets (从 play data 中获取)
    targets = {
        # 这些会在外部提供
    }

    return {'predictions': predictions, 'targets': targets}
```

---

## Phase 6: 实验验证 [P1] (3-5天)

### 6.1 消融实验设计 📊

#### 实验配置
| 实验名称 | Inverse Dyn | Dynamics | Perception | Contrastive | 说明 |
|---------|------------|----------|-----------|-------------|------|
| Baseline | ❌ | ❌ | ❌ | ❌ | 不使用 alignment |
| +InvDyn | ✅ | ❌ | ❌ | ❌ | 只用逆动力学 |
| +Dyn | ❌ | ✅ | ❌ | ❌ | 只用动力学 |
| +Percept | ❌ | ❌ | ✅ | ❌ | 只用感知对齐 |
| +All | ✅ | ✅ | ✅ | ❌ | 所有 alignment |
| +Contrast | ✅ | ✅ | ✅ | ✅ | 完整方法 |

#### 评估指标
1. **零样本成功率**: 不使用目标环境的任何演示数据
2. **少样本性能**: 使用 N 条 play trajectories 后的成功率
3. **适应速度**: 达到 X% 成功率需要的 play data 量
4. **计算效率**: 自对齐训练的时间和内存

### 6.2 迁移场景测试 🎯

#### 场景1: 相机视角变化
**设置**:
- 训练: Franka robot, 固定相机视角 A
- 测试: Franka robot, 新相机视角 B
- Play data: 机器人在视角 B 下随机运动 100 条轨迹

**期望**:
- W 应该能捕获相机外参的变化
- Perception expert 帮助理解新视角下的观测

#### 场景2: Action Space 变化
**设置**:
- 训练: Cartesian space control (x,y,z,roll,pitch,yaw)
- 测试: Joint space control (θ1,...,θ7)
- Play data: 关节空间的随机运动

**期望**:
- Inverse dynamics expert 学习新的 action space 映射
- W 编码 action space 的特性

#### 场景3: 跨机器人平台
**设置**:
- 训练: Franka Panda (7-DOF)
- 测试: UR5 (6-DOF)
- Play data: UR5 的随机运动

**期望**:
- 这是最难的场景
- 需要所有 alignment experts 配合
- 可能需要更多 play data

### 6.3 实验脚本 ⏳
**文件**: 创建 `experiments/alignment_ablation.py`

```python
"""
Alignment Expert 消融实验
"""

import torch
from openpi.models_pytorch.pi0_pytorch import Pi0Pytorch
from openpi.training.self_alignment import SelfAlignmentTrainer

def run_ablation_experiment(
    alignment_config,
    play_data_path,
    test_tasks,
):
    """
    运行单个消融实验

    Args:
        alignment_config: dict with expert switches
            {
                'use_inverse_dynamics': True/False,
                'use_dynamics': True/False,
                'use_perception': True/False,
                'use_contrastive': True/False,
            }
        play_data_path: path to play data
        test_tasks: list of test tasks
    """
    # 1. 创建模型
    config = Pi0Config(
        use_alignment_expert=True,
        # ... set based on alignment_config
    )
    model = Pi0Pytorch(config)
    model.load_pretrained("pretrained_checkpoint.pth")

    # 2. 自对齐训练
    trainer = SelfAlignmentTrainer(model)
    trainer.train(play_dataloader)

    # 3. 零样本评估
    zero_shot_results = evaluate_tasks(model, test_tasks)

    # 4. 少样本评估
    few_shot_results = {}
    for num_demos in [1, 5, 10, 20]:
        # Fine-tune with demonstrations
        finetuned_model = finetune_with_demos(model, num_demos)
        few_shot_results[num_demos] = evaluate_tasks(finetuned_model, test_tasks)

    return {
        'zero_shot': zero_shot_results,
        'few_shot': few_shot_results,
    }

if __name__ == "__main__":
    # 运行所有消融实验
    ablations = [
        {'name': 'baseline', 'use_inverse_dynamics': False, ...},
        {'name': '+inv_dyn', 'use_inverse_dynamics': True, ...},
        # ... 其他配置
    ]

    results = {}
    for ablation in ablations:
        print(f"Running ablation: {ablation['name']}")
        results[ablation['name']] = run_ablation_experiment(
            ablation,
            play_data_path="./play_data",
            test_tasks=["pick_cube", "place_cup", ...],
        )

    # 保存和可视化结果
    save_results(results, "ablation_results.json")
    plot_results(results, "ablation_plots.pdf")
```

---

## Phase 7: 文档和优化 [P2] (持续)

### 7.1 创建详细文档 📝

#### 文档列表
- [x] `self_alignment_implementation_plan.md` (本文档)
- [ ] `self_alignment_architecture.md` - 详细架构设计
- [ ] `alignment_experts_design.md` - 每个 expert 的设计细节
- [ ] `contrastive_learning_strategy.md` - 对比学习策略
- [ ] `self_alignment_tutorial.md` - 使用教程
- [ ] `experiment_results.md` - 实验结果和分析

### 7.2 性能优化 ⚡

#### 内存优化
- [ ] Alignment expert 使用 gradient checkpointing
- [ ] 考虑混合精度训练（FP16/BF16）
- [ ] 减少 alignment expert 层数（从 18 层减到 4-6 层？）

#### 计算优化
- [ ] 是否可以并行处理 obs_t 和 obs_t+1？
- [ ] 缓存重复计算的 features
- [ ] 使用 torch.compile() 加速

#### 训练速度优化
- [ ] 使用更大的 batch size（通过 gradient accumulation）
- [ ] 异步数据加载
- [ ] 多 GPU 训练支持

### 7.3 代码质量 ✨
- [ ] 添加单元测试
- [ ] 添加类型注解
- [ ] 代码 review 和重构
- [ ] 添加 docstrings

---

## 关键决策点 🔑

需要做出的重要决定：

### 1. W 的形式和管理
**问题**: W 应该如何组织？

**选项**:
- **A**: 全局单一的 W（所有 samples 共享）
  - 简单，但无法处理 batch 内的 embodiment 差异

- **B**: Per-sample W（每个 sample 独立的 W）
  - 灵活，但需要修改 TTT 实现
  - 内存占用更大

- **C**: Per-embodiment W（每个 embodiment 类型一个 W）
  - 折中方案，需要 embodiment ID

**建议**: 先用 A（全局 W），验证概念后再考虑 B

### 2. Alignment Expert 的架构深度
**问题**: Alignment expert 需要多少层？

**当前**: 18 层（gemma_300m）

**可能的优化**:
- 减少到 4-6 层
- 只需要轻量级的 prediction，不需要完整的语言理解

**建议**:
1. 先用 18 层验证功能
2. 做消融实验测试 4/6/8/18 层的性能差异
3. 创建新的 variant: `gemma_300m_4layer`

### 3. Inverse Dynamics 的输入表征
**问题**: 如何让 inverse dynamics expert 看到两帧？

**选项**:
- **A**: Concatenate hidden states from obs_t and obs_t+1
  - 需要两次 forward pass
  - 计算开销大

- **B**: 用 temporal attention 处理序列 [obs_t, obs_t+1]
  - 更优雅，但需要修改架构

- **C**: 简化为单帧预测（放弃 inverse dynamics）
  - 损失重要的自监督信号

**建议**: 先用 A，验证后考虑 B

### 4. 对比学习的优先级
**问题**: 对比学习何时实现？

**分析**:
- 对比学习理论上很重要（分离两种表征）
- 但实现复杂度高
- 可能不是 MVP 的必需项

**建议**:
1. Phase 1-2: 先不做对比学习，只用 alignment experts
2. 验证 alignment experts 本身是否有效
3. Phase 3: 再添加对比学习，看是否有提升

### 5. 数据增强的复杂度
**问题**: Embodiment configuration augmentation 需要运动学模型吗？

**分析**:
- 理想情况：需要准确的运动学模型做坐标转换
- 实际情况：可能没有所有机器人的运动学模型

**替代方案**:
- 简单的 linear transformation
- 使用历史数据做经验性的映射
- 先不做 negative samples，只用 positive samples

**建议**: 先用简单的 linear transformation 验证概念

---

## 时间线和里程碑 📅

### Week 1: 基础架构 (Phase 1-2)
- [ ] Day 1-2: 测试初始化，修改 forward 流程
- [ ] Day 3-4: 数据加载器改造，支持连续帧
- [ ] Day 5: 实现 AlignmentLossComputer
- [ ] Day 6-7: 端到端测试，确保能正确训练

**里程碑**: 能够训练一个带 alignment experts 的模型

### Week 2: 训练配置和自对齐 (Phase 4-5)
- [ ] Day 8-9: 添加训练配置，修改训练循环
- [ ] Day 10-11: 实现 SelfAlignmentTrainer
- [ ] Day 12-13: 测试自对齐流程
- [ ] Day 14: Debug 和优化

**里程碑**: 能够使用 play data 进行自对齐

### Week 3: 对比学习和实验 (Phase 3, 6)
- [ ] Day 15-17: 实现对比学习（如果需要）
- [ ] Day 18-19: 运行消融实验
- [ ] Day 20-21: 零样本迁移测试

**里程碑**: 有初步的实验结果

### Week 4: 优化和文档 (Phase 7)
- [ ] Day 22-24: 性能优化和 bug 修复
- [ ] Day 25-27: 完善文档
- [ ] Day 28: 最终 review

**里程碑**: 可发布的完整实现

---

## 风险和挑战 ⚠️

### 技术风险

1. **TTT 层可能不支持 per-sample W**
   - 当前实现可能是全局 W
   - 需要修改底层实现
   - **缓解**: 先用全局 W 验证概念

2. **Alignment expert 效果可能不明显**
   - 自监督信号可能不够强
   - W 可能无法有效编码 embodiment
   - **缓解**: 做充分的消融实验

3. **内存和计算开销**
   - 多了一个 300M 参数的 expert
   - 需要处理连续两帧
   - **缓解**: 使用 gradient checkpointing，减少层数

4. **数据加载器改造复杂**
   - 现有 dataloader 可能不支持连续帧
   - 需要确保时序对齐
   - **缓解**: 逐步修改，充分测试

### 实验风险

1. **零样本性能可能很差**
   - Domain gap 可能太大
   - Play data 可能不够
   - **缓解**: 降低期望，先测试简单场景

2. **对比学习可能难以实现**
   - 负样本构造困难
   - 需要运动学模型
   - **缓解**: 暂时不做对比学习

3. **实验时间可能很长**
   - 需要训练多个 ablation
   - 需要收集多个环境的 play data
   - **缓解**: 先用小规模 debug 配置

---

## 下一步行动 🚀

### 立即开始 (本周)
1. ✅ 完成架构设计和计划文档
2. ⏳ 测试模型初始化 (`test_alignment_expert_init.py`)
3. ⏳ 修改 `PaliGemmaWithExpertModel.forward()`
4. ⏳ 修改 `PI0Pytorch.forward()`

### 短期目标 (Week 1)
- 完成 Phase 1-2: 基础架构和数据流
- 能够训练一个端到端的模型

### 中期目标 (Week 2-3)
- 完成 Phase 4-5: 训练配置和自对齐
- 有初步的实验结果

### 长期目标 (Week 4+)
- 完整的消融实验
- 零样本迁移验证
- 发布和文档

---

## 参考资料 📚

### 相关论文
1. **TTT Layers**: [Test-Time Training](https://arxiv.org/abs/...)
2. **Cross-Embodiment Transfer**: [Latent Space Alignment](https://arxiv.org/abs/...)
3. **RICL**: [Robot In-Context Learning](http://arxiv.org/abs/2508.02062)
4. **Scaling Proprioceptive-Visual Learning**: [Heterogeneous Pre-trained Transformers](https://...)

### 代码参考
- TTT Implementation: `src/openpi/models_pytorch/transformers_replace/models/gemma/ttt_with_gate.py`
- PI0 Model: `src/openpi/models_pytorch/pi0_pytorch.py`
- Training Loop: `src/openpi/training/train.py`

### 已有文档
- `docs/analysis/gradient_checkpointing_analysis.md`
- `docs/analysis/attention_mask_analysis.md`
- `docs/analysis/TTT_Action_Expert_Integration_Plan.md`
- `docs/analysis/ttt_video_dit_comparison.md`

---

## 更新日志 📝

### 2025-10-14
- ✅ 创建初始计划文档
- ✅ 完成架构设计
- ✅ 添加 alignment expert 到代码
- ✅ 添加配置选项
- ✅ 规划详细的实现路径

---

**最后更新**: 2025-10-14
**状态**: Phase 1 进行中
**下一里程碑**: 完成基础架构测试
