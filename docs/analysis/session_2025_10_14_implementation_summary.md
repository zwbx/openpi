# Alignment Expert 实现总结

**会话日期**: 2025-10-14
**完成状态**: 核心实现完成 ✅

---

## 本次会话完成的工作

本次会话完成了 Alignment Expert 的**核心实现**，包括架构调整、TTT参数共享、embedding方法、attention mask优化以及完整的forward和loss计算流程。

---

## 1. 架构理解与设计决策

### 1.1 关键架构决策

**TTT 参数共享方式**：
- ❌ **最初理解错误**: 以为 Alignment Expert 不使用 TTT，而是接收 Action Expert 的输出
- ✅ **正确理解**: Alignment Expert 也有 TTT 层，且与 Action Expert **共享同一套 TTT 参数** (W1, b1)
- ✅ **实现方式**: 使用 **Singleton 模式**，基于 `layer_idx` 来确保相同层返回同一个实例

**Alignment Expert 输入**：
- ❌ **最初理解错误**: Alignment Expert 的输入是 Action Expert 的输出
- ✅ **正确理解**: Alignment Expert 的输入与 Action Expert 一样，都是 VLM 的 `prefix_output`
- ✅ **架构**: `VLM prefix → Action Expert` 和 `VLM prefix → Alignment Expert` 是并行的

**三个 Alignment 任务的组织**：
- ✅ **设计决策**: 三个任务（Perception, Dynamics, Inverse Dynamics）的输入 **concat 到一个 suffix** 中
- ✅ **好处**: 一次 forward 完成三个任务，效率高，任务间可以共享信息
- ✅ **Attention 结构**: Perception → Dynamics 和 Inverse Dynamics 都可以 attend，但 Dynamics ⊥ Inverse Dynamics（block diagonal）

**训练 vs 推理模式**：
- **训练**: `inputs_embeds = [prefix, action_suffix, alignment_suffix]` → 三者联合 attention
- **推理**: 分别调用（节省计算）
  - VLM only: `[prefix, None, None]`
  - Action Expert: `[prefix, action_suffix, None]`
  - Alignment Expert: `[prefix, None, alignment_suffix]`

---

## 2. TTT 参数共享实现

### 2.1 Singleton 模式

**文件**: `src/openpi/models_pytorch/transformers_replace/models/gemma/ttt_with_gate.py`

**核心思想**：
- 在 `TTTWithAdaptiveNorm` 类中实现 `__new__()` 方法
- 使用类变量 `_instances = {layer_idx: instance}` 存储已创建的实例
- 当创建新实例时，如果 `layer_idx` 已存在，直接返回现有实例

**实现代码**：
```python
class TTTWithAdaptiveNorm(nn.Module):
    # Class variable to store instances by layer_idx
    _instances = {}

    def __new__(cls, *args, layer_idx: int = -1, **kwargs):
        if layer_idx >= 0 and layer_idx in cls._instances:
            # Reuse existing instance for this layer
            existing_instance = cls._instances[layer_idx]
            print(f"[TTT Singleton] Layer {layer_idx}: Reusing existing instance (parameters shared)")
            return existing_instance
        else:
            # Create new instance
            instance = super(TTTWithAdaptiveNorm, cls).__new__(cls)
            if layer_idx >= 0:
                cls._instances[layer_idx] = instance
                print(f"[TTT Singleton] Layer {layer_idx}: Creating new instance")
            return instance

    def __init__(self, ..., layer_idx: int = -1):
        # Skip initialization if this instance was already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return

        super().__init__()
        # ... initialize W1, b1, etc. ...
        self._initialized = True
```

**效果**：
- Action Expert 和 Alignment Expert 的同一层 TTT 使用**完全相同的对象**
- `action_expert.layers[0].ttt_layer is alignment_expert.layers[0].ttt_layer` → `True`
- 梯度自动累加：`W1.grad = action_loss 的梯度 + alignment_loss 的梯度`

---

## 3. Alignment Suffix Embedding

### 3.1 embed_alignment_suffix() 方法

**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

**功能**: 将三个 alignment 任务的输入 embed 并 concat 成一个 suffix

**输入**：
```python
def embed_alignment_suffix(
    self,
    noisy_state,              # [batch, state_dim] - Perception task
    actions,                  # [batch, action_h, action_dim] - Dynamics task (clean)
    noisy_next_obs_features,  # [batch, feature_dim or seq_len, feature_dim] - Dynamics
    next_obs_embedding,       # [batch, feature_dim or seq_len, feature_dim] - Inv Dyn (clean)
    noisy_actions,            # [batch, action_h, action_dim] - Inverse Dynamics task
    timestep,                 # [batch] - 共享的扩散 timestep
):
```

**Suffix 结构**：
```
[perception_token] [actions...] [next_obs_features...] [next_obs_emb...] [noisy_actions...]
      (1 token)     (action_h)     (variable length)      (variable length)    (action_h)

   Perception          Dynamics                          Inverse Dynamics
```

**Attention Masks**：
```python
att_masks = [
    1,  # Perception: 新 block
    1,  # Dynamics actions: 新 block
    0,  # (actions 内部 causal)
    0,  # next_obs_features: 继续 Dynamics block
    1,  # Inverse Dynamics next_obs_emb: 新 block
    0,  # (next_obs_emb 内部 causal)
    0,  # noisy_actions: 继续 Inverse Dynamics block
]
```

**Block Diagonal Ranges**：
```python
block_diagonal_ranges = [
    (dynamics_start, dynamics_end),
    (inverse_dynamics_start, inverse_dynamics_end)
]
# 这两个 range 不能互相 attend（因为包含对方的标签）
```

---

## 4. Attention Mask 增强

### 4.1 make_att_2d_masks() 扩展

**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

**新增功能**: 支持 block diagonal masking

**修改**：
```python
def make_att_2d_masks(pad_masks, att_masks, block_diagonal_ranges=None):
    # ... 原有逻辑 ...

    result_masks = att_2d_masks & pad_2d_masks

    # Apply block diagonal masking if specified
    if block_diagonal_ranges is not None:
        for i, (start_i, end_i) in enumerate(block_diagonal_ranges):
            for j, (start_j, end_j) in enumerate(block_diagonal_ranges):
                if i != j:
                    # Block i cannot attend to block j (and vice versa)
                    result_masks[:, start_i:end_i, start_j:end_j] = False

    return result_masks
```

**作用**: 确保 Dynamics 和 Inverse Dynamics 任务不能互相看到对方的输入（避免标签泄露）

---

## 5. PaliGemmaWithExpertModel.forward() 扩展

### 5.1 支持三个 Experts

**文件**: `src/openpi/models_pytorch/gemma_pytorch.py`

**核心修改**：

1. **adarms_cond 扩展为 3 个元素**：
```python
if adarms_cond is None:
    adarms_cond = [None, None, None]  # [VLM, Action Expert, Alignment Expert]
```

2. **models 列表动态构建**：
```python
# Build models list: VLM + Action Expert + (optional) Alignment Expert
models = [self.paligemma.language_model, self.gemma_expert.model]
if self.alignment_expert is not None and len(inputs_embeds) > 2 and inputs_embeds[2] is not None:
    models.append(self.alignment_expert.model)
```

3. **compute_layer_complete 使用外部 models**：
```python
def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
    # Use models from outer scope (can be 2 or 3 experts)
    # 删除了内部的 models 重新定义
    ...
```

4. **返回值支持 3 个 outputs**：
```python
prefix_output = outputs_embeds[0]
suffix_output = outputs_embeds[1]
alignment_output = outputs_embeds[2] if len(outputs_embeds) > 2 else None

return_outputs = [prefix_output, suffix_output]
if alignment_output is not None:
    return_outputs.append(alignment_output)

return return_outputs, prefix_past_key_values
```

---

## 6. PI0Pytorch.forward() 完整实现

### 6.1 准备扩散输入

**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

**代码结构**：
```python
def forward(self, observation, actions, noise=None, time=None, next_obs_features=None):
    # 1. Action Expert: 准备扩散输入
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    # 2. Alignment Expert: 准备三个任务的扩散输入
    if use_alignment_expert and next_obs_features is not None:
        time_expanded_state = time[:, None]

        # Task 1: Perception (noisy_state -> state)
        noise_perception = self.sample_noise(state.shape, state.device)
        state_t = time_expanded_state * noise_perception + (1 - time_expanded_state) * state
        u_t_perception = noise_perception - state

        # Task 2: Dynamics (action + noisy_next_obs -> next_obs)
        noise_dynamics = self.sample_noise(next_obs_features.shape, next_obs_features.device)
        next_obs_t = time_expanded_state * noise_dynamics + (1 - time_expanded_state) * next_obs_features
        u_t_dynamics = noise_dynamics - next_obs_features

        # Task 3: Inverse Dynamics (next_obs + noisy_action -> action)
        noise_inv_dynamics = self.sample_noise(actions.shape, actions.device)
        actions_t_inv = time_expanded * noise_inv_dynamics + (1 - time_expanded) * actions
        u_t_inv_dynamics = noise_inv_dynamics - actions
```

**关键点**：
- 所有任务共享**同一个 timestep** `time`
- 每个任务的 **noise 独立采样**
- 使用 **velocity prediction** 框架：`u_t = noise - clean_target`

### 6.2 Embedding

```python
# Embed prefix (VLM: images + language)
prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)

# Embed action suffix (Action Expert)
suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

# Embed alignment suffix (Alignment Expert) if enabled
if use_alignment_expert and next_obs_features is not None:
    alignment_suffix_embs, alignment_pad_masks, alignment_att_masks, alignment_adarms_cond, block_diagonal_ranges = \
        self.embed_alignment_suffix(
            noisy_state=state_t,
            actions=actions,  # clean actions
            noisy_next_obs_features=next_obs_t,
            next_obs_embedding=next_obs_features,  # clean next_obs
            noisy_actions=actions_t_inv,
            timestep=time,
        )
```

### 6.3 联合 Forward

```python
def forward_func(prefix_embs, suffix_embs, alignment_suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, alignment_adarms_cond):
    # Prepare inputs_embeds
    if alignment_suffix_embs is not None:
        inputs_embeds = [prefix_embs, suffix_embs, alignment_suffix_embs]
        adarms_cond_list = [None, adarms_cond, alignment_adarms_cond]
    else:
        inputs_embeds = [prefix_embs, suffix_embs]
        adarms_cond_list = [None, adarms_cond]

    outputs, _ = self.paligemma_with_expert.forward(
        inputs_embeds=inputs_embeds,
        adarms_cond=adarms_cond_list,
        ...
    )

    # Return action output and alignment output (if exists)
    if len(outputs) > 2:
        return outputs[1], outputs[2]
    else:
        return outputs[1], None

action_out, alignment_out = self._apply_checkpoint(forward_func, ...)
```

### 6.4 Loss 计算

```python
# Action Expert loss
v_t = self.action_out_proj(action_out[:, -action_horizon:])
action_loss = F.mse_loss(u_t, v_t, reduction="none")

# Alignment losses (if enabled)
if alignment_out is not None:
    alignment_out = alignment_out.to(dtype=torch.float32)

    # Task 1: Perception
    perception_hidden = alignment_out[:, 0]
    v_t_perception = self.perception_head(perception_hidden)
    perception_loss = F.mse_loss(v_t_perception, u_t_perception, reduction="none")

    # Task 2: Dynamics
    dynamics_start = 1
    dynamics_end = 1 + action_horizon + next_obs_seq_len
    dynamics_hidden = alignment_out[:, dynamics_start:dynamics_end].mean(dim=1)
    v_t_dynamics = self.dynamics_head(dynamics_hidden)
    dynamics_loss = F.mse_loss(v_t_dynamics, u_t_dynamics, reduction="none")

    # Task 3: Inverse Dynamics
    inv_dynamics_hidden = alignment_out[:, -action_horizon:]
    v_t_inv_dynamics = self.inverse_dynamics_head(inv_dynamics_hidden)
    inverse_dynamics_loss = F.mse_loss(v_t_inv_dynamics, u_t_inv_dynamics, reduction="none")

    # Return both losses
    alignment_losses = {
        'perception_loss': perception_loss,
        'dynamics_loss': dynamics_loss,
        'inverse_dynamics_loss': inverse_dynamics_loss,
    }
    return action_loss, alignment_losses

return action_loss
```

---

## 7. Loss 计算架构文档

**文件**: `docs/analysis/alignment_loss_architecture.md`

已创建详细的文档，包含：
1. Action Expert 的 velocity prediction 框架
2. 三个 Alignment 任务的详细 loss 计算
3. 为什么所有任务都预测 velocity
4. 为什么三个任务要合并
5. 为什么 Dynamics 和 Inverse Dynamics 不能互相 attend
6. 完整的代码示例

---

## 8. 文件修改清单

### 8.1 核心模型文件

1. **`src/openpi/models_pytorch/transformers_replace/models/gemma/ttt_with_gate.py`**
   - ✅ 实现 Singleton 模式（`__new__()` 方法）
   - ✅ 添加 `_instances` 类变量
   - ✅ 添加 `_initialized` 标志

2. **`src/openpi/models_pytorch/gemma_pytorch.py`**
   - ✅ 扩展 `adarms_cond` 为 3 个元素
   - ✅ 动态构建 `models` 列表（支持 2 或 3 个 experts）
   - ✅ 修改 `compute_layer_complete` 使用外部 `models`
   - ✅ 返回值支持 3 个 outputs

3. **`src/openpi/models_pytorch/pi0_pytorch.py`**
   - ✅ 添加 `embed_alignment_suffix()` 方法
   - ✅ 修改 `make_att_2d_masks()` 支持 block diagonal
   - ✅ 修改 `forward()` 方法：
     - 准备三个任务的扩散输入
     - 调用 alignment expert
     - 计算三个 alignment losses
   - ✅ 添加 `next_obs_features` 参数到 `forward()`

### 8.2 配置文件

4. **`src/openpi/models/pi0_config.py`**
   - ✅ 已有 `use_alignment_expert: bool = False`
   - ✅ 已有 `alignment_expert_variant: _gemma.Variant = "gemma_300m"`

### 8.3 文档文件

5. **`docs/analysis/alignment_loss_architecture.md`** (新创建)
   - ✅ 详细的 Loss 计算架构文档

6. **`docs/analysis/self_alignment_implementation_plan.md`** (已存在，待更新)
   - ⏳ 需要更新 "已完成工作" 章节

---

## 9. 待完成的工作

### 9.1 高优先级 (P0)

1. **⚠️⚠️ ATTENTION MASK DESIGN 需要验证** ⚠️⚠️
   - **问题**: Alignment suffix 的 attention mask 设计可能存在错误
   - **当前设计** (在 `embed_alignment_suffix()` line 442-530):
     ```
     Token 结构:
     [perception(1)] [actions(action_h)] [next_obs(n)] [next_obs_emb(m)] [noisy_actions(action_h)]
          ↓                  ↓                  ↓               ↓                     ↓
     att_masks:
          [1]         [1, 0, ..., 0]      [0, 0, ...]      [1, 0, ..., 0]        [0, 0, ...]
          ↑               ↑                    ↑                ↑                     ↑
        新block         新block            继续dynamics      新block            继续inv_dyn

     Perception:       可被所有任务 attend
     Dynamics:         perception + actions + next_obs (内部causal)
     Inverse Dynamics: perception + next_obs_emb + noisy_actions (内部causal)

     Block Diagonal: Dynamics ⊥ Inverse Dynamics
     ```

   - **疑问点** (需要用户确认):
     1. ✅ Perception 应该可以被所有任务 attend（当前设计正确）
     2. ❓ **Dynamics 内部**: actions 的每个 token 能否 attend 到 next_obs 的 tokens？
        - 当前: `att_masks = [1] + [0]*(action_h-1) + [0]*next_obs_seq_len`
        - 含义: action tokens 之间 causal，然后 next_obs tokens 继续同一个 block
        - **问题**: 这意味着 action_t 能 attend 到 action_{0:t}，next_obs tokens 能 attend 到所有 actions 和之前的 next_obs tokens
        - **是否正确**？还是应该让 actions 和 next_obs 分开成两个 causal block？

     3. ❓ **Inverse Dynamics 内部**: next_obs_emb 的每个 token 能否 attend 到 noisy_actions 的 tokens？
        - 当前: `att_masks = [1] + [0]*(next_obs_emb_seq_len-1) + [0]*action_h`
        - 含义: next_obs_emb tokens 之间 causal，然后 noisy_action tokens 继续同一个 block
        - **问题**: 这意味着 noisy_action_t 能 attend 到所有 next_obs_emb tokens 和之前的 noisy_actions
        - **是否正确**？还是应该让它们分开？

     4. ❓ **Dynamics 的 clean actions**: 是否应该对 Dynamics 任务本身可见？
        - 当前设计: 输入的 actions 是 clean 的（非 noisy），用于预测 next_obs
        - **问题**: 这样 Dynamics 任务就能直接看到 ground truth action，是否合理？

   - **建议的修复方案** (待用户确认):
     - 方案 A: 保持当前设计（所有内部 tokens 都 causal）
     - 方案 B: 让每个任务的输入和输出分开成两个 causal blocks
     - 方案 C: 完全重新设计 attention structure

   - **STATUS**: ✅ Block diagonal masking 已应用于 forward() (line 645)，但基础 att_masks 设计需要验证

2. **处理可变长度的 next_obs_features**
   - 当前假设 `next_obs_seq_len = 1`
   - 实际应该从 `alignment_suffix_embs` 的结构中动态计算

3. **训练循环集成**
   - 训练脚本需要处理 `(action_loss, alignment_losses)` 的返回值
   - 计算 total_loss: `action_loss.mean() + λ * (perception_loss + dynamics_loss + inverse_dynamics_loss)`

4. **数据加载器修改**
   - 需要支持加载 `next_obs_features`（下一帧的观察特征）
   - 可能需要通过 vision encoder（如 DINO）预处理下一帧

### 9.2 中优先级 (P1)

5. **验证 TTT 参数共享**
   - 添加测试代码验证 singleton 模式是否正确工作
   - 检查 `id(action_expert.layers[0].ttt_layer) == id(alignment_expert.layers[0].ttt_layer)`

6. **梯度流测试**
   - 验证两个 expert 的梯度是否正确累加到共享的 W 上

7. **内存和速度优化**
   - Alignment expert 可能需要减少层数（从 18 层到 4-6 层）
   - 使用 gradient checkpointing

### 9.3 低优先级 (P2)

8. **对比学习** (Phase 3)
   - 当前未实现
   - 可能在基础 alignment 验证后再添加

9. **自对齐训练脚本** (Phase 5)
   - 创建 `self_alignment.py`
   - 实现只优化 W 的训练流程

10. **消融实验** (Phase 6)
    - 测试每个 alignment task 的贡献
    - 零样本迁移实验

---

## 10. 代码架构总结

### 10.1 训练时的 Forward 流程

```
Input: observation, actions, next_obs_features

1. 准备扩散输入
   ├─ Action Expert: (noise_action, x_t_action, u_t_action)
   └─ Alignment Expert:
       ├─ Perception: (noise_perception, state_t, u_t_perception)
       ├─ Dynamics: (noise_dynamics, next_obs_t, u_t_dynamics)
       └─ Inverse Dynamics: (noise_inv_dynamics, actions_t_inv, u_t_inv_dynamics)

2. Embedding
   ├─ prefix_embs = embed_prefix(images, lang_tokens)
   ├─ action_suffix_embs = embed_suffix(state, x_t_action, time)
   └─ alignment_suffix_embs = embed_alignment_suffix(
       state_t, actions, next_obs_t, next_obs_features, actions_t_inv, time
     )

3. 联合 Forward (三个 experts 一起)
   inputs_embeds = [prefix_embs, action_suffix_embs, alignment_suffix_embs]
   [prefix_out, action_out, alignment_out] = model.forward(inputs_embeds)

4. 计算 Losses
   ├─ v_t_action = action_out_proj(action_out)
   │  action_loss = MSE(v_t_action, u_t_action)
   │
   └─ From alignment_out:
       ├─ v_t_perception = perception_head(alignment_out[:, 0])
       │  perception_loss = MSE(v_t_perception, u_t_perception)
       │
       ├─ v_t_dynamics = dynamics_head(alignment_out[:, dynamics_range].mean(1))
       │  dynamics_loss = MSE(v_t_dynamics, u_t_dynamics)
       │
       └─ v_t_inv_dynamics = inverse_dynamics_head(alignment_out[:, -action_h:])
          inverse_dynamics_loss = MSE(v_t_inv_dynamics, u_t_inv_dynamics)

5. 返回
   return action_loss, {
       'perception_loss': perception_loss,
       'dynamics_loss': dynamics_loss,
       'inverse_dynamics_loss': inverse_dynamics_loss
   }
```

### 10.2 推理时的 Forward 流程

```
# Only Action Expert
inputs_embeds = [prefix_embs, action_suffix_embs, None]
[prefix_out, action_out] = model.forward(inputs_embeds)

# Only Alignment Expert (for self-alignment)
inputs_embeds = [prefix_embs, None, alignment_suffix_embs]
[prefix_out, alignment_out] = model.forward(inputs_embeds)
```

---

## 11. 关键技术要点

### 11.1 为什么使用 Velocity Prediction？

- **一致性**: Action Expert 和 Alignment Experts 使用相同的框架
- **理论依据**: Flow Matching 比传统 DDPM 更稳定
- **目标**: 预测 `v_t = noise - clean_target`，而不是直接预测 `clean_target`

### 11.2 为什么 TTT 参数要共享？

- **Embodiment Context**: TTT 的参数 W 编码机器人的身体特性（动作空间、坐标系、自由度等）
- **自对齐**: 通过 alignment loss 优化 W，使 W 更好地适应新环境
- **迁移**: 优化后的 W 反过来提升 Action Expert 的性能

### 11.3 为什么 Dynamics 和 Inverse Dynamics 不能互相 attend？

- **标签泄露**: Dynamics 的输入包含 `clean_action`，Inverse Dynamics 的输入包含 `clean_next_obs`
- **自监督**: 如果互相 attend，模型可能直接复制标签，失去自监督的意义
- **Block Diagonal**: 通过 mask 确保两个任务不能看到对方的输入

---

## 12. 验证清单

### 12.1 功能验证

- [ ] 模型初始化不报错
- [ ] Forward 可以正常运行（无 alignment）
- [ ] Forward 可以正常运行（有 alignment）
- [ ] TTT 参数确实共享（通过 `id()` 检查）
- [ ] Alignment losses 计算正确
- [ ] Gradient 正确反向传播

### 12.2 训练验证

- [ ] 可以正常训练（只有 action loss）
- [ ] 可以正常训练（有 alignment losses）
- [ ] Loss 值合理（不是 NaN 或 Inf）
- [ ] 训练速度可接受
- [ ] 内存占用可接受

### 12.3 自对齐验证

- [ ] 可以只优化 W（冻结其他参数）
- [ ] 使用 play data 可以降低 alignment losses
- [ ] 优化后的 W 提升 action prediction 性能

---

## 13. 下一步行动

### 13.1 立即修复 (Today)

1. **修复 block diagonal masking 应用**
   - 在 `forward()` 的 `make_att_2d_masks()` 调用处传入 `block_diagonal_ranges`

2. **修复 `next_obs_seq_len` 硬编码**
   - 从 `alignment_suffix_embs` 结构中动态计算

### 13.2 本周完成 (This Week)

3. **集成到训练循环**
   - 修改训练脚本处理 alignment losses
   - 添加 loss weights 配置

4. **数据加载器修改**
   - 支持加载 `next_obs_features`

5. **基础测试**
   - 测试模型初始化
   - 测试 forward pass
   - 测试 TTT 参数共享

### 13.3 下周完成 (Next Week)

6. **端到端训练测试**
   - 使用 fake data 测试完整训练流程
   - 验证 losses 收敛

7. **自对齐脚本**
   - 实现 `SelfAlignmentTrainer`
   - 测试只优化 W 的流程

---

**最后更新**: 2025-10-14
**状态**: 核心实现完成，待集成和测试
**下一里程碑**: 完成训练循环集成和基础测试
