# Alignment Expert Loss 计算架构

**创建时间**: 2025-10-14
**目的**: 详细记录 Alignment Expert 的 Loss 计算方式，与 Action Expert 保持统一的扩散模型框架

---

## 核心原则：统一的扩散模型框架

Action Expert 和所有 Alignment Experts 都使用 **Flow Matching / Rectified Flow** 框架：

- **不预测 clean target**，而是预测 **velocity** `v_t`
- **velocity** 定义为：`v_t = noise - clean_target`
- **timestep** 通过 **adaRMS** 注入，不直接 concat 到输入
- **Loss** 使用 MSE 匹配预测的 velocity 和真实 velocity

---

## 1. Action Expert Loss（已实现）

### 1.1 准备扩散输入

```python
# Input: clean actions [batch_size, action_horizon, action_dim]
time = sample_time(batch_size)  # [batch_size], 采样自 Beta(1.5, 1.0)
noise = sample_noise(actions.shape)  # [batch_size, action_horizon, action_dim]

# 在时刻 t 的 noisy action
time_expanded = time[:, None, None]  # [batch_size, 1, 1]
x_t = time_expanded * noise + (1 - time_expanded) * actions  # 线性插值

# 目标：velocity (not clean action!)
u_t = noise - actions  # [batch_size, action_horizon, action_dim]
```

### 1.2 模型 Forward

```python
# Embed inputs
prefix_embs = embed_prefix(images, lang_tokens)  # VLM: [batch, prefix_len, hidden_dim]
suffix_embs = embed_suffix(state, x_t, time)     # Action Expert: [batch, suffix_len, hidden_dim]

# Forward through model
(_, suffix_out), _ = paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs, suffix_embs],
    adarms_cond=[None, adarms_cond],  # timestep 通过 adaRMS 注入
)
```

`adarms_cond` 是如何生成的？

```python
# In embed_suffix():
time_emb = create_sinusoidal_pos_embedding(time, hidden_dim, ...)  # [batch, hidden_dim]
adarms_cond = time_mlp(time_emb)  # [batch, hidden_dim]
# adarms_cond 被传入每一层的 LayerNorm，调制 scale 和 shift
```

### 1.3 预测 Velocity

```python
# 提取最后 action_horizon 个 tokens
suffix_out = suffix_out[:, -action_horizon:]  # [batch, action_horizon, hidden_dim]

# 投影到 action 空间
v_t = action_out_proj(suffix_out)  # [batch, action_horizon, action_dim]
```

### 1.4 计算 MSE Loss

```python
action_loss = F.mse_loss(v_t, u_t, reduction="none")  # [batch, action_horizon, action_dim]
# 通常在训练循环中会 .mean() 得到 scalar loss
```

---

## 2. Alignment Expert Loss（待实现）

### 核心思想

Alignment Expert 包含三个任务：
1. **Perception**: 从 noisy state 预测 clean state
2. **Dynamics**: 从 action + noisy next_obs 预测 clean next_obs features
3. **Inverse Dynamics**: 从 next_obs + noisy action 预测 clean action

**所有三个任务都使用相同的扩散框架！**

### 2.1 Task 1: Perception（状态感知）

#### 2.1.1 准备扩散输入

```python
# Input: clean state [batch_size, state_dim]
time = sample_time(batch_size)  # [batch_size]，和 Action Expert 共享同一个 time!
noise_perception = sample_noise(state.shape)  # [batch_size, state_dim]

# 在时刻 t 的 noisy state
time_expanded = time[:, None]  # [batch_size, 1]
state_t = time_expanded * noise_perception + (1 - time_expanded) * state

# 目标：velocity
u_t_perception = noise_perception - state  # [batch_size, state_dim]
```

#### 2.1.2 Embed Alignment Suffix

```python
# 三个任务的 suffix 合并在一起
alignment_suffix_embs, pad_masks, att_masks, adarms_cond, block_diagonal_ranges = \
    embed_alignment_suffix(
        noisy_state=state_t,                  # Perception 输入
        actions=actions,                      # Dynamics 输入 (part 1)
        noisy_next_obs_features=next_obs_t,   # Dynamics 输入 (part 2)
        next_obs_embedding=next_obs_emb,      # Inverse Dynamics 输入 (part 1)
        noisy_actions=actions_t,              # Inverse Dynamics 输入 (part 2)
        timestep=time,                        # 共享的 timestep
    )

# Suffix 结构：
# [perception_token] [action_tokens...] [next_obs_tokens...] [next_obs_emb_tokens...] [noisy_action_tokens...]
#       (1)              (action_h)         (next_obs_len)         (next_obs_emb_len)        (action_h)
```

#### 2.1.3 Forward 通过 Alignment Expert

```python
# Alignment Expert forward
(_, alignment_out), _ = alignment_expert.forward(
    inputs_embeds=[prefix_embs, alignment_suffix_embs],
    attention_mask=alignment_att_2d_masks_4d,  # 包含 block diagonal 约束
    position_ids=alignment_position_ids,
    adarms_cond=[None, adarms_cond],  # 和 Action Expert 相同的注入方式
)
```

#### 2.1.4 提取 Perception Token 并预测

```python
# Perception 是第 0 个 token
perception_hidden = alignment_out[:, 0]  # [batch_size, hidden_dim]

# 预测 velocity
v_t_perception = perception_head(perception_hidden)  # [batch_size, state_dim]
```

#### 2.1.5 计算 MSE Loss

```python
perception_loss = F.mse_loss(v_t_perception, u_t_perception, reduction="none")
# [batch_size, state_dim]
```

---

### 2.2 Task 2: Dynamics（动态预测）

#### 2.2.1 准备扩散输入

```python
# Input: clean next_obs_features [batch_size, feature_dim]
# (feature_dim 可能是 DINO 特征维度，或其他视觉特征)
noise_obs = sample_noise(next_obs_features.shape)  # [batch_size, feature_dim]

# 在时刻 t 的 noisy next_obs
next_obs_t = time_expanded * noise_obs + (1 - time_expanded) * next_obs_features

# 目标：velocity
u_t_dynamics = noise_obs - next_obs_features  # [batch_size, feature_dim]
```

注意：`actions` 是 clean actions（不加噪声），因为 Dynamics 任务是：**给定 clean action 和 noisy obs，预测 clean obs**

#### 2.2.2 提取 Dynamics Tokens 并预测

```python
# Dynamics tokens 范围：
# [perception_end : perception_end + action_horizon + next_obs_seq_len]
dynamics_start = 1  # perception 占 1 个 token
dynamics_end = dynamics_start + action_horizon + next_obs_seq_len

dynamics_hidden = alignment_out[:, dynamics_start:dynamics_end]
# [batch_size, action_horizon + next_obs_seq_len, hidden_dim]

# Pool over sequence（或者只取最后一个 token）
dynamics_pooled = dynamics_hidden.mean(dim=1)  # [batch_size, hidden_dim]

# 预测 velocity
v_t_dynamics = dynamics_head(dynamics_pooled)  # [batch_size, feature_dim]
```

#### 2.2.3 计算 MSE Loss

```python
dynamics_loss = F.mse_loss(v_t_dynamics, u_t_dynamics, reduction="none")
# [batch_size, feature_dim]
```

---

### 2.3 Task 3: Inverse Dynamics（逆动力学）

#### 2.3.1 准备扩散输入

```python
# Input: clean actions [batch_size, action_horizon, action_dim]
noise_action = sample_noise(actions.shape)  # [batch_size, action_horizon, action_dim]

# 在时刻 t 的 noisy action
actions_t = time_expanded[:, :, None] * noise_action + (1 - time_expanded[:, :, None]) * actions

# 目标：velocity
u_t_inv_dynamics = noise_action - actions  # [batch_size, action_horizon, action_dim]
```

注意：`next_obs_embedding` 是 clean next obs features（不加噪声），因为 Inverse Dynamics 任务是：**给定 clean obs_t, clean obs_t+1，预测 action_t**

#### 2.3.2 提取 Inverse Dynamics Tokens 并预测

```python
# Inverse Dynamics action tokens：最后 action_horizon 个 tokens
inv_dynamics_hidden = alignment_out[:, -action_horizon:]
# [batch_size, action_horizon, hidden_dim]

# 预测 velocity
v_t_inv_dynamics = inverse_dynamics_head(inv_dynamics_hidden)
# [batch_size, action_horizon, action_dim]
```

#### 2.3.3 计算 MSE Loss

```python
inverse_dynamics_loss = F.mse_loss(v_t_inv_dynamics, u_t_inv_dynamics, reduction="none")
# [batch_size, action_horizon, action_dim]
```

---

## 3. 总 Loss 组合

```python
# Total training loss
total_loss = action_loss.mean() + \
             λ_perception * perception_loss.mean() + \
             λ_dynamics * dynamics_loss.mean() + \
             λ_inv_dynamics * inverse_dynamics_loss.mean()

# λ 是超参数（loss weights）
# 建议初始值：λ_perception = 1.0, λ_dynamics = 1.0, λ_inv_dynamics = 1.0
```

---

## 4. 关键设计决策

### 4.1 为什么所有任务都预测 velocity？

**一致性**：
- Action Expert 已经在预测 velocity
- Alignment Experts 使用相同的框架，确保训练稳定性
- 所有 experts 可以共享相同的 timestep 和 adaRMS 条件

**理论依据**：
- Flow Matching / Rectified Flow 框架已被证明比传统 DDPM 更稳定
- Velocity prediction 直接建模数据流，收敛更快

### 4.2 为什么三个任务合并在一个 suffix？

**效率**：
- 一次 forward 完成三个任务
- 共享 VLM prefix 的计算
- 共享 TTT 参数的更新

**表征共享**：
- 三个任务之间有相关性
- Perception 的输出可以被 Dynamics 和 Inverse Dynamics 看到（通过 attention）
- 促进学习更好的 embodiment context W

### 4.3 为什么 Dynamics 和 Inverse Dynamics 不能互相 attend？

**避免信息泄露**：
- Dynamics 的输入包含 `action`（标签）
- Inverse Dynamics 的输入包含 `next_obs`（标签）
- 如果互相 attend，会导致模型直接复制标签，失去自监督意义

**Block Diagonal Mask**：
```
           prefix  percept  dynamics  inv_dyn
prefix      [✓]     [✓]      [✓]       [✓]
percept     [✓]     [✓]      [✓]       [✓]
dynamics    [✓]     [✓]      [✓]       [✗]    ← 不能看到 inv_dyn
inv_dyn     [✓]     [✓]      [✗]       [✓]    ← 不能看到 dynamics
```

### 4.4 为什么使用 adaRMS 而不是 concat timestep？

**更好的条件注入**：
- adaRMS 通过调制 LayerNorm 的 scale 和 shift 来注入 timestep
- 比直接 concat 更灵活，每一层都能获得 timestep 信息
- 不增加序列长度

**与 Action Expert 一致**：
- Action Expert 已经使用 adaRMS
- Alignment Expert 使用相同机制，确保 TTT 参数共享的有效性

---

## 5. 实现 Checklist

### 5.1 已完成 ✅
- [x] 理解 Action Expert 的 velocity prediction 框架
- [x] 设计 Alignment Expert 的三个任务输入
- [x] 实现 `embed_alignment_suffix()` 方法
- [x] 实现 block diagonal attention mask

### 5.2 待完成 ⏳
- [ ] 在 `forward()` 中调用 alignment expert
- [ ] 实现 perception/dynamics/inverse_dynamics 的 velocity prediction
- [ ] 计算三个 alignment losses
- [ ] 组合 total loss
- [ ] 测试端到端训练

---

## 6. 代码示例

### 6.1 Forward 流程（伪代码）

```python
def forward(self, observation, actions, obs_next, noise=None, time=None):
    # 1. 准备 Action Expert 的输入
    noise_action = sample_noise(actions.shape) if noise is None else noise
    time = sample_time(batch_size) if time is None else time

    x_t_action = time_expanded * noise_action + (1 - time_expanded) * actions
    u_t_action = noise_action - actions

    # 2. Action Expert forward
    prefix_embs = embed_prefix(images, lang_tokens)
    suffix_embs = embed_suffix(state, x_t_action, time)
    (_, action_out), _ = paligemma_with_expert.forward([prefix_embs, suffix_embs], ...)

    v_t_action = action_out_proj(action_out[:, -action_horizon:])
    action_loss = mse_loss(v_t_action, u_t_action)

    # 3. 准备 Alignment Expert 的输入（如果启用）
    if self.use_alignment_expert:
        # Perception
        noise_perception = sample_noise(state.shape)
        state_t = time_expanded * noise_perception + (1 - time_expanded) * state
        u_t_perception = noise_perception - state

        # Dynamics
        noise_obs = sample_noise(next_obs_features.shape)
        next_obs_t = time_expanded * noise_obs + (1 - time_expanded) * next_obs_features
        u_t_dynamics = noise_obs - next_obs_features

        # Inverse Dynamics
        actions_t = time_expanded * noise_action + (1 - time_expanded) * actions
        u_t_inv_dynamics = noise_action - actions

        # Embed alignment suffix
        alignment_suffix_embs, ..., block_diagonal_ranges = embed_alignment_suffix(
            noisy_state=state_t,
            actions=actions,  # clean actions for dynamics
            noisy_next_obs_features=next_obs_t,
            next_obs_embedding=next_obs_features,  # clean for inverse dynamics
            noisy_actions=actions_t,
            timestep=time,
        )

        # 4. Alignment Expert forward
        alignment_att_2d_masks = make_att_2d_masks(
            alignment_pad_masks, alignment_att_masks, block_diagonal_ranges
        )
        (_, alignment_out), _ = alignment_expert.forward([prefix_embs, alignment_suffix_embs], ...)

        # 5. 提取三个任务的 hidden states 并预测
        v_t_perception = perception_head(alignment_out[:, 0])
        v_t_dynamics = dynamics_head(alignment_out[:, dynamics_start:dynamics_end].mean(1))
        v_t_inv_dynamics = inverse_dynamics_head(alignment_out[:, -action_horizon:])

        # 6. 计算 alignment losses
        perception_loss = mse_loss(v_t_perception, u_t_perception)
        dynamics_loss = mse_loss(v_t_dynamics, u_t_dynamics)
        inverse_dynamics_loss = mse_loss(v_t_inv_dynamics, u_t_inv_dynamics)

        # 7. 组合 total loss
        total_loss = action_loss.mean() + \
                     λ_perception * perception_loss.mean() + \
                     λ_dynamics * dynamics_loss.mean() + \
                     λ_inv_dynamics * inverse_dynamics_loss.mean()

        return total_loss, {
            'action_loss': action_loss.mean().item(),
            'perception_loss': perception_loss.mean().item(),
            'dynamics_loss': dynamics_loss.mean().item(),
            'inverse_dynamics_loss': inverse_dynamics_loss.mean().item(),
        }

    return action_loss.mean()
```

---

## 7. 训练细节

### 7.1 Loss Weights 调优

建议的初始值：
```python
λ_perception = 1.0       # 状态预测
λ_dynamics = 1.0         # 动态预测
λ_inv_dynamics = 1.0     # 逆动力学
```

如果某个任务 loss 不稳定或收敛慢，可以调整对应的 λ。

### 7.2 Timestep 采样

所有任务共享同一个 timestep：
```python
time = sample_beta(alpha=1.5, beta=1.0, bsize=batch_size)
time = time * 0.999 + 0.001  # 避免边界值 0 和 1
```

### 7.3 Noise 采样

每个任务的 noise 是独立采样的：
```python
noise_action = sample_noise(actions.shape)      # 为 Action Expert 和 Inverse Dynamics
noise_perception = sample_noise(state.shape)     # 为 Perception
noise_obs = sample_noise(next_obs.shape)         # 为 Dynamics
```

这样可以增加训练的多样性。

---

## 8. 与 Action Expert 的对比

| 特性 | Action Expert | Alignment Experts |
|------|--------------|-------------------|
| 输入 | state + noisy_actions | 三个任务的不同输入 |
| 输出 tokens | action_horizon 个 | perception(1) + dynamics(variable) + inv_dyn(action_horizon) |
| Attention mask | Causal within actions | Block diagonal (Dyn ⊥ InvDyn) |
| 预测目标 | velocity of actions | velocity of state/obs/actions |
| Timestep 注入 | adaRMS | adaRMS（共享同一个 time）|
| TTT 参数 | 自己的 W | 与 Action Expert 共享 W |

---

**最后更新**: 2025-10-14
**状态**: 设计完成，待实现
**下一步**: 实现 forward 方法并测试端到端训练
