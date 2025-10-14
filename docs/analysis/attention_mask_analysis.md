# Attention Mask 流程分析

**日期**: 2025-10-14
**发现**: VLM 不能看到 Action Expert（单向信息流）

---

## 核心发现

训练时 VLM 和 Action Expert 的 Attention 是**单向的**，而非双向的：

```
VLM  ⇄  VLM      ✅ VLM tokens 之间全连接（prefix-LM）
VLM  →  Expert   ❌ VLM 不能看到 Expert
VLM  ←  Expert   ✅ Expert 可以看到 VLM（获取视觉语言信息）
Expert ⇄ Expert  ✅ Expert tokens 之间因果注意力
```

---

## Attention Mask 完整流程

### 1. 生成 1D att_masks

#### Prefix (VLM) - `pi0_pytorch.py:228, 243`
```python
# 图像 tokens: 3张图 × 256 = 768
att_masks += [0] * 768

# 语言 tokens: 200
att_masks += [0] * 200

# prefix_att_masks = [0, 0, ..., 0]  (968个0)
```

#### Suffix (Action Expert) - `pi0_pytorch.py:325`
```python
# action_horizon = 4
att_masks += [1] + ([0] * 3)

# suffix_att_masks = [1, 0, 0, 0]
```

#### 拼接 - `pi0_pytorch.py:358`
```python
att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
# [0, 0, ..., 0, 1, 0, 0, 0]
#  ↑ 968个0    ↑ 4个
```

**att_masks 语义**:
- `0`: 属于当前 attention block（可以互相 attend）
- `1`: 开启新的因果 block

---

### 2. 计算 cumsum - `pi0_pytorch.py:86`

```python
cumsum = torch.cumsum(att_masks, dim=1)
# [0, 0, ..., 0, 1, 1, 1, 1]
#  ↑ 968个0    ↑ 4个1
```

**Cumsum 将 attention 模式映射为可比较的层级**:
- Block 0: cumsum = 0 (VLM)
- Block 1: cumsum = 1 (Expert)

---

### 3. 生成 2D Attention Mask - `pi0_pytorch.py:87`

```python
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
# 形状: [batch, 972, 972]
```

**规则**: Token i 可以 attend 到 token j 当且仅当 `cumsum[j] <= cumsum[i]`

#### 实际的 Attention Matrix

```
                    Keys
                ┌──────────┬────────┐
                │   VLM    │ Expert │
                │  (968)   │  (4)   │
                │cumsum=0  │cumsum=1│
        ┌───────┼──────────┼────────┤
        │ VLM   │    ✓     │   ✗    │  cumsum[query]=0
Queries │ (968) │          │        │  只能 attend 到 cumsum<=0
        │cumsum │  全连接  │ 不可见 │
        │  =0   │          │        │
        ├───────┼──────────┼────────┤
        │Expert │    ✓     │ Causal │  cumsum[query]=1
        │  (4)  │          │        │  可以 attend 到 cumsum<=1
        │cumsum │  可见    │  因果  │
        │  =1   │          │        │
        └───────┴──────────┴────────┘

✓ = 可以 attend (True)
✗ = 不能 attend (False)
Causal = 三角矩阵（只看前面的）
```

**具体示例**:
```
att_masks = [0, 0, 0, 1, 0, 0, 0]
cumsum    = [0, 0, 0, 1, 1, 1, 1]

Attention Matrix:
         img1 img2 lang1 act1 act2 act3 act4
img1  [   1    1     1    0    0    0    0  ]  ← cumsum=0, attend to cumsum<=0
img2  [   1    1     1    0    0    0    0  ]
lang1 [   1    1     1    0    0    0    0  ]
act1  [   1    1     1    1    0    0    0  ]  ← cumsum=1, attend to cumsum<=1, causal
act2  [   1    1     1    1    1    0    0  ]
act3  [   1    1     1    1    1    1    0  ]
act4  [   1    1     1    1    1    1    1  ]
```

---

### 4. 转换为 4D Additive Mask - `pi0_pytorch.py:176-177`

```python
att_2d_masks_4d = att_2d_masks[:, None, :, :]
# 形状: [batch, 1, 972, 972]
#             ↑ head 维度（会 broadcast）

# 转换为 additive mask
return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)
#                                   ↑ 可attend  ↑ 不可attend (≈-∞)
```

**为什么用负无穷？**
```python
scores = Q @ K.T + mask
weights = softmax(scores)

# 当 mask = -∞:
#   scores + (-∞) = -∞
#   exp(-∞) ≈ 0
#   → softmax 后该位置权重 ≈ 0
```

---

### 5. 应用到 Attention 计算 - `gemma_pytorch.py:210-217`

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling):
    # Step 1: 计算 attention scores
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    # 形状: [batch, 8_heads, 972, 972]

    # Step 2: 应用 mask ← 关键位置
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask  # additive mask

    # Step 3: Softmax
    attn_weights = F.softmax(attn_weights, dim=-1)
    # 被禁止的位置: softmax(-∞) ≈ 0

    # Step 4: 加权求和
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights
```

**Mask 应用位置**: 在 softmax **之前** 加到 attention scores 上

---

## 设计意图分析

### 为什么 VLM 不能看到 Expert？

这是合理的架构设计：

1. **VLM 的职责**: 理解图像和语言，提取多模态特征
   - 不需要知道动作预测
   - 专注于视觉语言理解

2. **Expert 的职责**: 基于 VLM 特征生成动作
   - 需要获取视觉语言上下文
   - 根据状态 + 时间 + VLM 特征做决策

3. **信息流设计**:
   ```
   Images + Language → VLM → 特征
                              ↓
   State + Time + VLM特征 → Expert → Actions
   ```

### 单向 vs 双向的权衡

| 方面 | 单向（当前） | 双向（假设） |
|------|-------------|-------------|
| **VLM 复杂度** | 简单，只处理视觉语言 | 复杂，需要理解动作 |
| **计算效率** | 高，VLM 不依赖 Expert | 低，需要迭代交互 |
| **模块解耦** | 高，VLM 独立 | 低，VLM 和 Expert 耦合 |
| **推理一致性** | 好，VLM 可以 cache | 差，VLM 需要重新计算 |

**结论**: 单向设计更合理，符合模块化和效率原则。

---

## 与推理时的对比

### 训练时（路径3）
```python
# compute_layer_complete (gemma_pytorch.py:167-247)
# 拼接 Q/K/V，在同一个 attention 中计算
query_states = torch.cat([Q_vlm, Q_expert], dim=2)
key_states = torch.cat([K_vlm, K_expert], dim=2)

att_output = eager_attention_forward(
    query_states,  # 包含 VLM + Expert
    key_states,    # 包含 VLM + Expert
    attention_mask,  # 控制单向流动
    ...
)
```

### 推理时（路径1+2）
```python
# 路径1: 计算 VLM 的 KV cache
_, past_kv = vlm.forward(prefix_embs)

# 路径2: Expert 使用 cached KV
output = expert.forward(
    suffix_embs,
    past_key_values=past_kv  # 可以 attend 到 VLM 的 K/V
)
```

两种方式实现相同的信息流：Expert 可以看到 VLM，但 VLM 看不到 Expert。

---

## 相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 生成 prefix att_masks | pi0_pytorch.py | 228, 243 |
| 生成 suffix att_masks | pi0_pytorch.py | 325 |
| 拼接 att_masks | pi0_pytorch.py | 358 |
| make_att_2d_masks | pi0_pytorch.py | 60-89 |
| 转换为 4D mask | pi0_pytorch.py | 174-177 |
| 应用 mask | gemma_pytorch.py | 210-217 |
| 联合 Attention 计算 | gemma_pytorch.py | 167-247 |

---

## 总结

- **Attention 模式**: VLM ⇄ VLM (全连接) + Expert → VLM (单向) + Expert ⇄ Expert (因果)
- **信息流**: VLM 提供上下文 → Expert 做决策
- **实现方式**: 通过 cumsum + 比较生成 2D mask，再转为 additive mask 应用到 attention scores
- **设计理念**: 模块解耦，VLM 专注视觉语言理解，Expert 专注动作生成
