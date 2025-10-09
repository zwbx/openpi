# TTT Layer Integration - 关键设计笔记

> 记录 TTT (Test-Time Training) 层集成到 Action Expert 的关键设计决策和实现细节

## 📋 目录
- [使用场景](#使用场景)
- [关键设计决策](#关键设计决策)
- [架构细节](#架构细节)
- [实现要点](#实现要点)
- [参考资料](#参考资料)

---

## 使用场景

### Action Expert 的作用
- **模型**: `GemmaForCausalLM` (不是 PaliGemma)
- **任务**: 对噪声 Action 进行去噪 (denoising)
- **输入**: Noisy action sequences
- **处理**: 使用 TTT 进行 batch-level 优化

### 为什么需要 TTT
- Action 的形状是确定的
- TTT 用于 batch-level 优化，**不是 sequential 的**
- 每个样本的所有 tokens 一起参与优化，去除噪声

---

## 关键设计决策

### 1. TTT 的放置位置

**参考论文**: http://arxiv.org/abs/2504.05298

```
正确的放置顺序:
residual → LayerNorm → Attention → TTT → Residual Connection
                                    ↓
                         (TTT 在 attention 之后，
                          但在 residual 连接之前)
```

**关键引用** (用户原话):
> "ttt 具体怎么加入到 attention 中 参照一下 http://arxiv.org/abs/2504.05298，ttt 应该是在 attention 之后，但是在 attention 的 residual 之前"

### 2. Dual-Gate 机制

**必须使用两个独立的 gate**:
- `gate_attn`: 控制 attention 分支
- `gate_ttt`: 控制 TTT 分支

**架构**:
```python
output = residual + gate_attn * attn_out + gate_ttt * ttt_out
```

**用户原话**:
> "我觉得这里应该分两个 Gate，gate_attn 和 gate_ttt"

### 3. Adaptive Normalization

**关键修正** (用户原话):
> "去掉 GemmaDecoderLayer 中的 ttt 之后的 norm，把他的 norm 集成到 ttt layer 自身里面"

**原因**:
- TTT layer 的 inner loop 优化目标也有残差设计 (参见 `ttt.py` lines 503, 554, 556)
- normalization 应该是 TTT 内部的一部分，不应该在外部

**实现**:
- TTT layer 内部使用 `GemmaRMSNorm`
- 与 `GemmaDecoderLayer` 的实现保持统一
- `adarms_cond` 同时影响 norm 和 gate

---

## 架构细节

### TTT 优化模式

**不同于标准 TTT**:

| 特性 | 标准 TTT (ttt.py) | Action Expert TTT |
|------|------------------|-------------------|
| Position IDs | ✅ 需要 | ❌ 不需要 |
| RoPE | ✅ 使用 | ❌ 不使用 |
| Sequential Scan | ✅ 使用 | ❌ 不使用 |
| Mini-batch 分段 | ✅ 分段处理 | ❌ 整个序列一起 |
| 优化方式 | Position-wise | Batch-parallel |

**用户原话**:
> "我这里其实不用 cache 的，因为这里的 ttt 是 batch level 的优化，而不是 sequential 的"
>
> "所以这个 TTT 的运行，这个 TTT 的优化是在所有的噪声 Action Token 上进行的"
>
> "所以这里的 Position IDS 可以去掉，不需要这个参数"

### Batch-Parallel 优化

**核心概念**:
```python
# 每个 batch 样本维护独立的 W 参数
W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone()  # [B, num_heads, head_dim, head_dim]

# 所有 tokens 同时参与优化
grad_W1 = torch.einsum("bhld,bhlf->bhdf", X1, grad_l_wrt_Z1)
W1_updated = W1_init - eta * grad_W1
```

**关键点**:
- **Batch 维度**: 不同样本有不同的 W (用于去噪不同的 noisy action)
- **Sequence 维度**: 同一样本的所有 tokens 共享一个 W，一起参与优化
- **无位置依赖**: 所有 tokens 平等对待

**用户原话**:
> "注意在这里里 mini_batch_size 的大小等于 seq_len，所以这里不存在沿着 sequence 扫描的情况"

---

## 实现要点

### 1. 文件结构

```
src/openpi/models_pytorch/
├── ttt_with_gate.py                    # 新文件：TTT layer 实现
└── transformers_replace/models/gemma/
    ├── configuration_gemma.py          # 添加 TTT 配置
    └── modeling_gemma.py               # 集成 TTT 到 decoder layer
```

### 2. GemmaRMSNorm 统一

**用户要求**:
> "是的，也换成 GemmaRMSNorm 跟 GemmaDecoderLayer 里面的实现统一"

**实现**:
- TTT layer 内部直接使用 `GemmaRMSNorm`
- 保持与 `GemmaDecoderLayer` 完全一致的 adaptive normalization 行为
- `adarms_cond` 影响 scale、shift 和 gate

### 3. Gate 的形状

**关键**: Gate 的形状是 `[B, 1, hidden_size]`，不是 `[B, 1]`

**原因**:
```python
# GemmaRMSNorm.forward 返回:
# - normed_inputs: [B, L, hidden_size]
# - gate: [B, 1, hidden_size]  (经过 unsqueeze(1) 后的 modulation chunk)

# 在 residual connection 中使用:
hidden_states = hidden_states + gate_ttt * ttt_output
# gate_ttt: [B, 1, hidden_size] broadcasts with ttt_output: [B, L, hidden_size]
```

### 4. 不要修改原始 ttt.py

**用户明确指示**:
> "不要直接修改 @ttt.py 将其作为参照，你可以在 /opt/tiger/openpi/src/openpi/models_pytorch 这个里面新建类"
>
> "并不只是 wrapper 而是 ttt layer 本身，你可以参照 @ttt.py 或者复制过来都行"

**执行方式**:
- 创建新文件 `ttt_with_gate.py`
- 复制并改造 `TTTLinear` 的核心逻辑
- 保持原始 `ttt.py` 不变作为参考

---

## 配置参数

### GemmaConfig 新增参数

```python
use_ttt: bool = False                      # 是否启用 TTT
ttt_mini_batch_size: int = 64              # 兼容参数（实际不使用）
ttt_mode: str = "after_attn"               # 集成模式（兼容参数）
ttt_layer_positions: Optional[list] = None # 指定哪些层使用 TTT
```

**用户原话**:
> "ttt_layer_positions ttt_mode 都加上"

### TTTWithAdaptiveNorm 参数

```python
TTTWithAdaptiveNorm(
    num_heads=config.num_attention_heads,
    hidden_size=config.hidden_size,
    mini_batch_size=64,           # 保留兼容性，实际不使用
    rope_theta=config.rope_theta, # 保留兼容性，实际不使用
    use_adarms=True,              # 是否使用 adaptive normalization
    adarms_cond_dim=cond_dim,     # Timestep embedding 维度
    eps=config.rms_norm_eps
)
```

---

## 代码关键点

### TTT Layer Forward Flow

```python
def forward(self, hidden_states, adarms_cond=None, cache_params=None):
    # 1. Adaptive RMS Norm (返回 normalized 和 gate)
    normalized_hidden_states, gate = self.input_norm(hidden_states, adarms_cond)

    # 2. Q/K/V projections
    XQ, XK, XV = self.get_qkv_projections(normalized_hidden_states)

    # 3. Batch-parallel TTT optimization
    output = self.ttt_batch_parallel(XQ, XK, XV, normalized_hidden_states)

    # 4. Post-norm and projection
    output = self.post_norm(output)
    output = self.o_proj(output)

    return output, gate  # gate: [B, 1, hidden_size]
```

### GemmaDecoderLayer Integration

```python
# [1] Attention + TTT Block
residual = hidden_states
hidden_states, gate_attn = self.input_layernorm(hidden_states, adarms_cond)

# Attention
attn_output, _ = self.self_attn(hidden_states, ...)

# Gated residual for attention
hidden_states = _gated_residual(residual, attn_output, gate_attn)

# TTT (if enabled)
if self.ttt_layer is not None:
    ttt_output, gate_ttt = self.ttt_layer(attn_output, adarms_cond)
    if gate_ttt is not None:
        hidden_states = hidden_states + gate_ttt * ttt_output
    else:
        hidden_states = hidden_states + ttt_output

# [2] MLP Block
residual = hidden_states
hidden_states, gate_mlp = self.post_attention_layernorm(hidden_states, adarms_cond)
hidden_states = self.mlp(hidden_states)
hidden_states = _gated_residual(residual, hidden_states, gate_mlp)
```

### TTT Batch-Parallel Optimization

```python
def ttt_batch_parallel(self, XQ, XK, XV, X):
    # 每个样本独立的 W 参数
    W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone()
    b1_init = self.b1.unsqueeze(0).expand(B, -1, -1, -1).clone()

    # 计算学习率 (基于 X 的平均值)
    X_mean = X.mean(dim=1)
    eta = self.ttt_base_lr * ttt_lr / head_dim

    # TTT 优化目标: 最小化重构误差
    Z1 = torch.einsum("bhld,bhdf->bhlf", XK, W1_init) + b1_init
    reconstruction_target = XV - XK  # 残差设计

    # 计算梯度
    grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

    # Batch-parallel 梯度下降
    grad_W1 = torch.einsum("bhld,bhlf->bhdf", XK, grad_l_wrt_Z1)
    grad_b1 = grad_l_wrt_Z1.sum(dim=2, keepdim=True)

    W1_updated = W1_init - eta * grad_W1
    b1_updated = b1_init - eta * grad_b1

    # 使用更新后的参数前向传播
    Z1_updated = torch.einsum("bhld,bhdf->bhlf", XQ, W1_updated) + b1_updated
    Z1_normalized = ln_fwd(Z1_updated, ln_weight, ln_bias)

    return XQ + Z1_normalized  # 残差连接
```

---

## 参考资料

### 论文
- **TTT Paper**: http://arxiv.org/abs/2504.05298
  - 描述了 TTT 如何加入到 attention 中
  - TTT 应该在 attention 之后，但在 residual 之前

### 代码参考
- **Original TTT**: `/opt/tiger/openpi/ttt.py`
  - Lines 234-239: TTT 内部 LayerNorm 初始化
  - Line 503: 重构目标的残差设计 `reconstruction_target = XV - XK`
  - Lines 554, 556: 输出的残差连接 `XQW = XQ + Z1_bar`

### 相关文件
- `configuration_gemma.py`: TTT 配置参数
- `modeling_gemma.py`: TTT 集成到 decoder layer
- `ttt_with_gate.py`: TTT layer 实现
- `debug_action_expert.py`: 测试脚本

---

## 测试要点

### 测试场景

1. **无 AdaRMS 模式**
   - `use_adarms=False`
   - gate 应该为 `None`
   - TTT 正常工作

2. **有 AdaRMS 模式**
   - `use_adarms=True`
   - gate 形状: `[B, 1, hidden_size]`
   - gate 初始值接近 0 (因为 dense layer 初始化为 0)

3. **梯度流测试**
   - 验证梯度可以正确反向传播
   - `adarms_cond` 的梯度应该正常

4. **不同序列长度**
   - 测试 seq_len = 8, 16, 32, 64
   - 所有长度都应该正常工作

### 运行测试

```bash
# TTT 独立测试
python src/openpi/models_pytorch/ttt_with_gate.py

# Action Expert 集成测试
python debug_action_expert.py
```

---

## 常见问题

### Q: 为什么不需要 position_ids？
**A**: 因为这里的 TTT 是用于去噪，不是 sequential generation。所有 tokens 平等对待，在整个序列上进行 batch-level 优化。

### Q: 为什么不需要 scan？
**A**: 因为 `mini_batch_size == seq_len`，整个序列就是一个 mini-batch，不需要分段处理。

### Q: 为什么需要两个 gate？
**A**: 因为 attention 和 TTT 是两个独立的分支，需要独立控制它们的贡献。这样模型可以学习到在不同的 timestep 下如何平衡两个分支。

### Q: gate 的形状为什么是 [B, 1, hidden_size]？
**A**: 这是 `GemmaRMSNorm` 的设计。modulation 经过 `unsqueeze(1)` 后 chunk 成 scale、shift、gate，每个都是 `[B, 1, hidden_size]`，可以和 `[B, L, hidden_size]` 正确 broadcast。

### Q: adarms_cond 影响什么？
**A**:
- **Normalization**: 通过 scale 和 shift 调整归一化结果
- **Gate**: 控制分支的贡献程度
- 两者都依赖于 diffusion timestep embedding

---

## Git Commit

**Commit Hash**: `a14a247`

**Title**: Add TTT (Test-Time Training) layer integration to Action Expert

**Files Changed**:
- `configuration_gemma.py` (+16 lines)
- `modeling_gemma.py` (+59 lines, -6 lines)
- `ttt_with_gate.py` (+419 lines, new file)
- `.gitignore` (+1 line)

**Total**: 489 insertions, 6 deletions

---

_Last Updated: 2025-10-09_
_Author: 用户指导 + Claude Code 实现_
