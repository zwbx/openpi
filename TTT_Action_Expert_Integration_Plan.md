# TTT 集成到 Action Expert 实施记录

## 概述

本文档记录了将 Test-Time Training (TTT) layers 集成到 PI0.5 的 Action Expert (GemmaForCausalLM) 中的完整实施过程，包括设计决策、参考实现分析和已完成的工作。

---

## ✅ 已完成工作 (Phase 1 & 2)

### 1. TTT 核心组件实现

#### 1.1 TTTLinear with Batch-Parallel Optimization ✅
**文件**: `/opt/tiger/openpi/src/openpi/models_pytorch/ttt_with_gate.py`

**核心特性**:
- ✅ **Batch-parallel TTT**: 所有 tokens 同时优化，无 sequential scan
- ✅ **Non-causal attention**: 去除因果掩码，允许全局 token 交互（适合去噪）
- ✅ **Learnable input-dependent LR**: `eta = ttt_base_lr * sigmoid(X @ W_lr + b_lr) / head_dim`
- ✅ **Closed-form dual form**: 一步闭式解，高效计算
- ✅ **Per-dimension learnable gating**: 类似 ttt-video-dit 的 SSMGating
- ✅ **Adaptive normalization (可选)**: 支持 AdaRMS 动态 gate（未启用）

**设计选择**:
```python
class TTTWithAdaptiveNorm(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        use_dual_form: bool = True,         # 使用 dual form（更高效）
        gating_alpha_init: float = 0.1,    # 静态 gate 初始化为 0.1
    ):
        # TTT 参数 (W1, b1)
        self.W1 = nn.Parameter(...)
        self.b1 = nn.Parameter(...)

        # 可学习的、输入依赖的学习率参数
        self.learnable_ttt_lr_weight = nn.Parameter(...)  # [num_heads, hidden_size, 1]
        self.learnable_ttt_lr_bias = nn.Parameter(...)    # [num_heads, 1]

        # 静态可学习 gate (tanh(alpha) ≈ 0.1 at start)
        self.gating_alpha = nn.Parameter(torch.ones(hidden_size) * gating_alpha_init)
```

**关键优化**:
- **移除了不必要的 input normalization**: TTT 输入已经是 attention output，不需要再 norm
- **Non-causal**: `Attn1 = XQ @ X1.transpose(-2, -1)` (去掉 `torch.tril`)
- **Learnable eta**: `eta = ttt_base_lr * sigmoid(X @ W_lr + b_lr) / head_dim` [B, num_heads, L, 1]
- **Dual form**: `Z1_bar = XQ @ W1_init - Attn1 @ (eta * grad_l_wrt_Z1) + b1_bar`

#### 1.2 GemmaDecoderLayer 集成 ✅
**文件**: `/opt/tiger/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`

**集成模式**: **After Attention** (Sequential)
```python
# [1] Attention Block
residual = hidden_states
hidden_states, gate_attn = self.input_layernorm(hidden_states, adarms_cond)
attn_output = self.self_attn(hidden_states, ...)
hidden_states = _gated_residual(residual, attn_output, gate_attn)

# [2] TTT Block (if enabled)
if self.ttt_layer is not None:
    ttt_output, gate_ttt = self.ttt_layer(attn_output, adarms_cond)
    hidden_states = hidden_states + gate_ttt * ttt_output

# [3] MLP Block
residual = hidden_states
hidden_states, gate_mlp = self.post_attention_layernorm(hidden_states, adarms_cond)
hidden_states = self.mlp(hidden_states)
hidden_states = _gated_residual(residual, hidden_states, gate_mlp)
```

**支持的配置**:
- `ttt_layer_positions`: 指定哪些层使用 TTT（支持 `"all"` 或 `[14, 15, 16, 17]`）
- `ttt_layer_type`: `"linear"` (当前只支持 linear，可扩展到 MLP)
- `use_dual_form`: `True` (使用闭式解)

#### 1.3 配置系统 ✅
**文件**:
- `/opt/tiger/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py`
- `/opt/tiger/openpi/src/openpi/models/pi0_config.py`
- `/opt/tiger/openpi/src/openpi/training/config.py`

**训练配置示例**:
```python
TrainConfig(
    name="pi05_simpler_zscore_ttt",
    model=pi0_config.Pi0Config(
        pi05=True,
        discrete_state_input=True,
        use_ttt=True,
        ttt_layer_type="linear",           # Linear TTT with closed-form solution
        ttt_layer_positions="all",          # Apply to all layers
        use_dual_form=True,                 # Use dual form for efficiency
    ),
    ...
)
```

### 2. Git 提交历史

**Commit 1**: `83cbfde` - Simplify TTT layer to use fixed learning rate with closed-form solution
- 移除 learnable LR（dual form 不需要）
- 添加 `ttt_layer_type="linear"` 配置

**Commit 2**: `180252b` - Add use_dual_form parameter to TTT layer for optimization method selection
- 添加 dual form 支持
- 支持 `ttt_layer_positions="all"`

**Commit 3**: (未提交) - Remove unnecessary input normalization and add learnable gating
- 移除 TTT 内部的 input norm
- 添加静态 learnable `gating_alpha`

**Commit 4**: `[latest]` - Replace fixed eta with learnable input-dependent learning rate (non-causal)
- ✅ 添加 `learnable_ttt_lr_weight` 和 `learnable_ttt_lr_bias` 参数
- ✅ 移除因果掩码 `torch.tril`（non-causal denoising）
- ✅ 计算输入依赖的 `eta = ttt_base_lr * sigmoid(X @ W_lr + b_lr) / head_dim`
- ✅ 只使用 `ttt_lr_eta`，不使用 `token_eta`（所有 token 平等对待）

---

## 📊 参考实现分析: ttt-video-dit

我们详细分析了 `/opt/tiger/openpi/ttt-video-dit` 项目，识别出以下关键技术：

### 已实现的技术 ✅

| 技术 | ttt-video-dit | OpenPI | 状态 |
|------|---------------|--------|------|
| **Batch-parallel optimization** | ❌ (uses sequential scan) | ✅ | 已实现 |
| **Closed-form dual form** | ✅ | ✅ | 已实现 |
| **Learnable static gating (SSMGating)** | ✅ | ✅ | 已实现 |
| **AdaRMS/AdaLN dynamic gating** | ✅ (AdaLN) | ✅ (AdaRMS) | 已实现 |

### 值得借鉴的技术 🔄

#### 高优先级 (建议实现)

1. **TTTMLP Variant** ⭐⭐⭐
   ```python
   class TTTMLP(TTTBase):
       def __init__(self, config):
           self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, head_dim, 4*head_dim)))
           self.b1 = nn.Parameter(torch.zeros(num_heads, 1, 4*head_dim))
           self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, 4*head_dim, head_dim)))
           self.b2 = nn.Parameter(torch.zeros(num_heads, 1, head_dim))
   ```
   - 更强的表达能力（2-layer MLP vs linear）
   - 容易实现（只需扩展当前 TTTLinear）
   - 可能显著提升去噪质量

2. **`@torch.compile` Decorators** ⭐⭐⭐
   ```python
   @torch.compile
   def ttt_batch_parallel(self, XQ, XK, XV):
       # ... existing code ...
   ```
   - 零成本的 10-30% 加速
   - 只需加装饰器

3. **Reconstruction Target Normalization** ⭐⭐
   ```python
   reconstruction_target = XV - XK
   # Add LayerNorm
   mean = reconstruction_target.mean(dim=-1, keepdim=True)
   std = reconstruction_target.std(dim=-1, keepdim=True)
   reconstruction_target = (reconstruction_target - mean) / (std + eps)
   reconstruction_target = self.ttt_norm_weight * reconstruction_target + self.ttt_norm_bias
   ```
   - 稳定训练
   - 处理不同尺度的动作

#### 中优先级 (值得实验)

4. **Bidirectional TTT** ⭐⭐⭐
   ```python
   # Forward pass
   emb = forward_ssm(emb, seq_metadata)
   emb = residual + forward_gate(emb)

   # Reverse pass (same parameters, different input order)
   emb = torch.flip(emb, dims=[1])
   emb = reverse_ssm(emb, seq_metadata)
   emb = torch.flip(emb, dims=[1])
   emb = residual + reverse_gate(emb)
   ```
   - 捕获双向依赖
   - 使用相同参数（parameter efficient）
   - 可能显著提升质量

5. **L2 Normalization on Q/K** ⭐⭐
   ```python
   XQ = F.normalize(XQ, p=2, dim=-1)
   XK = F.normalize(XK, p=2, dim=-1)
   ```
   - 训练稳定性
   - 一行代码

6. **Gradient Checkpointing** ⭐⭐
   - 为未来更长序列准备
   - PyTorch 原生支持

#### 低优先级 (不推荐)

7. **Sequential mini-batch scan** ⭐
   - 与 batch-parallel 设计冲突
   - **Skip**

8. **Triton/CUDA kernels** ⭐
   - 序列太短（256 tokens），PyTorch 够用
   - **Skip unless bottleneck**

9. **RoPE in TTT** ⭐
   - 与 batch-parallel 冲突
   - Attention 已有 RoPE
   - **Skip**

### Gating 机制对比

**ttt-video-dit 使用双层 gating**:
1. **SSMGating (TTT 内部)**: 静态可学习 `tanh(alpha)`
2. **AdaLN gate (外部)**: 动态 timestep-conditioned gate

**OpenPI 当前设计** (与 ttt-video-dit 一致):
1. **Static `gating_alpha`**: `tanh(self.gating_alpha)` [hidden_size]
2. **AdaRMS gate (外部)**: Timestep-conditioned gate from `adarms_cond`

**重要发现**: ❌ ttt-video-dit 的 SSMGating **没有 cond_dim**，是纯静态可学习的！

---

## 🔧 当前实现细节

### TTT Layer 架构

```python
# Input: attn_output [B, L, hidden_size] (from attention)
# No normalization needed!

# Q/K/V projections
XQ, XK, XV = self.get_qkv_projections(hidden_states)  # [B, L, num_heads * head_dim]
XQ = XQ.reshape(B, L, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]

# Batch-parallel TTT (all tokens optimize simultaneously)
W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, head_dim, head_dim]
b1_init = self.b1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, 1, head_dim]

# Compute input-dependent learning rate
ttt_lr = torch.einsum("blc,hci->bhli", hidden_states, self.learnable_ttt_lr_weight)
ttt_lr = ttt_lr + self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1)
ttt_lr = torch.sigmoid(ttt_lr)  # [B, num_heads, L, 1], range (0, 1)
eta = self.ttt_base_lr * ttt_lr / head_dim

# TTT optimization (dual form - closed-form solution)
reconstruction_target = XV - XK
grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)
Attn1 = XQ @ XK.transpose(-2, -1)  # Non-causal: full attention matrix (no tril)
b1_bar = b1_init - (eta * grad_l_wrt_Z1).sum(dim=2, keepdim=True)
Z1_bar = XQ @ W1_init - Attn1 @ (eta * grad_l_wrt_Z1) + b1_bar

# Output
Z1_normalized = ln_fwd(Z1_bar, ln_weight, ln_bias)
output = XQ + Z1_normalized  # Residual

# Post-processing
output = self.post_norm(output)
output = self.o_proj(output)

# Static learnable gating
gating_alpha = torch.tanh(self.gating_alpha)  # [hidden_size], init at 0.1
return output, gating_alpha
```

### 使用方式 (in modeling_gemma.py)

```python
# TTT receives attention output (no norm)
ttt_output, gate_ttt = self.ttt_layer(attn_output, adarms_cond=None)

# Apply static learnable gate
hidden_states = hidden_states + gate_ttt * ttt_output
```

---

## 📝 待办事项和未来改进

### 短期 (1-2周)

- [ ] **实现 TTTMLP 变体**
  - 创建 `TTTMLPWithAdaptiveNorm` 类
  - 2-layer MLP: `Z = W2 @ GELU(W1 @ X + b1) + b2`
  - 添加 `ttt_layer_type` 配置选项

- [ ] **添加 `@torch.compile`**
  - `ttt_batch_parallel` 方法
  - `get_qkv_projections` 方法

- [ ] **Reconstruction target normalization**
  - 添加 LayerNorm 到 `reconstruction_target = XV - XK`

- [ ] **训练和验证**
  - Debug 训练跑通
  - 监控 TTT gate 的学习情况
  - 对比有无 TTT 的效果

### 中期 (1-2月)

- [ ] **Bidirectional TTT**
  - Forward + Reverse passes
  - 双 gate 机制
  - 消融实验

- [ ] **L2 Normalization on Q/K**
  - 添加到 projection 之后
  - 监控训练稳定性

- [ ] **性能优化**
  - Gradient checkpointing
  - Mixed precision 优化
  - 内存 profiling

### 长期 (3-6月)

- [ ] **Triton kernels** (如果成为瓶颈)
- [ ] **多模态 TTT** (扩展到 vision encoder)
- [ ] **自适应层选择** (动态决定哪些层用 TTT)

---

## 🎯 设计决策记录

### 1. Batch-Parallel vs Sequential Scan

**选择**: Batch-parallel
**理由**:
- Action sequences 较短（256 tokens）
- 无需跨 mini-batch 的状态传递
- 更快的并行计算

### 2. Dual Form vs Primal Form

**选择**: Dual form (closed-form)
**理由**:
- 一步闭式解，无需迭代优化
- 更高效（避免显式计算 `grad_W1`）
- 支持 learnable input-dependent LR（通过 `eta * grad`）

### 3. 移除 Input Normalization

**理由**:
- TTT 输入是 `attn_output`，已经是良好的表示
- 避免重复 normalization
- 与 ttt-video-dit 设计一致

### 4. Static Learnable Gating

**选择**: `gating_alpha = nn.Parameter(torch.ones(hidden_size) * 0.1)`
**理由**:
- 与 ttt-video-dit SSMGating 一致
- 初始化为 0.1，训练中自适应调整
- Per-dimension 控制（不是 scalar）

### 5. AdaRMS Gate (可选，未启用)

**保留但不使用**:
- 已有外层 AdaRMS gate（在 `input_layernorm`）
- 保留接口供未来实验
- 目前只用静态 `gating_alpha`

### 6. Non-Causal Attention (去除因果掩码)

**选择**: 去除 `torch.tril`，使用完整 attention matrix
**理由**:
- 去噪任务中所有 tokens 同时可见，无因果依赖
- 允许每个 token 利用整个序列信息
- 与标准 DiT self-attention 对齐

### 7. Input-Dependent Learning Rate

**选择**: `eta = ttt_base_lr * sigmoid(X @ W_lr + b_lr) / head_dim`
**理由**:
- 让学习率适应不同输入特征
- 保留原始 ttt.py 中 `ttt_lr_eta` 的设计思想
- 去掉 `token_eta`（位置权重），因为非因果任务中所有 token 平等

---

## 📚 参考文档

### 内部文档
- `/opt/tiger/openpi/ttt_video_dit_comparison.md` - ttt-video-dit 对比分析
- `/opt/tiger/openpi/ttt-video-dit/` - 参考实现

### 外部资源
- [TTT Paper](https://arxiv.org/abs/2407.04620) - Test-Time Training layers
- [ttt-video-dit repo](https://test-time-training.github.io/video-dit/) - Video generation with TTT
- [CogVideoX](https://github.com/THUDM/CogVideo) - Base architecture

---

## 🐛 已知问题

### 当前无已知问题 ✅

---

## 📈 性能基准 (待测试)

### 预期提升
- **长序列建模**: 更好处理 256 token 的 action sequences
- **去噪质量**: TTT 自适应优化提升动作预测
- **训练稳定性**: Learnable gating 自动调整 TTT 贡献

### 待测量指标
- [ ] Loss curves (with vs without TTT)
- [ ] Action prediction accuracy
- [ ] Training time overhead
- [ ] Memory usage
- [ ] Gate values distribution

---

**最后更新**: 2025-10-10
**状态**: Phase 1 & 2 & 3 完成，待训练验证
**下一步**: 训练测试 + 可选实现 TTTMLP 变体

---

## 🆕 Phase 3 更新 (2025-10-10)

### 核心改进：Non-Causal + Input-Dependent Learning Rate

**动机**:
原始 TTT 实现为语言建模设计（因果依赖），但 diffusion 去噪任务中所有 tokens 应同时可见。同时，固定学习率无法适应不同输入特征。

**关键变更**:
1. **去除因果掩码**: `Attn1 = XQ @ X1.transpose(-2, -1)` (no `torch.tril`)
2. **可学习 LR**: `eta = ttt_base_lr * sigmoid(X @ W_lr + b_lr) / head_dim` [B, num_heads, L, 1]
3. **去掉 token_eta**: 不再使用位置相关的递减权重（所有 token 平等）

**参数增加**:
- `learnable_ttt_lr_weight`: [num_heads, hidden_size, 1]
- `learnable_ttt_lr_bias`: [num_heads, 1]

**与原始 ttt.py 对比**:

| 项目 | 原始 ttt.py | OpenPI TTT |
|------|-------------|------------|
| 因果掩码 | ✅ `torch.tril` | ❌ 去除 |
| token_eta | ✅ `[1.0, 0.5, 0.33, ...]` | ❌ 去除 |
| ttt_lr_eta | ✅ 可学习 | ✅ 可学习 |
| eta 组合 | `token_eta * ttt_lr_eta` | `ttt_lr_eta` only |
| 适用场景 | 因果语言建模 | 非因果去噪 |

**测试结果**: ✅ 所有单元测试通过（4/4）
