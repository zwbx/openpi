# Codex Plan — OpenPI Self‑Alignment with PEFT Prefix Token

更新时间: 2025-10-17

## 目标
- 联合优化：`L_total = L_action + λ_p·L_perc + λ_d·L_dyn + λ_i·L_inv`。
  - `L_action`：主 action flow-matching 损失。
  - `L_perc`/`L_dyn`/`L_inv`：Alignment Expert 的三项自监督损失。
- **完全移除 TTT**：不再使用 TTT 层及其相关的重构损失 `L_ttt`。
- **采用 PEFT 前缀 Token（E-Token）**：在 Action Expert 和 Alignment Expert 输入前各插入一个可训练的前缀 token，两处共享同一参数，用于在两路 expert 之间传递"具身/环境上下文"。
- 在线自对齐：inference → buffer → 定频 align()，对齐阶段仅使用三项自监督损失，默认仅更新 E-Token 参数。

## 当前进度
- 多专家骨干与前向
  - 训练：VLM + Action Expert +（可选）Alignment Expert 三路联合 attention；推理可分路执行。
  - 参考: `src/openpi/models_pytorch/gemma_pytorch.py:1`（`PaliGemmaWithExpertModel`）。
- Alignment Suffix 与掩码
  - `embed_alignment_suffix()` 将三任务输入拼接为一个 suffix，输出 pad/att masks 和 adarms cond；`make_att_2d_masks()` 支持 block‑diagonal。
  - 参考: `src/openpi/models_pytorch/pi0_pytorch.py:466`、`src/openpi/models_pytorch/pi0_pytorch.py:143`。
- 在线自对齐闭环
  - `buffer`、`align()`、`sample_actions()` 已打通；当前 align 优化三项自监督损失。
  - 参考: `src/openpi/models_pytorch/pi0_pytorch.py:20`、`:820`、`:927`。
- 验证脚本
  - `test_align_flow.py` 可用于快速自检对齐流程。

## 最新决策更新（2025-10-17）
- **完全移除 TTT 组件**：不再使用 TTT 层、TTT 损失 `L_ttt`、TTT 相关配置和参数。
- **简化为纯 PEFT 方案**：仅通过可训练的 E-Token（前缀 token）在两个 expert 之间传递上下文。
- **损失函数简化**：`L_total = L_action + λ_p·L_perc + λ_d·L_dyn + λ_i·L_inv`，移除所有 TTT 相关项。
- **对齐策略简化**：在线对齐 `align()` 仅优化 E-Token 参数，使用三项自监督损失。

## 设计变更与落地
- `src/openpi/models_pytorch/pi0_pytorch.py`
  - 新增 `self.peft_prefix_token = nn.Parameter(torch.zeros(1, hidden_size))`：可训练的前缀 token，在两个 expert 输入前共享。
  - `embed_suffix()`：在 action suffix 前插入 E-Token，更新 `pad_masks/att_masks`。
  - `embed_alignment_suffix()`：在 alignment suffix 前插入 E-Token，更新 `pad_masks/att_masks` 与 `block_diagonal_ranges`。
  - 训练前向：合并损失 `L_total = L_action + λ_p·L_perc + λ_d·L_dyn + λ_i·L_inv`。
  - `align()`：仅更新 E-Token 参数，使用三项自监督损失。
  - **移除所有 TTT 相关代码**：删除 TTT 层创建、TTT 损失计算、TTT 参数更新逻辑。
- `src/openpi/models_pytorch/gemma_pytorch.py`
  - **移除 TTT 层集成**：删除 TTT 层替换逻辑，恢复标准 Transformer 架构。
  - 移除 TTT 损失聚合代码。
- `src/openpi/training/config.py`
  - 新增 PEFT 相关配置：`use_peft_prefix_token: bool = True`、`peft_num_tokens: int = 1`、`peft_init: str = "zeros"`、`peft_lr: float = 1e-3`。
  - 保留对齐损失权重：`lambda_perc/lambda_dyn/lambda_inv`，**移除 `lambda_ttt`**。
- `src/openpi/models/pi0_config.py`
  - **移除所有 TTT 相关配置项**：`ttt_layer_positions`、`ttt_base_lr`、`meta_learning` 等。

## 待完成与差距
- **实现 E-Token 参数与注入逻辑**：在 `PI0Pytorch.__init__()` 中创建 `peft_prefix_token`，在 `embed_suffix()` 和 `embed_alignment_suffix()` 中插入。
- **更新掩码逻辑**：确保 E-Token 在两路 expert 中正确可见；调整 `block_diagonal_ranges` 使 E-Token 单独一块。
- **清理 TTT 相关代码**：从 `pi0_pytorch.py`、`gemma_pytorch.py`、配置文件中移除所有 TTT 引用。
- **训练循环更新**：合并权重 `λ_p·L_perc + λ_d·L_dyn + λ_i·L_inv`，日志打印各项损失。
- **对齐函数简化**：`align()` 仅优化 E-Token，移除 TTT 参数过滤逻辑。
- `next_obs_seq_len` 动态推断：移除硬编码（`pi0_pytorch.py:705`），从 `alignment_suffix_embs` 真实结构确定各任务范围。
- 测试增强：
  - "E-Token 梯度"验证：训练与对齐阶段 E-Token 有非零梯度。
  - "E-Token 共享"验证：Action 与 Alignment 路的 E-Token 使用同一参数实例。
  - "block‑diagonal 正确性"验证：Dyn 与 InvDyn 不互看；E-Token 对所有块可见。
  - "三项损失收敛"验证：对齐阶段三项自监督损失可下降。

## 下一步（建议顺序）
1) **移除 TTT 依赖**：从代码与配置中删除所有 TTT 相关内容（层、损失、配置项）。
2) **实现 E-Token**：在 `pi0_pytorch.py` 中添加 `peft_prefix_token` 参数与注入逻辑。
3) **更新掩码**：调整 `embed_suffix()` 和 `embed_alignment_suffix()` 以支持 E-Token。
4) **更新训练与对齐**：简化损失函数为 `L_action + 三项自监督`；`align()` 仅更新 E-Token。
5) **增加测试**：编写 E-Token 注入、梯度、共享的单测；fake data 端到端测试。

## 验证清单
- 前向/推理：移除 TTT 后行为正常；训练返回正确的 `L_total`。
- E-Token 参数：训练与对齐后 E-Token 值发生变化；初始化符合配置。
- E-Token 共享：Action 与 Alignment 路使用同一 `peft_prefix_token` 实例。
- 掩码：E-Token 对所有后续 token 可见；Dyn 与 InvDyn 互斥；Perception 可被两者访问。
- 训练/对齐：`L_total`、分项损失下降，显存/吞吐可接受。
- 无 TTT 残留：代码中无 `ttt_layer`、`L_ttt`、`TTTWithAdaptiveNorm` 等引用。

## 参考文件
- `src/openpi/models_pytorch/gemma_pytorch.py:1`
- `src/openpi/models_pytorch/pi0_pytorch.py:143`
- `src/openpi/models_pytorch/pi0_pytorch.py:466`
- `src/openpi/models_pytorch/pi0_pytorch.py:559`
- `test_align_flow.py:1`

---

说明：本计划用于跟踪"自对齐 + PEFT 前缀 Token"的实现与验证进度，随着训练集成与测试推进可逐步勾选与扩展。

## 架构概览：纯 PEFT 方案（无 TTT）

### 核心设计
- **移除 TTT 层**：不再使用 TTT 层及其重构损失；Action Expert 和 Alignment Expert 均使用标准 Transformer 架构。
- **PEFT 前缀 Token（E-Token）**：引入一个可训练的前缀 token，分别插入到 Action Expert 和 Alignment Expert 的输入最前端；两处共享同一参数实例，用于在两路 expert 之间传递"具身/环境上下文"。
- **联合优化**：`L_total = L_action + λ_p·L_perc + λ_d·L_dyn + λ_i·L_inv`，E-Token 与模型其他参数通过同一外层优化器更新（可为 E-Token 设置独立学习率 `peft_lr`）。

### 架构与掩码设计
- **Action Expert 输入**：`[ ... VLM prefix ... ] + [E] + [action_suffix_tokens]`
  - `[E]` 放在 action suffix 的最前；action_suffix 内部保持原因果掩码。
  - 设置 `att_masks`：E-Token 对后续所有 action token 可见（因果掩码允许后续 token 看到 E）。
- **Alignment Expert 输入**：`[ ... VLM prefix ... ] + [E] + [Perception] + [Dynamics block] + [Inverse Dynamics block]`
  - `[E]` 为 alignment suffix 的第一个块，所有后续任务块可见 E-Token。
  - 维持 Dyn 与 InvDyn 的 block-diagonal：两者互斥可见，但都可见 `[E]` 与 `Perception`。
  - `block_diagonal_ranges` 调整为：`E` 单独一块（不与任何块互斥），`Dyn` 与 `InvDyn` 互斥；`Perception` 可与两者互见。

### 参数与实现要点
- **PEFT E-Token 参数**：`self.peft_prefix_token = nn.Parameter(torch.zeros(1, hidden_size))`，与两个 expert 的 `hidden_size` 对齐。
- **前向拼接**：
  - Action 路：在 `embed_suffix()` 返回的 `action_time_emb` 前拼接 `E`；同步扩展 `pad_masks/att_masks`。
  - Alignment 路：在 `embed_alignment_suffix()` 的最前插入 `E`；同步扩展 `pad_masks/att_masks` 与 `block_diagonal_ranges`（确保 `E` 不被屏蔽）。
- **优化策略**：
  - 训练：`E-Token` 与模型其他参数共同更新；可为 `E-Token` 设置单独学习率（如 `peft_lr`）。
  - 在线对齐 `align()`：仅更新 `E-Token`，联合使用三项自监督损失，不依赖 GT actions 亦可运行。
- **配置扩展**：
  - `use_peft_prefix_token: bool = True`
  - `peft_num_tokens: int = 1`
  - `peft_init: str = "zeros"` 或 `"normal"`（默认零初始化）
  - `peft_lr: float = 1e-3`（可与主 lr 分离）
  - `lambda_perc/lambda_dyn/lambda_inv`: 各损失权重（**移除 `lambda_ttt`**）

### 落地步骤（代码）
1) **移除 TTT 依赖**：
   - 从 `pi0_pytorch.py`、`gemma_pytorch.py` 中删除所有 TTT 层创建、TTT 损失计算代码。
   - 从配置文件（`pi0_config.py`、`training/config.py`）中移除 `ttt_layer_positions`、`ttt_base_lr`、`lambda_ttt` 等。
2) **实现 E-Token**：
   - 在 `PI0Pytorch.__init__()` 中引入 `self.peft_prefix_token = nn.Parameter([...])`。
   - 提供 `make_peft_token(batch_size)` 生成批量复制。
3) **更新嵌入函数**：
   - `embed_suffix()`：在返回前将 `E` 拼到 `action_time_emb` 之前，更新 `pad_masks/att_masks`。
   - `embed_alignment_suffix()`：在 suffix 最前放置 `E`，更新 `pad_masks/att_masks`，并重算 `block_diagonal_ranges`（E 单独一块；Dyn/InvDyn 互斥）。
4) **更新训练与对齐**：
   - 训练：合并 `L_total = L_action + λ_p·L_perc + λ_d·L_dyn + λ_i·L_inv`；确保 `E-Token` 有梯度并被优化。
   - 对齐：默认仅更新 `E-Token`；记录三项对齐损失曲线。
5) **配置与日志**：
   - 在 `training/config.py` 增加 `use_peft_prefix_token/peft_num_tokens/peft_lr` 与各损失权重。
   - 日志中单独汇报 `||E||`、梯度范数与学习率。

### 验证与测试
- **形状与掩码**：两路 expert 的输入长度各 +`peft_num_tokens`；`E` 的可见性与 block-diagonal 生效；VLM prefix 不受干扰。
- **梯度**：`E-Token` 的梯度非零；关闭 `E-Token` 优化时，损失变化符合预期。
- **收敛**：引入 `E-Token` 后，对齐阶段三项自监督损失均可下降。
- **兼容**：`use_peft_prefix_token=False` 时行为回退到无 E-Token 方案。
- **无 TTT 残留**：代码中无 `ttt_layer`、`L_ttt`、`TTTWithAdaptiveNorm` 等引用。

### 后续里程碑
- [ ] 移除所有 TTT 相关代码与配置
- [ ] 实现 E-Token 参数与注入逻辑
- [ ] 更新掩码以支持 E-Token
- [ ] 训练循环合并权重与日志
- [ ] 在线对齐仅更新 E-Token 的路径
- [ ] 小规模假数据回归测试 + 可视化
- [ ] （可选）扩展到 `peft_num_tokens>1` 与宽度不一致时的投影层

