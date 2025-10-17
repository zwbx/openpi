# Codex Plan — OpenPI Self‑Alignment + TTT

更新时间: 2025-10-17

## 目标
- 联合优化：`L_total = L_action + λ_ttt·L_ttt + λ_p·L_perc + λ_d·L_dyn + λ_i·L_inv`。
  - `L_action`：主 action flow-matching 损失。
  - `L_ttt`：TTT 自监督重构/去噪损失（不再做内环/元学习更新）。
  - `L_perc`/`L_dyn`/`L_inv`：Alignment Expert 的三项自监督损失。
- 取消 TTT 元学习：去掉 per-sample/inner-loop（meta-learning）更新，将 TTT 的 `W1/b1` 作为常规可训练参数，仅由外层优化器更新。
- 在线自对齐：inference → buffer → 定频 align()，对齐阶段同样联合 `L_ttt + 三项自监督`，默认仅更新 TTT 参数。

## 当前进度
- 多专家骨干与前向
  - 训练：VLM + Action Expert +（可选）Alignment Expert 三路联合 attention；推理可分路执行。
  - 参考: `src/openpi/models_pytorch/gemma_pytorch.py:1`（`PaliGemmaWithExpertModel`）。
- Alignment Suffix 与掩码
  - `embed_alignment_suffix()` 将三任务输入拼接为一个 suffix，输出 pad/att masks 和 adarms cond；`make_att_2d_masks()` 支持 block‑diagonal。
  - 参考: `src/openpi/models_pytorch/pi0_pytorch.py:466`、`src/openpi/models_pytorch/pi0_pytorch.py:143`。
- TTT 层（Transformers 替换）
  - 按 `ttt_layer_positions` 创建 `TTTWithAdaptiveNorm`，与 AdaRMS、门控集成。
  - 参考: `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py:320`。
- TTT 单例（参数共享）
  - `TTTWithAdaptiveNorm` 按层单例：同 `layer_idx` 复用实例，Action/Alignment Expert 共享 W。
  - 参考: `src/openpi/models_pytorch/transformers_replace/models/gemma/ttt_with_gate.py:159`、`:189`。
- 在线自对齐闭环
  - `buffer`、`align()`、`sample_actions()` 已打通；当前 align 仅优化三项自监督（待加入 `L_ttt`）。
  - 参考: `src/openpi/models_pytorch/pi0_pytorch.py:20`、`:820`、`:927`。
- 验证脚本
  - `test_align_flow.py`、`test_ttt_training.py` 可用于快速自检。

## 最新决策更新（本次）
- 不再只由“三项自监督”驱动：引入 `L_ttt`（TTT 自监督重构损失）共同优化。
- 去掉 TTT 的 meta-learning/内环更新：取消 per‑sample 的 W 更新与 learnable per-sample LR 驱动，统一由外层优化器驱动参数学习。
- 训练阶段与在线对齐阶段，都合并 `L_ttt` 与三项对齐损失；推理阶段保持现状。

## 设计变更与落地
- `src/openpi/models_pytorch/transformers_replace/models/gemma/ttt_with_gate.py`
  - 新增开关：`meta_learning: bool = False`（默认关闭内环）。
  - 在 `ttt_batch_parallel()`：
    - `if not meta_learning:` 直接使用当前 `W1/b1` 计算 `Z1 = X1@W1 + b1`；不执行 `grad_W1/b1` 与 `W1_updated/b1_updated` 的内环更新与 dual/primal 分支。
    - 计算 `reconstruction_target = XV - XK`，并缓存 `last_recon_loss = mse(Z1, reconstruction_target)` 供上层汇总为 `L_ttt`。
  - 保留门控与 AdaRMS，对 forward 输出无功能性改变（仍为 `output = XQ + ln_fwd(Z1, ...)` 的残差形式）。
- `src/openpi/models_pytorch/gemma_pytorch.py`
  - 在联合前向完成后，遍历（去重）各层 `ttt_layer.last_recon_loss` 聚合为 `ttt_loss_total`。
  - 将 `ttt_loss_total` 作为附加返回值（或模型属性）传递给上层。
- `src/openpi/models_pytorch/pi0_pytorch.py`
  - 训练前向：合并损失 `L_total = L_action + λ_ttt·L_ttt + λ_p·L_perc + λ_d·L_dyn + λ_i·L_inv`；新增 `lambda_ttt` 等权重（从 config 读取，若无则默认 1.0）。
  - `align()`：把 `L_ttt` 合入对齐损失并仅更新 TTT 参数（维持当前只优化 TTT 的策略）。
  - 移除/忽略对 TTT 内环相关的 `ttt_base_lr/dual_form/keep_state` 行为（保留配置兼容但在 `meta_learning=False` 下不生效）。
- `src/openpi/training/config.py`
  - 新增对齐损失权重：`lambda_ttt/lambda_perc/lambda_dyn/lambda_inv`，训练日志打印各项标量与总损失。

## 待完成与差距
- 训练循环集成 `L_ttt` 与权重，日志与可视化。
- `next_obs_seq_len` 动态推断：移除硬编码（`pi0_pytorch.py:705`），从 `alignment_suffix_embs` 真实结构确定各任务范围。
- 测试增强：
  - “TTT 元学习关闭”验证：前向前后 `W1/b1` 不因内环改变（仅因外层优化器更新）。
  - “参数共享”验证：同层 `id(ttt_layer)` 一致；`L_ttt` 只计一次。
  - “block‑diagonal 正确性”验证：Dyn 与 InvDyn 不互看。
  - “加入 `L_ttt` 的收益”验证：对齐/训练中 loss 收敛对比。

## 下一步（建议顺序）
1) 在 `pi0_pytorch.py` 聚合并返回 `ttt_loss_total`，训练合并 `L_ttt`（加权）。
2) `ttt_with_gate.py` 增加 `meta_learning=False` 分支并返回/缓存 `last_recon_loss`。
3) `align()` 合并 `L_ttt`，仅优化 TTT 参数；增加最小日志。
4) 去掉 `next_obs_seq_len` 硬编码，动态范围计算。
5) 增加 2–3 个关键单测与最小训练配置（fake data）。

## 验证清单
- 前向/推理：维持现有行为；训练返回并合并 `L_ttt`。
- TTT 参数：关闭内环后，单次 forward 不改变 `W1/b1`；多个 step 后由外层优化器更新。
- 参数共享：`action_expert.layers[i].ttt_layer is alignment_expert.layers[i].ttt_layer`。
- 掩码：Dyn 与 InvDyn 互斥，Perception 可被两者访问。
- 训练/对齐：`L_total`、分项损失下降，显存/吞吐可接受。

## 参考文件
- `src/openpi/models_pytorch/gemma_pytorch.py:1`
- `src/openpi/models_pytorch/pi0_pytorch.py:143`
- `src/openpi/models_pytorch/pi0_pytorch.py:466`
- `src/openpi/models_pytorch/pi0_pytorch.py:559`
- `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py:320`
- `src/openpi/models_pytorch/transformers_replace/models/gemma/ttt_with_gate.py:159`
- `test_align_flow.py:1`
- `test_ttt_training.py:1`

---

说明：本计划用于跟踪“自对齐 + TTT（无元学习）联合优化”的实现与验证进度，随着训练集成与测试推进可逐步勾选与扩展。
