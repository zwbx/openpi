# Analysis Documentation

本目录包含了对 OpenPI 项目的各种技术分析和集成计划文档。

## 文档列表

### 1. [Gradient Checkpointing Analysis](./gradient_checkpointing_analysis.md)
**梯度检查点机制分析**

- Gradient Checkpointing 的概念和权衡
- 完整的开启流程（从训练脚本到实际使用）
- 覆盖范围（哪些模块使用了它）
- 验证方法和生效条件

**关键发现**：确认了整个训练过程中 Gradient Checkpointing 都处于开启状态，节省显存但增加约 15-30% 训练时间。

---

### 2. [Attention Mask Analysis](./attention_mask_analysis.md)
**注意力掩码机制分析**

- Attention Mask 的作用和实现
- 在不同模型组件中的使用方式
- 掩码的生成和传播流程

---

### 3. [TTT Action Expert Integration Plan](./TTT_Action_Expert_Integration_Plan.md)
**TTT Action Expert 集成计划**

- Test-Time Training (TTT) 层的集成方案
- Action Expert 模型的架构设计
- 集成步骤和实现细节

---

### 4. [TTT vs Video DiT Comparison](./ttt_video_dit_comparison.md)
**TTT 层与 Video DiT 的对比分析**

- TTT (Test-Time Training) 层的技术特点
- Video DiT (Diffusion Transformer) 的技术特点
- 两种技术的对比和适用场景

---

## 文档创建时间

- `gradient_checkpointing_analysis.md`: 2025-10-14
- `attention_mask_analysis.md`: 2025-10-14
- `TTT_Action_Expert_Integration_Plan.md`: 2025-10-12
- `ttt_video_dit_comparison.md`: 2025-10-12

## 贡献

这些文档是在开发和调试过程中生成的技术分析，用于记录关键发现和设计决策。
