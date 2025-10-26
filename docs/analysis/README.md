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

### 5. [Self-Alignment Implementation Plan](./self_alignment_implementation_plan.md)
**自对齐VLA完整实现计划**

- 核心思想：解耦 embodiment-agnostic 和 embodiment-relevant 表征
- Alignment Experts 架构设计（Inverse Dynamics, Dynamics, Perception）
- 对比学习策略（正样本/负样本构造）
- 自对齐训练流程（两阶段训练）
- 详细的实现路径和时间线（7个Phase）

**关键创新**：使用 TTT 层参数作为 embodiment context W，通过 play data 实现零样本迁移

---

### 6. [Data Loading Pipeline Analysis](./data_loading_pipeline_analysis.md)
**数据载入流程完整分析**

- 配置系统架构（TrainConfig → DataConfigFactory → DataConfig）
- 配置文件解析流程（tyro CLI + 命令行覆盖）
- 完整数据加载 Pipeline（从配置到训练循环）
- 数据集创建流程（LeRobot / RLDS 两种格式）
- 四阶段数据转换（Repack → Data Transform → Normalize → Model Transform）
- **数据增强机制详解**（在模型 forward 中应用，而非 DataLoader）
- 关键代码路径追踪和示例解析

**核心价值**：理解如何读取配置、创建数据集、应用转换链，以及如何添加自定义数据集

**重要发现**：
- ⚠️ 数据增强不在 DataLoader 中，而是在模型 forward pass 中动态应用
- 几何增强（Random Crop 95%, Random Rotation ±5°）仅对非手腕相机应用
- 颜色增强（Brightness, Contrast, Saturation）对所有相机应用
- 位置：`src/openpi/models_pytorch/preprocessing_pytorch.py`

---

## 文档创建时间

- `gradient_checkpointing_analysis.md`: 2025-10-14
- `attention_mask_analysis.md`: 2025-10-14
- `TTT_Action_Expert_Integration_Plan.md`: 2025-10-12
- `ttt_video_dit_comparison.md`: 2025-10-12
- `self_alignment_implementation_plan.md`: 2025-10-14
- `data_loading_pipeline_analysis.md`: 2025-10-26

## 贡献

这些文档是在开发和调试过程中生成的技术分析，用于记录关键发现和设计决策。
