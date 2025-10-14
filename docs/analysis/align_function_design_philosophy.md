# Align 函数设计哲学

**日期**: 2025-10-14
**状态**: 设计理念确认

---

## 核心思想

模型有三种使用模式，每种模式的目的和机制完全不同：

### 1. 预训练 (forward) - 源环境学习阶段

**目的**: 在源环境数据上学习通用策略和 embodiment 编码能力

**优化对象**:
- VLM (PaliGemma)
- Action Expert
- Alignment Expert
- TTT parameters (W1, b1)

**数据来源**: 源环境的标注数据（observations + actions + next_obs_features）

**关键点**: 学习两种能力
1. 如何执行任务（策略本身）
2. 如何将 embodiment 特性编码到 TTT 参数中

---

### 2. 在线适应 (align) - 新环境适应阶段

**目的**: 在新环境中，通过无监督数据调整 embodiment context，使模型适应新的机器人/环境

**核心机制**:
```
新环境无监督数据 → Alignment Expert → 反向传播 → 只更新 TTT 参数
```

**关键设计决策**:

#### ✅ 优化对象
- **只优化**: TTT parameters (W1, b1)
- **冻结**: VLM, Action Expert, Alignment Expert 的所有其他参数

#### ✅ Forward 路径
- **只使用**: VLM prefix + Alignment Expert
- **不使用**: Action Expert（不参与 forward，不参与 backward）

**为什么不用 Action Expert？**
- Alignment Expert 的三个自监督任务完全独立，不需要 Action Expert 的输出
- 节省计算资源
- Action Expert 会通过 TTT 参数共享间接受益

#### ✅ 数据来源
- **无监督 play data**: 通过 gripper random 或随机探索获得
- **不需要任务标注**: 不需要知道"正确"的动作是什么
- **自监督信号**:
  - Perception: 当前观察的状态信息
  - Dynamics: 观察的时序变化
  - Inverse Dynamics: 观察变化隐含的动作信息

#### ✅ 损失函数
- **只使用 Alignment losses**:
  - Perception Loss
  - Dynamics Loss
  - Inverse Dynamics Loss
- **不使用 Action Loss**: 因为 Action Expert 没有参与

---

### 3. 测试 (sample_actions) - 新环境推理阶段

**目的**: 在新环境中执行任务，预测动作

**使用对象**:
- VLM prefix
- Action Expert（使用更新后的 TTT 参数）

**关键机制**:
```
观察 → VLM → Action Expert (with updated TTT) → 预测动作
                     ↑
                 TTT Layer 调制
                 (使用 align 更新后的参数)
```

**为什么 Action Expert 能适应新环境？**
- 虽然 Action Expert 本身没有在 align 时训练
- 但它内部的 TTT layer 参数被更新了
- TTT layer 调制 hidden states，改变了 Action Expert 的行为
- 通过 Singleton pattern，Action Expert 和 Alignment Expert 共享同一个 TTT instance

---

## 核心洞察

### 洞察 1: TTT 作为 Embodiment Context Adapter

```
预训练好的策略（冻结）
         ↓
    [TTT Layer] ← 可调整的 embodiment adapter
         ↓
    调制后的行为
         ↓
    适应新环境
```

**关键理解**:
- TTT 参数 (W1, b1) 编码了 **embodiment context**
- 包括：机器人的动作空间、坐标系、自由度、物理特性等
- 调整这些参数 = 告诉模型"你现在在一个不同的机器人身上"
- 策略本身不变，但通过 TTT 调制后的行为适应新环境

### 洞察 2: Alignment Expert 作为 Embodiment Sensor

```
新环境数据
    ↓
Alignment Expert (3 个自监督任务)
    ↓
"感知"新环境的 embodiment 特性
    ↓
通过梯度反向传播到 TTT 参数
    ↓
TTT 参数调整以适应新 embodiment
```

**Alignment Expert 的三个任务**:
1. **Perception**: 理解新环境的状态空间结构
2. **Dynamics**: 理解新环境的动力学规律
3. **Inverse Dynamics**: 理解新环境的动作-效果映射

这三个任务共同"探测"新环境的 embodiment 特性

### 洞察 3: 参数共享的巧妙设计

```
Action Expert ←──┐
                 │
            [TTT Layer] (singleton)
                 │
Alignment Expert ←──┘
```

**为什么这样设计？**
- Alignment Expert 优化 TTT 参数时，Action Expert 自动获得更新
- 因为它们共享同一个 TTT instance（不是复制，是同一个对象）
- 不需要显式地将 Alignment 的更新"传递"给 Action Expert
- 推理时，Action Expert 的 TTT layer 已经是更新后的版本

---

## 设计原则

### 原则 1: 分离策略和 Embodiment
- **策略**: 如何完成任务的通用知识（冻结）
- **Embodiment**: 如何在特定机器人上执行（可调）
- 迁移到新环境时，只调整 embodiment，不改变策略

### 原则 2: 自监督适应
- 不需要新环境的任务标注
- 使用机器人与环境的自然交互数据
- 通过内在的一致性约束（perception, dynamics, inverse dynamics）学习

### 原则 3: 最小化推理成本
- Align 时只用 Alignment Expert（可以离线/异步进行）
- 测试时只用 Action Expert（推理快速）
- 不需要在推理时运行 Alignment Expert

### 原则 4: 参数高效
- 只优化 TTT 参数（可能只有几 MB）
- 不改动大模型的主体参数（可能有几 GB）
- 适合在线/边缘设备部署

---

## 工作流程概览

```
┌──────────────────────────────────────────────────┐
│ Phase 1: 预训练 (源环境)                          │
│                                                  │
│ 数据: 源环境标注数据                              │
│ 优化: 所有参数                                    │
│ 目标: 学习策略 + embodiment 编码能力              │
└──────────────────────────────────────────────────┘
                    ↓
                保存模型
                    ↓
┌──────────────────────────────────────────────────┐
│ Phase 2: 在线适应 (新环境)                        │
│                                                  │
│ 数据: 新环境无监督 play data                      │
│      (gripper random / 探索数据)                  │
│                                                  │
│ Forward: VLM + Alignment Expert only             │
│ Backward: 只更新 TTT 参数                         │
│                                                  │
│ 循环: 多次调用 align() 直到 loss 收敛             │
└──────────────────────────────────────────────────┘
                    ↓
            TTT 参数更新完成
                    ↓
┌──────────────────────────────────────────────────┐
│ Phase 3: 测试 (新环境)                            │
│                                                  │
│ Forward: VLM + Action Expert                     │
│         (使用更新后的 TTT 参数)                    │
│                                                  │
│ 输出: 适应新环境的动作预测                         │
└──────────────────────────────────────────────────┘
```

---

## 与传统迁移学习的区别

### 传统 Fine-tuning
```
新环境数据 → 更新整个模型 → 推理
```
**问题**:
- 需要大量标注数据
- 容易过拟合
- 计算成本高
- 可能遗忘源环境知识

### Self-Alignment (我们的方法)
```
新环境无监督数据 → 只更新 TTT (embodiment adapter) → 推理
```
**优势**:
- 不需要标注数据（自监督）
- 策略保持不变（不会遗忘）
- 只调整 embodiment 适配层
- 参数高效（只优化少量参数）
- 可以快速适应多个环境（切换 TTT 参数即可）

---

## 理论基础

### 为什么这个方法可行？

#### 假设 1: 策略的可迁移性
- 任务的高层策略（如"抓取物体"）在不同机器人上是通用的
- 只有底层的 embodiment 细节（如关节角度范围）不同

#### 假设 2: TTT 的表达能力
- TTT 参数 (W1, b1) 可以有效编码 embodiment 信息
- 通过线性变换 + 偏置，可以调制 hidden states 以适应不同 embodiment

#### 假设 3: 自监督信号的充分性
- Perception, Dynamics, Inverse Dynamics 三个任务提供足够的信号
- 这些信号反映了 embodiment 的关键特性
- 优化这些任务等价于学习新的 embodiment context

---

## 关键技术细节（待实现）

### 1. 参数管理
- 只启用 TTT 参数的 `requires_grad`
- 保持其他参数 `requires_grad = False`
- **不改变模型的 training/eval mode**（保持 eval mode）

### 2. Forward 路径
- Alignment Expert only: `inputs_embeds = [prefix, None, alignment_suffix]`
- Action Expert 不参与

### 3. Loss 计算
- 只使用 Alignment losses（perception + dynamics + inverse_dynamics）
- 使用 `reduction="mean"` 得到标量 loss

### 4. 数据格式
- `observation`: 标准的 Observation 对象
- `actions`: [batch, action_horizon, action_dim] - 来自 play data
- `next_obs_features`: [batch, feature_dim] - 使用 DINO 等提取

### 5. 优化策略
- 使用 Adam optimizer
- learning_rate: 默认 1e-4
- 支持 early stopping (loss_threshold)
- 支持自定义 loss_weights

---

## 预期效果

### 成功的标志
1. **Alignment loss 下降**: 说明模型理解了新环境的 embodiment
2. **Action prediction 提升**: 在新环境的任务上性能提升
3. **快速适应**: 只需少量 play data（10-100 条轨迹）
4. **零样本泛化**: 适应后的模型在新环境的未见任务上也能工作

### 失败的情况
1. Alignment loss 不下降 → TTT 参数梯度可能有问题
2. Alignment loss 下降但 action 性能不提升 → TTT 和 Action Expert 的连接可能有问题
3. 需要大量数据才能适应 → 自监督信号可能不够强

---

## 总结

**Align 的本质**: 通过自监督学习，将新环境的 embodiment 特性编码到 TTT 参数中

**关键机制**:
- Alignment Expert 作为"传感器"，感知新环境
- TTT 作为"适配器"，调制模型行为
- Singleton pattern 确保参数共享

**设计优势**:
- 无需标注数据
- 参数高效
- 不遗忘源知识
- 快速适应

**最后更新**: 2025-10-14
