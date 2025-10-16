# PEFT Insertion Strategies for Multi-Embodiment Adaptation

## 问题背景

在实现 multi-embodiment adaptation 时，核心挑战是：
- 主网络需要根据不同的 embodiment_id 使用不同的 PEFT 参数
- 不同的 PEFT 方法（TTT, LoRA, Prefix, Adapter）插入位置和机制完全不同
- 希望能通过配置文件方便切换 PEFT 方法，而不需要每次手工修改主网络代码

**核心问题**: 如何在主网络里控制不同 PEFT 的插入，且主网络的 forward 过程都要修改？

## 设计原则

### 分层架构

```
┌─────────────────────────────────────────────────────┐
│  Configuration Layer (配置层)                        │
│  - 选择使用哪种 PEFT (TTT/LoRA/Prefix/Adapter)      │
│  - 指定哪些层需要 adaptation                         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Parameter Management Layer (参数管理层)             │
│  - EmbodimentRegistry: config → W index             │
│  - 为每个 embodiment 分配独立的参数空间              │
│  - 在 forward 时根据 embodiment_id 选择参数          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Method-Specific Insertion Layer (方法特定插入层)    │
│  - TTT: 修改 attention layer 内部                    │
│  - LoRA: wrap Linear layers                         │
│  - Prefix: 修改 model 输入 pipeline                  │
│  - Adapter: 在 FFN 后插入新层                        │
└─────────────────────────────────────────────────────┘
```

### 核心洞察

**不应该统一插入机制，而是统一选择和管理机制**

- **统一的部分**: 配置接口、参数管理、embodiment_id 传递
- **不统一的部分**: 插入位置、参数形状、forward 逻辑

## 四种实现方案

### 方案 1: Hook-Based Insertion (推荐)

**核心思路**: 利用 PyTorch 的 forward hook 机制，不修改主网络代码

```python
class TTTPlugin(PEFTPlugin):
    def setup_model(self, model):
        # 不修改 forward，而是注册 hook
        for layer_idx in self.config['target_layers']:
            layer = model.model.layers[layer_idx]

            # 在 attention 的 forward 前后插入逻辑
            layer.self_attn.register_forward_pre_hook(
                self._make_pre_hook(layer_idx)
            )
            layer.self_attn.register_forward_hook(
                self._make_post_hook(layer_idx)
            )

        return model

    def _make_pre_hook(self, layer_idx):
        def hook(module, inputs):
            # 获取 embodiment_id（从 inputs 或全局状态）
            embodiment_id = self._get_current_embodiment_id()

            # 设置这一层要用的 W 参数
            module._current_W = self.W_params[embodiment_id][layer_idx]

            return inputs
        return hook
```

**优点**:
- ✅ 完全不需要修改主网络的 forward 代码
- ✅ 可以动态开关（remove_hook）
- ✅ 多个 PEFT 可以叠加使用
- ✅ 符合 PyTorch 的设计哲学

**缺点**:
- ❌ Hook 的执行顺序需要注意
- ❌ Embodiment_id 需要通过某种方式传递
- ❌ 调试时 hook 内的错误不够直观

**适用场景**:
- PEFT 需要在特定层的 forward 前后注入逻辑（TTT, LoRA）
- 希望保持主网络代码不变
- 需要支持多种 PEFT 叠加

---

### 方案 2: Wrapper Module Pattern

**核心思路**: 用一个 wrapper 包装整个模型，在外层控制 embodiment_id 传递

```python
class EmbodimentConditionedModel(nn.Module):
    def __init__(self, base_model, peft_plugin):
        super().__init__()
        self.base_model = base_model
        self.peft_plugin = peft_plugin

        # Plugin 修改 base_model 的结构
        self.peft_plugin.setup_model(self.base_model)

    def forward(self, input_ids, embodiment_id, **kwargs):
        # 设置全局状态（让 hooks/modules 知道当前的 embodiment_id）
        self.peft_plugin.set_current_embodiment(embodiment_id)

        # 调用原始 forward
        output = self.base_model(input_ids, **kwargs)

        # 清理状态
        self.peft_plugin.clear_current_embodiment()

        return output
```

**优点**:
- ✅ 主网络代码完全不改
- ✅ Embodiment_id 传递清晰
- ✅ 容易切换不同的 PEFT
- ✅ 可以在 wrapper 层添加额外逻辑（logging, metrics）

**缺点**:
- ❌ 增加了一层包装
- ❌ 需要管理全局状态
- ❌ Forward signature 改变（增加了 embodiment_id 参数）

**适用场景**:
- 需要清晰的接口来传递 embodiment_id
- 希望在外层统一管理 PEFT 状态
- 可以接受 forward 签名的修改

---

### 方案 3: Thread-Local State (最灵活)

**核心思路**: 使用 thread-local 变量传递 embodiment_id，完全解耦

```python
import threading

class EmbodimentContext:
    """线程本地的 embodiment context"""
    _local = threading.local()

    @classmethod
    def set(cls, embodiment_id):
        cls._local.embodiment_id = embodiment_id

    @classmethod
    def get(cls):
        return getattr(cls._local, 'embodiment_id', None)

    @classmethod
    def clear(cls):
        cls._local.embodiment_id = None

# 在 forward 开始时设置
def forward(self, input_ids, embodiment_id, **kwargs):
    EmbodimentContext.set(embodiment_id)
    try:
        output = self.base_model(input_ids, **kwargs)
    finally:
        EmbodimentContext.clear()
    return output

# 在任何深层的 module 中都能获取
class TTTAttentionMultiW(nn.Module):
    def forward(self, hidden_states):
        embodiment_id = EmbodimentContext.get()
        W = self.W_params[embodiment_id]
        # ... 使用 W 进行计算
```

**优点**:
- ✅ 完全解耦，不需要修改任何 forward 签名
- ✅ 深层 module 直接访问 embodiment_id
- ✅ 支持多线程（每个线程独立）
- ✅ 可以用 context manager 优雅地管理生命周期

**缺点**:
- ❌ 全局状态，可能不够显式
- ❌ 调试时不太直观（隐式依赖）
- ❌ 如果忘记 clear 可能导致状态泄漏

**适用场景**:
- 网络层级很深，传递参数不方便
- 希望保持所有 forward 签名不变
- 能接受隐式的全局状态

**最佳实践**: 使用 context manager 确保清理

```python
from contextlib import contextmanager

class EmbodimentContext:
    @classmethod
    @contextmanager
    def set_context(cls, embodiment_id):
        cls.set(embodiment_id)
        try:
            yield
        finally:
            cls.clear()

# 使用
with EmbodimentContext.set_context(embodiment_id):
    output = model(input_ids, **kwargs)
```

---

### 方案 4: Monkey-Patching Forward (最直接但不优雅)

**核心思路**: 直接替换模型的 forward 方法

```python
class PrefixPlugin(PEFTPlugin):
    def setup_model(self, model):
        # 保存原始 forward
        original_forward = model.forward

        # 创建新的 forward
        def new_forward(input_ids, embodiment_id, **kwargs):
            # Prepend prefix tokens
            prefix = self.prefix_embeddings[embodiment_id]

            # 修改 input_ids 或 embeddings
            # ...

            # 调用原始 forward
            return original_forward(modified_input, **kwargs)

        # 替换 forward
        model.forward = new_forward

        return model
```

**优点**:
- ✅ 最直接，想怎么改就怎么改
- ✅ 适合需要大幅修改 forward 逻辑的 PEFT（如 Prefix）
- ✅ 实现简单

**缺点**:
- ❌ 不优雅，破坏了封装
- ❌ 多个 PEFT 叠加会很乱（forward 被多次替换）
- ❌ 难以恢复原始行为
- ❌ 可能破坏 JIT/TorchScript

**适用场景**:
- PEFT 需要根本性改变 forward 逻辑（如 Prefix Tuning）
- 只使用一种 PEFT，不需要叠加
- 原型开发，快速验证想法

---

## 推荐方案: Hook + Wrapper 组合

结合方案 1 和方案 2 的优点：

```python
# 1. 主网络完全不改
class PI0PyTorch(nn.Module):
    def forward(self, input_ids, **kwargs):
        # 原始逻辑，不需要知道 embodiment
        return self.model(input_ids, **kwargs)

# 2. Wrapper 负责 embodiment_id 传递
class EmbodimentConditionedPI0(nn.Module):
    def __init__(self, base_model, peft_plugin):
        super().__init__()
        self.base_model = base_model
        self.plugin = peft_plugin

        # Plugin 通过 hook 修改 base_model
        self.plugin.setup_model(self.base_model)

    def forward(self, input_ids, embodiment_id, **kwargs):
        # 方案 A: 用 context manager
        with EmbodimentContext.set_context(embodiment_id):
            return self.base_model(input_ids, **kwargs)

        # 或方案 B: 用 plugin 的 API
        # self.plugin.set_current_embodiment(embodiment_id)
        # output = self.base_model(input_ids, **kwargs)
        # self.plugin.clear_current_embodiment()
        # return output

# 3. Plugin 通过 hook 在需要的地方获取 embodiment_id
class TTTPlugin(PEFTPlugin):
    def setup_model(self, model):
        for idx in self.config['target_layers']:
            layer = model.model.layers[idx]

            # 注册 hook，在 attention 计算时选择 W
            layer.self_attn.register_forward_pre_hook(
                lambda module, inputs: self._inject_W(module, inputs)
            )

    def _inject_W(self, module, inputs):
        embodiment_id = EmbodimentContext.get()
        # 设置 module 的 W 参数
        module._active_W = self.W_params[embodiment_id]
        return inputs
```

### 为什么推荐这个组合？

1. **主网络不改**: `PI0PyTorch` 保持原样，符合开闭原则
2. **清晰的接口**: `EmbodimentConditionedPI0` 明确地接受 `embodiment_id`
3. **灵活的插入**: Hook 机制让不同 PEFT 自由地在需要的地方注入逻辑
4. **解耦的传递**: Thread-local 让深层 module 能获取 embodiment_id，无需层层传递
5. **易于切换**: 更换 PEFT 只需要换 plugin，其他都不变

---

## 不同 PEFT 方法的具体插入示例

### TTT (Test-Time Training)

**插入位置**: Attention layer 内部，替换或修改 QKV 计算

```python
class TTTPlugin(PEFTPlugin):
    def setup_model(self, model):
        for idx in self.config['target_layers']:
            attn = model.model.layers[idx].self_attn

            # 方式 1: 替换整个 attention module
            model.model.layers[idx].self_attn = TTTAttentionMultiW(
                attn, self.registry
            )

            # 方式 2: 用 hook 注入 W 参数
            attn.register_forward_pre_hook(self._inject_W_hook)
```

### LoRA (Low-Rank Adaptation)

**插入位置**: 包装 Linear layers

```python
class LoRAPlugin(PEFTPlugin):
    def setup_model(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._should_adapt(name):
                # 替换 Linear 为 LoRA 版本
                parent = self._get_parent(model, name)
                setattr(parent, name.split('.')[-1],
                        LoRALinearMultiW(module, self.registry, rank=self.config['rank']))
```

### Prefix Tuning

**插入位置**: Model 输入层，prepend 学习到的 prefix tokens

```python
class PrefixPlugin(PEFTPlugin):
    def setup_model(self, model):
        # 添加 prefix embeddings
        num_embodiments = len(self.registry)
        prefix_len = self.config['prefix_length']
        hidden_size = model.config.hidden_size

        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_embodiments, prefix_len, hidden_size)
        )

        # Monkey-patch forward（Prefix 确实需要改 forward）
        original_forward = model.forward

        def forward_with_prefix(input_ids, embodiment_id=None, **kwargs):
            if embodiment_id is None:
                embodiment_id = EmbodimentContext.get()

            # 获取 prefix
            prefix = self.prefix_embeddings[embodiment_id].unsqueeze(0)

            # 修改 inputs_embeds
            embeddings = model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prefix, embeddings], dim=1)

            # 调用原始 forward
            return original_forward(inputs_embeds=inputs_embeds, **kwargs)

        model.forward = forward_with_prefix
```

### Adapter

**插入位置**: FFN 后添加 bottleneck layers

```python
class AdapterPlugin(PEFTPlugin):
    def setup_model(self, model):
        for idx in self.config['target_layers']:
            layer = model.model.layers[idx]

            # 在 FFN 后添加 adapter
            original_forward = layer.forward

            def forward_with_adapter(hidden_states, *args, **kwargs):
                # 原始层的输出
                output = original_forward(hidden_states, *args, **kwargs)

                # Adapter: down_proj -> ReLU -> up_proj
                embodiment_id = EmbodimentContext.get()
                adapter_output = self.adapters[embodiment_id][idx](output)

                return output + adapter_output  # 残差连接

            layer.forward = forward_with_adapter
```

---

## 关键设计决策

### 1. Embodiment ID 传递方式

| 方式 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| 函数参数 | 显式、类型安全 | 需要修改所有 forward 签名 | ⭐⭐⭐ |
| Thread-local | 不修改签名、深层可访问 | 隐式、调试困难 | ⭐⭐⭐⭐ |
| 全局变量 | 简单 | 线程不安全、状态混乱 | ⭐ |
| Module 属性 | 直接 | 需要手动传播 | ⭐⭐ |

**推荐**: Thread-local + Context Manager

### 2. PEFT 参数管理

所有 PEFT 方法都应该使用统一的参数管理：

```python
class PEFTPlugin:
    def __init__(self, registry, config):
        self.registry = registry
        num_embodiments = len(registry)

        # 为每个 embodiment 创建参数
        self.params = nn.ParameterList([
            self._create_params_for_embodiment(i)
            for i in range(num_embodiments)
        ])

    def get_params(self, embodiment_id):
        """统一的参数获取接口"""
        return self.params[embodiment_id]
```

### 3. 配置驱动的 PEFT 选择

```yaml
# config.yaml
peft:
  method: "ttt"  # or "lora", "prefix", "adapter"

  # TTT 特定配置
  ttt:
    target_layers: [10, 11, 12]
    num_mini_batch_per_forward: 1

  # LoRA 特定配置
  lora:
    rank: 8
    alpha: 16
    target_modules: ["q_proj", "v_proj"]

  # Prefix 特定配置
  prefix:
    prefix_length: 10

  # Adapter 特定配置
  adapter:
    bottleneck_size: 64
```

加载方式：

```python
# 从配置创建 plugin
config = load_config("config.yaml")
plugin = create_peft_plugin(config['peft'], registry)

# Setup model
model = plugin.setup_model(base_model)
```

---

## 实现路线图

### Phase 1: 基础框架
- [x] `EmbodimentConfig` 和 `EmbodimentRegistry`
- [ ] `PEFTPlugin` 抽象基类
- [ ] `EmbodimentContext` (Thread-local)
- [ ] `EmbodimentConditionedModel` wrapper

### Phase 2: TTT Implementation
- [ ] `TTTPlugin` 实现
- [ ] Multi-W TTT Attention module
- [ ] Hook-based injection
- [ ] 测试和验证

### Phase 3: 其他 PEFT 方法
- [ ] `LoRAPlugin`
- [ ] `PrefixPlugin`
- [ ] `AdapterPlugin`

### Phase 4: 配置系统
- [ ] YAML 配置解析
- [ ] Plugin registry 和自动发现
- [ ] 配置验证

### Phase 5: 训练集成
- [ ] 修改 `train_pytorch.py` 支持 multi-embodiment
- [ ] DataLoader 提供 embodiment_id
- [ ] Loss 计算和反向传播
- [ ] Checkpoint 保存/加载

---

## 总结

### 核心问题
> 如何在主网络里控制不同 PEFT 的插入，且主网络的 forward 过程都要修改？

### 答案
**主网络的 forward 不需要修改！**

通过以下机制实现：
1. **Wrapper** 在外层接收 embodiment_id 并设置 context
2. **Hook** 让 PEFT 在需要的地方自动注入逻辑
3. **Thread-local** 让深层 module 能获取 embodiment_id
4. **Plugin** 让每个 PEFT 自己决定如何插入

### 推荐架构

```
用户代码
  ↓
EmbodimentConditionedModel (wrapper, 接收 embodiment_id)
  ↓ set context
EmbodimentContext.set_context(embodiment_id)
  ↓
Base Model Forward (不变)
  ↓
Hooks 在各层触发
  ↓ get context
PEFT Modules 使用 EmbodimentContext.get() 获取 embodiment_id
  ↓
选择对应的参数并计算
```

**关键优势**:
- ✅ 主网络代码完全不变
- ✅ 易于切换 PEFT 方法（配置文件控制）
- ✅ 支持多种 PEFT 叠加
- ✅ 清晰的接口和职责分离
