# 临时方案：随机 Embodiment Keys 生成

## ⚠️ 重要提示

**这是一个临时开发方案**，用于在多数据集 dataloader 完成之前，先让网络结构部分能够运行和测试。

**当多数据集 dataloader 完成后，必须移除这些临时代码！**

---

## 📁 文件位置

### 临时代码已添加到：
- **`src/openpi/models_pytorch/pi0_pytorch.py`** (行 20-87)
  - `_generate_random_embodiment_key()` - 生成单个随机 key
  - `_generate_random_embodiment_keys_batch()` - 生成 batch 的随机 keys
  - 在 `forward()` 方法中（行 786-796）自动生成 keys

### 测试文件：
- **`test_temp_embodiment_keys.py`** - 验证临时方案的功能

---

## 🚀 如何使用

### 1. 训练时自动生成

在 `PI0Pytorch.forward()` 中，如果 `embodiment_keys=None`，会自动生成随机 keys：

```python
# 在 forward() 中 (pi0_pytorch.py:786-796)
if embodiment_keys is None:
    batch_size = actions.shape[0]
    embodiment_keys = _generate_random_embodiment_keys_batch(
        batch_size=batch_size,
        same_embodiment=True  # 改为 False 测试多 embodiment
    )
```

**默认行为**：
- `same_embodiment=True`：整个 batch 使用相同的 embodiment（模拟单数据集训练）
- `same_embodiment=False`：每个样本随机选择 embodiment（模拟多数据集混合）

### 2. 测试两种场景

**场景 1：单数据集训练（默认）**

```python
# pi0_pytorch.py:794
same_embodiment=True  # 当前设置
```

运行训练后，日志会显示：
```
[TEMP] Generated random embodiment_keys: EmbodimentKey(robot_type='simpler', dof=7, ...)
```

**场景 2：多数据集混合训练**

```python
# pi0_pytorch.py:794
same_embodiment=False  # 修改为 False
```

这会在每个 batch 中混合不同的 embodiment，测试模型是否能正确处理多样性。

---

## ✅ 验证临时方案

运行测试脚本：

```bash
uv run python test_temp_embodiment_keys.py
```

**测试内容**：
- ✅ 随机 key 生成
- ✅ batch 生成（单/多 embodiment）
- ✅ 与 EmbodimentRegistry 集成
- ✅ Key 字段验证

---

## 🔍 生成的 Key 示例

### 单 embodiment batch：
```python
embodiment_keys = [
    EmbodimentKey(robot_type='simpler', dof=7, action_space='cartesian', ...),
    EmbodimentKey(robot_type='simpler', dof=7, action_space='cartesian', ...),
    EmbodimentKey(robot_type='simpler', dof=7, action_space='cartesian', ...),
    EmbodimentKey(robot_type='simpler', dof=7, action_space='cartesian', ...),
]
```

### 多 embodiment batch：
```python
embodiment_keys = [
    EmbodimentKey(robot_type='ur5', dof=6, action_space='joint', ...),
    EmbodimentKey(robot_type='franka', dof=7, action_space='cartesian', ...),
    EmbodimentKey(robot_type='simpler', dof=7, action_space='cartesian', ...),
    EmbodimentKey(robot_type='aloha', dof=14, action_space='joint', ...),
]
```

---

## 📊 支持的 Robot 配置

临时方案会从以下配置中随机选择：

| Robot Type | DOF | Action Space |
|------------|-----|--------------|
| simpler    | 7   | cartesian    |
| franka     | 7   | cartesian    |
| aloha      | 14  | joint        |
| ur5        | 6   | joint        |

其他字段（`state_space`, `coordinate_frame`, `image_crop`, `image_rotation`）也会随机生成。

---

## 🎯 当前可以做什么

使用这个临时方案，你现在可以：

### ✅ 开发和测试网络结构
- Prefix token bank 的参数初始化
- `get_embodiment_token()` 的逻辑
- EmbodimentRegistry 的注册和查询
- Token embedding 的维度验证

### ✅ 验证模型 Forward
```python
# 不需要提供 embodiment_keys
loss = model(observation, actions)  # 会自动生成

# 或者手动提供（测试用）
from openpi.models_pytorch.pi0_pytorch import _generate_random_embodiment_keys_batch
embodiment_keys = _generate_random_embodiment_keys_batch(batch_size=8)
loss = model(observation, actions, embodiment_keys=embodiment_keys)
```

### ✅ 测试两种训练场景
1. **单数据集**：`same_embodiment=True`（默认）
2. **多数据集混合**：`same_embodiment=False`

---

## 🔧 调试技巧

### 1. 查看生成的 keys

在代码中添加日志：

```python
# pi0_pytorch.py forward() 中
if embodiment_keys is None:
    embodiment_keys = _generate_random_embodiment_keys_batch(...)
    logging.info(f"[TEMP] Generated keys: {embodiment_keys}")  # 添加这行
```

### 2. 验证 Registry 注册

在 `get_embodiment_token()` 中：

```python
def get_embodiment_token(self, embodiment_keys, batch_size):
    # ... 获取 embodiment_ids ...

    print(f"Batch embodiment_ids: {embodiment_ids}")  # 添加调试
    print(f"Registry size: {len(self.embodiment_registry)}")
```

### 3. 固定随机种子（可复现）

如果需要固定的 embodiment 用于调试：

```python
# 修改 pi0_pytorch.py:792
embodiment_keys = _generate_random_embodiment_keys_batch(
    batch_size=batch_size,
    same_embodiment=True
)

# 改为使用固定的 key
from openpi.shared.embodiment_config import EmbodimentKey
fixed_key = EmbodimentKey(
    robot_type='simpler',
    dof=7,
    action_space='cartesian',
    state_space='cartesian',
    coordinate_frame='base',
    image_crop=False,
    image_rotation=False,
    image_flip=False,
    camera_viewpoint_id='default'
)
embodiment_keys = [fixed_key] * batch_size
```

---

## 🗑️ 何时移除临时代码

### 时机：完成多数据集 dataloader 后

当你完成以下工作时，应该移除临时代码：

1. ✅ 在 `Dataset.__getitem__()` 中注入真实的 `embodiment_keys`
2. ✅ 在 `DataConfig` 中配置 `embodiment_config`
3. ✅ DataLoader 返回包含 `embodiment_keys` 的 batch
4. ✅ 训练循环能正确传递 `embodiment_keys`

### 需要删除的代码：

**文件：`src/openpi/models_pytorch/pi0_pytorch.py`**

删除以下部分：

```python
# 删除 1: 临时函数定义 (行 20-87)
def _generate_random_embodiment_key(...):
    ...

def _generate_random_embodiment_keys_batch(...):
    ...

# 删除 2: forward() 中的自动生成 (行 786-796)
if embodiment_keys is None:
    embodiment_keys = _generate_random_embodiment_keys_batch(...)
    logging.debug(f"[TEMP] Generated random embodiment_keys...")
```

**文件：根目录**

删除测试文件：
```bash
rm test_temp_embodiment_keys.py
rm test_temp_embodiment_keys_README.md
rm TEMP_EMBODIMENT_KEYS_README.md  # 本文件
```

---

## 📋 Checklist

开发网络结构时：
- [x] 临时代码已添加到 `pi0_pytorch.py`
- [x] 测试脚本可以运行
- [x] `same_embodiment=True` 默认启用（模拟单数据集）
- [ ] 使用 `same_embodiment=False` 测试多数据集场景
- [ ] 验证 prefix token bank 工作正常
- [ ] 验证 EmbodimentRegistry 自动注册

完成多数据集 dataloader 后：
- [ ] Dataset.__getitem__() 注入真实 embodiment_keys
- [ ] DataConfig 配置 embodiment_config
- [ ] DataLoader 返回 embodiment_keys
- [ ] 训练循环传递 embodiment_keys
- [ ] **删除所有临时代码**
- [ ] 验证真实数据流工作正常

---

## 🎓 总结

**这个临时方案让你能够：**

✅ **现在就开始开发网络结构**，不用等 dataloader 完成
✅ **测试 embodiment token bank** 的所有逻辑
✅ **验证模型 forward** 能正确处理 embodiment_keys
✅ **模拟单/多数据集训练** 场景

**记住：**
- 这只是临时开发方案
- 真实训练时必须使用从 dataloader 传入的 embodiment_keys
- 完成后删除所有临时代码

---

## 📞 问题排查

### Q: 为什么要生成随机 keys 而不是固定一个？

A: 因为要测试：
- Registry 的自动注册逻辑
- 多 embodiment 场景下的 token 选择
- Batch 中不同样本使用不同 keys 的情况

### Q: same_embodiment 应该设置为 True 还是 False？

A:
- **开发初期**：用 `True`（简单场景，方便调试）
- **测试阶段**：用 `False`（验证多 embodiment 逻辑）
- **真实训练**：由 dataloader 提供真实 keys

### Q: 生成的 keys 会影响训练结果吗？

A: 会的！因为：
- 不同的 embodiment_key → 不同的 w_index
- 不同的 w_index → 不同的 prefix token
- 但这只是临时方案，真实训练会使用数据集的真实配置

---

**祝开发顺利！记得完成后删除临时代码！** 🚀
