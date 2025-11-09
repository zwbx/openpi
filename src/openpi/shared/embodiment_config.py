"""
Embodiment Configuration System

用于标识和管理不同的 embodiment configurations，为每个 configuration 分配独立的 W 参数。

核心概念:
- EmbodimentKey: 唯一标识一个 embodiment (NamedTuple, immutable, hashable)
- EmbodimentConfig: 完整的配置信息 (dataclass, 包含所有字段)
- EmbodimentRegistry: 管理 config -> W index 的映射

设计原则:
1. 影响 embodiment context 的因素：
   - Robot configuration (type, DOF)
   - Action/State space (cartesian vs joint)
   - Coordinate frame (base vs world)
   - Image layout transformations (crop, rotation, flip)
   - Camera viewpoint

2. 不影响 embodiment context 的因素：
   - 外观增强 (color jitter, brightness, contrast)
   这些只改变外观，不改变 observation-action-state 的对应关系

使用方式:
    # 创建 registry
    registry = EmbodimentRegistry(mode="auto")

    # 创建配置
    config = EmbodimentConfig(
        robot_type="franka",
        image_crop=True,
        image_rotation=False,
    )

    # 获取 W index
    w_index = registry.get_or_register(config)

    # 使用 w_index 选择对应的 W 参数
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, NamedTuple

logger = logging.getLogger("openpi")


class EmbodimentKey(NamedTuple):
    """
    轻量级的 Embodiment 唯一标识

    使用 NamedTuple 的优点:
    1. 性能好：比 dataclass 更轻量
    2. 天生 immutable 和 hashable，可以作为 dict key
    3. 内存占用小
    4. 创建速度快

    字段说明:
    - dataset_name: 数据集名称
    - robot_type: 机器人类型 (如 "franka", "ur5")
    - camera_view: 相机视角（tuple，如 ("front", "left")）
    - dof: 自由度数量
    - action_space: 动作空间类型 ("cartesian" | "joint")
    - absolute_mode: 是否使用 absolute action space
    - state_space: 状态空间类型 ("cartesian" | "joint")
    - coordinate_frame: 坐标系 ("base" | "world" | "camera")
    - image_crop: 是否做了裁剪
    - image_rotation: 是否做了旋转
    - image_flip: 是否做了翻转
    """
    # 必填字段
    dataset_name: str

    # Robot configuration
    robot_type: str = "franka"
    camera_view: tuple[str, ...] = ("front", "left")
    dof: int = 7

    # Action/State space
    action_space: str = "cartesian"
    absolute_mode: bool = False
    state_space: str = "cartesian"
    coordinate_frame: str = "base"

    # Image layout transformations
    image_crop: bool | None = None
    image_rotation: bool | None = None
    image_flip: bool | None = None


# 为了兼容性，创建别名
EmbodimentConfig = EmbodimentKey
BaseEmbodimentConfig = EmbodimentKey


class EmbodimentRegistry:
    """
    管理所有 embodiment configurations 和 W index 的映射

    职责:
    1. 维护 tuple -> W index 的映射（直接使用 tuple 作为 key，性能最优）
    2. 自动注册新的 embodiment configurations
    3. 保存/加载 registry 到文件

    使用方式:
        registry = EmbodimentRegistry(mode="auto")

        # 方式1：直接传递 tuple（推荐，性能最好）
        key = ("bridge", "franka", ("front", "left"), 7, "cartesian", False, ...)
        w_index = registry.get_or_register(key)

        # 方式2：传递 EmbodimentKey（会自动转换为 tuple）
        key = EmbodimentKey(dataset_name="bridge", robot_type="franka", ...)
        w_index = registry.get_or_register(key)
    """

    def __init__(self, mode: str = "auto"):
        """
        初始化 registry

        Args:
            mode: 注册模式
                - "auto": 自动注册新配置（训练时使用）
                - "manual": 只允许预注册的配置（推理时使用）
        """
        if mode not in ["auto", "manual"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'auto' or 'manual'")

        self.mode = mode
        # 直接用 tuple 作为 key，性能最优（无需创建对象）
        self.key_to_idx: Dict[tuple, int] = {}
        self.key_to_idx[0] = 'lerobot-pi0-bridge_widowx_no_geom_aug|act_tx=1|act_rt=1'

        logger.info(f"Initialized EmbodimentRegistry in {mode} mode")

    def get_or_register(self, key: tuple | EmbodimentKey) -> int:
        """
        获取 embodiment key 对应的 W index

        接受两种输入：
        1. tuple: 直接使用（性能最优，推荐用于训练循环）
        2. EmbodimentKey: 自动转换为 tuple

        如果是新配置：
        - auto 模式：自动注册并分配新 index
        - manual 模式：抛出异常

        Args:
            key: tuple 或 EmbodimentKey 实例

        Returns:
            w_index: 对应的 W 参数索引

        Raises:
            ValueError: manual 模式下遇到未注册的配置
        """
        # 如果是 NamedTuple，自动转换为 tuple（NamedTuple 本质上是 tuple 的子类）
        # 这个转换是零成本的，因为 NamedTuple 内部就是 tuple

        if key in self.key_to_idx:
            return self.key_to_idx[key]

        # 新配置
        if self.mode == "manual":
            raise ValueError(
                f"Unknown embodiment key (manual mode):\n"
                f"  Key: {key}\n"
                f"Please pre-register this config or switch to auto mode."
            )

        # 自动注册
        new_idx = len(self.key_to_idx)
        self.key_to_idx[key] = new_idx

        # logger.info(
        #     f"[EmbodimentRegistry] Registered new embodiment:\n"
        #     f"  W Index: {new_idx}\n"
        #     f"  Key: {key}"
        # )
        return new_idx
