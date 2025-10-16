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

import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, NamedTuple, Optional

logger = logging.getLogger("openpi")


class EmbodimentKey(NamedTuple):
    """
    Embodiment configuration 的唯一标识

    使用 NamedTuple 的优点:
    1. 有明确的字段名（不依赖顺序）
    2. 不可变（immutable），可以作为 dict key
    3. 自动实现 __hash__ 和 __eq__
    4. 打印时显示字段名，易于调试

    字段说明:
    - robot_type: 机器人类型 (如 "franka", "ur5", "aloha", "simpler")
    - dof: 自由度数量
    - action_space: 动作空间类型 ("cartesian" | "joint")
    - state_space: 状态空间类型 ("cartesian" | "joint")
    - coordinate_frame: 坐标系 ("base" | "world" | "camera")
    - image_crop: 是否应用了图像裁剪（改变 layout）
    - image_rotation: 是否应用了图像旋转（改变 layout）
    - image_flip: 是否应用了图像翻转（改变 layout）
    - camera_viewpoint_id: 相机视角 ID
    """

    robot_type: str
    dof: int
    action_space: str
    state_space: str
    coordinate_frame: str
    image_crop: bool
    image_rotation: bool
    image_flip: bool
    camera_viewpoint_id: str


@dataclasses.dataclass
class EmbodimentConfig:
    """
    完整的 Embodiment Configuration

    包含所有配置信息，可以转换为 EmbodimentKey
    区分"影响 embodiment"和"不影响 embodiment"的因素
    """

    # ========== 影响 embodiment context 的因素 ==========
    # 这些字段会参与 EmbodimentKey 的计算

    # Robot configuration
    robot_type: str = "franka"
    dof: int = 7

    # Action/State space
    action_space: str = "cartesian"  # "cartesian" | "joint"
    state_space: str = "cartesian"  # "cartesian" | "joint"
    coordinate_frame: str = "base"  # "base" | "world" | "camera"

    # Image layout transformations (影响 observation layout)
    image_crop: bool = False  # 是否做了裁剪
    image_rotation: bool = False  # 是否做了旋转
    image_flip: bool = False  # 是否做了翻转

    # Camera configuration
    camera_viewpoint_id: str = "default"

    # ========== 不影响 embodiment context 的因素 ==========
    # 这些字段只是记录，不参与 EmbodimentKey 计算
    # 以下划线 _ 开头表示"内部字段"

    # Appearance augmentations (只改变外观，不改变 layout)
    _color_jitter: bool = False
    _brightness_aug: bool = False
    _contrast_aug: bool = False
    _saturation_aug: bool = False
    _gaussian_noise: bool = False

    # Metadata
    _dataset_name: Optional[str] = None  # 数据集名称（仅记录）
    _description: Optional[str] = None  # 人类可读的描述

    def to_key(self) -> EmbodimentKey:
        """
        转换为 EmbodimentKey（只包含影响 embodiment 的因素）

        注意：外观增强（color_jitter 等）不参与 key 计算
        因为它们只改变图像外观，不改变 observation-action-state 的对应关系

        Returns:
            EmbodimentKey: 唯一标识这个 embodiment configuration
        """
        return EmbodimentKey(
            robot_type=self.robot_type,
            dof=self.dof,
            action_space=self.action_space,
            state_space=self.state_space,
            coordinate_frame=self.coordinate_frame,
            image_crop=self.image_crop,
            image_rotation=self.image_rotation,
            image_flip=self.image_flip,
            camera_viewpoint_id=self.camera_viewpoint_id,
        )

    def to_readable_string(self) -> str:
        """
        生成可读的字符串表示

        格式: {robot}_{dof}dof_{action_space}_{frame}_{aug_flags}_cam{camera}

        Examples:
            "franka_7dof_cartesian_base_noaug_camdefault"
            "franka_7dof_cartesian_base_crop_rot_camdefault"
            "ur5_6dof_joint_world_noaug_camtop"
        """
        # 构造增强标记
        aug_flags = []
        if self.image_crop:
            aug_flags.append("crop")
        if self.image_rotation:
            aug_flags.append("rot")
        if self.image_flip:
            aug_flags.append("flip")

        aug_str = "_".join(aug_flags) if aug_flags else "noaug"

        return (
            f"{self.robot_type}_"
            f"{self.dof}dof_"
            f"{self.action_space}_"
            f"{self.coordinate_frame}_"
            f"{aug_str}_"
            f"cam{self.camera_viewpoint_id}"
        )

    @classmethod
    def from_training_mode(
        cls,
        robot_type: str = "franka",
        dof: int = 7,
        train: bool = False,
        enable_geometric_aug: bool = True,
    ) -> "EmbodimentConfig":
        """
        根据训练模式创建配置

        这个工厂方法对应 preprocessing_pytorch.py 中的数据增强逻辑:
        - train=True 且 enable_geometric_aug=True: 应用 crop + rotation
        - train=False: 不应用几何增强

        Args:
            robot_type: 机器人类型
            dof: 自由度
            train: 是否训练模式
            enable_geometric_aug: 是否启用几何增强

        Returns:
            EmbodimentConfig 实例
        """
        return cls(
            robot_type=robot_type,
            dof=dof,
            # 训练时启用几何增强
            image_crop=train and enable_geometric_aug,
            image_rotation=train and enable_geometric_aug,
            # 训练时启用外观增强（记录用，不影响 key）
            _color_jitter=train,
            _brightness_aug=train,
            _contrast_aug=train,
            _saturation_aug=train,
        )


class EmbodimentRegistry:
    """
    管理所有 embodiment configurations 和 W index 的映射

    职责:
    1. 维护 EmbodimentKey -> W index 的映射
    2. 自动注册新的 embodiment configurations
    3. 保存/加载 registry 到文件

    使用方式:
        registry = EmbodimentRegistry(mode="auto")
        w_index = registry.get_or_register(config)
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
        self.key_to_idx: Dict[EmbodimentKey, int] = {}
        self.idx_to_config: Dict[int, EmbodimentConfig] = {}

        logger.info(f"Initialized EmbodimentRegistry in {mode} mode")

    def get_or_register(self, config: EmbodimentConfig) -> int:
        """
        获取 embodiment config 对应的 W index

        如果是新配置：
        - auto 模式：自动注册并分配新 index
        - manual 模式：抛出异常

        Args:
            config: EmbodimentConfig 实例

        Returns:
            w_index: 对应的 W 参数索引

        Raises:
            ValueError: manual 模式下遇到未注册的配置
        """
        key = config.to_key()

        if key in self.key_to_idx:
            return self.key_to_idx[key]

        # 新配置
        if self.mode == "manual":
            raise ValueError(
                f"Unknown embodiment config (manual mode):\n"
                f"  Readable: {config.to_readable_string()}\n"
                f"  Key: {key}\n"
                f"Please pre-register this config or switch to auto mode."
            )

        # 自动注册
        new_idx = len(self.key_to_idx)
        self.key_to_idx[key] = new_idx
        self.idx_to_config[new_idx] = config

        logger.info(
            f"[EmbodimentRegistry] Registered new embodiment:\n"
            f"  W Index: {new_idx}\n"
            f"  Name: {config.to_readable_string()}\n"
            f"  Key: {key}"
        )

        return new_idx

    def get_config(self, w_index: int) -> Optional[EmbodimentConfig]:
        """
        根据 W index 获取配置

        Args:
            w_index: W 参数索引

        Returns:
            EmbodimentConfig 或 None（如果 index 不存在）
        """
        return self.idx_to_config.get(w_index)

    def __len__(self) -> int:
        """返回注册的 embodiment 数量"""
        return len(self.key_to_idx)

    def __contains__(self, config: EmbodimentConfig) -> bool:
        """检查配置是否已注册"""
        return config.to_key() in self.key_to_idx

    def summary(self) -> str:
        """
        生成 registry 的摘要信息

        Returns:
            人类可读的摘要字符串
        """
        lines = [
            f"EmbodimentRegistry Summary",
            f"  Mode: {self.mode}",
            f"  Total embodiments: {len(self)}",
            "",
            "  Registered configurations:",
        ]

        for idx in sorted(self.idx_to_config.keys()):
            config = self.idx_to_config[idx]
            lines.append(f"    [{idx}] {config.to_readable_string()}")

        return "\n".join(lines)

    def save(self, path: str | Path):
        """
        保存 registry 到 JSON 文件

        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "mode": self.mode,
            "num_embodiments": len(self),
            "configs": {
                str(idx): {
                    "key": key._asdict(),  # NamedTuple 转 dict
                    "config": dataclasses.asdict(config),
                    "readable_name": config.to_readable_string(),
                }
                for key, idx in self.key_to_idx.items()
                for config in [self.idx_to_config[idx]]
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved embodiment registry to {path}: {len(self)} embodiments")

    def load(self, path: str | Path):
        """
        从 JSON 文件加载 registry

        Args:
            path: 文件路径

        Raises:
            FileNotFoundError: 文件不存在
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        self.mode = data["mode"]
        self.key_to_idx.clear()
        self.idx_to_config.clear()

        for idx_str, config_data in data["configs"].items():
            idx = int(idx_str)

            # 重建 EmbodimentKey
            key = EmbodimentKey(**config_data["key"])

            # 重建 EmbodimentConfig
            config = EmbodimentConfig(**config_data["config"])

            self.key_to_idx[key] = idx
            self.idx_to_config[idx] = config

        logger.info(f"Loaded embodiment registry from {path}: {len(self)} embodiments")

    def merge(self, other: "EmbodimentRegistry"):
        """
        合并另一个 registry

        用于组合多个数据集的 configurations

        Args:
            other: 另一个 EmbodimentRegistry 实例

        Note:
            如果有重复的 key，会保留当前 registry 的 index
        """
        for key, config in zip(other.key_to_idx.keys(), other.idx_to_config.values()):
            if key not in self.key_to_idx:
                # 新配置，添加
                new_idx = len(self.key_to_idx)
                self.key_to_idx[key] = new_idx
                self.idx_to_config[new_idx] = config
                logger.info(f"Merged new embodiment [{new_idx}]: {config.to_readable_string()}")
            else:
                logger.debug(f"Skipped duplicate embodiment: {config.to_readable_string()}")


# 预定义的常见配置
def get_default_config(robot_type: str = "franka") -> EmbodimentConfig:
    """
    获取默认配置（无增强）

    Args:
        robot_type: 机器人类型

    Returns:
        默认的 EmbodimentConfig
    """
    return EmbodimentConfig(
        robot_type=robot_type,
        dof=7 if robot_type == "franka" else 6,
        action_space="cartesian",
        state_space="cartesian",
        coordinate_frame="base",
        image_crop=False,
        image_rotation=False,
        image_flip=False,
    )


def get_training_config(robot_type: str = "franka", enable_geometric_aug: bool = True) -> EmbodimentConfig:
    """
    获取训练配置（启用增强）

    Args:
        robot_type: 机器人类型
        enable_geometric_aug: 是否启用几何增强

    Returns:
        训练用的 EmbodimentConfig
    """
    return EmbodimentConfig.from_training_mode(
        robot_type=robot_type,
        dof=7 if robot_type == "franka" else 6,
        train=True,
        enable_geometric_aug=enable_geometric_aug,
    )
