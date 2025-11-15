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
import random
import warnings
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
        self.idx_to_key: Dict[int, tuple] = {}
        self.idx_to_obs_aug_param: Dict[int, dict] = {}
        self.idx_to_act_aug_param: Dict[int, dict] = {}
        self.key_to_idx['lerobot-pi0-bridge_widowx_no_geom_aug|act_tx=1|act_rt=1'] = 0
        self.idx_to_key[0] = 'lerobot-pi0-bridge_widowx_no_geom_aug|act_tx=1|act_rt=1'
        self.idx_to_obs_aug_param[0] = {'base_0_rgb': {'crop_scale': 1.0, 'crop_pos': 'C', 'rotation_deg': 0.0, 'flip': 0, 'cj_preset': 0}}
        self.idx_to_act_aug_param[0] = {'transition': 1.0, 'rotation': 1.0}

        logger.info(f"Initialized EmbodimentRegistry in {mode} mode")

    def get_or_register(self, key: tuple | EmbodimentKey, obs_aug_param: str | None = None, act_aug_param: str | None = None) -> int:
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
        self.idx_to_key[new_idx] = key
        self.idx_to_obs_aug_param[new_idx] = obs_aug_param
        self.idx_to_act_aug_param[new_idx] = act_aug_param

        return new_idx


class PrefixTokenBank:
    """
    PEFT prefix token bank (E-Token Bank) using nn.Embedding

    Manages prefix tokens for different embodiment configurations with composite IDs
    based on: base keys (data sources) x observation augmentations x action augmentations

    Usage:
        config = {...}  # Contains peft config and augmentation settings
        token_bank = PrefixTokenBank(config, width=768)

        # Get embodiment tokens
        e_tokens = token_bank.get_embodiment_token(
            base_embodiment_keys=['default'],
            obs_aug_metadata={'params': [...]},
            act_aug_metadata={'params': [...]}
        )
    """

    def __init__(self, config, width: int):
        """
        Initialize prefix token bank

        Args:
            config: Configuration object with attributes:
                - use_peft_prefix_token: bool, whether to use PEFT prefix tokens
                - peft_num_tokens: int, number of prefix tokens
                - peft_init: str, initialization method ('normal' or 'zeros')
                - base_keys: list[str] | None, base data source keys
            width: int, hidden dimension of the model
        """
        import torch
        import torch.nn as nn

        # Import augmentation constants
        from ..models_pytorch.preprocessing_pytorch import (
            _CROP_SCALES, _CROP_POS, _ROT_DEGS, _FLIP,
            _ACTION_TRANS_SCALES, _ACTION_ROT_SCALES
        )

        # PEFT prefix token bank configuration
        self.use_peft_prefix_token = config.use_peft_prefix_token
        self.peft_num_tokens = config.peft_num_tokens
        self.peft_init = config.peft_init
        self.width = width
        self.token_hidden_dim = self.width * self.peft_num_tokens

        # Base keys (data sources); default single base
        base_keys = config.base_keys
        if base_keys is None:
            base_keys = ['default']
        self.base_keys: list[str] = list(base_keys)
        self.base_key_to_id = {k: i for i, k in enumerate(self.base_keys)}
        B = len(self.base_keys)

        # Observation augmentation discrete options (deterministic order)
        self.obs_crop_scales = _CROP_SCALES
        self.obs_crop_pos = _CROP_POS
        self.obs_rot_degs = _ROT_DEGS
        self.obs_flip = _FLIP
        # Color preset currently fixed to 0 in forced pipeline

        # Action augmentation discrete options
        self.act_trans_scales = _ACTION_TRANS_SCALES
        self.act_rot_scales = _ACTION_ROT_SCALES

        # Sizes and total capacity
        self._n_cs = len(self.obs_crop_scales)
        self._n_pos = len(self.obs_crop_pos)
        self._n_rot = len(self.obs_rot_degs)
        self._n_flip = len(self.obs_flip)
        NO = self._n_cs * self._n_pos * self._n_rot * self._n_flip
        self._n_t = len(self.act_trans_scales)
        self._n_r = len(self.act_rot_scales)
        NA = self._n_t * self._n_r

        if NO <= 0 or NA <= 0 or B <= 0:
            raise ValueError(f"Invalid prefix capacity: B={B}, NO={NO}, NA={NA}")

        self._B = B
        self._NO = NO
        self._NA = NA

        self.num_embeddings = self._B * self._NO * self._NA

        # prefix token bank
        self.prefix_token_bank = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.token_hidden_dim,
        )

        # Initialize weights
        if self.peft_init == 'normal':
            nn.init.normal_(self.prefix_token_bank.weight, mean=0.0, std=0.02)
        else:  # 'zeros'
            nn.init.zeros_(self.prefix_token_bank.weight)

    def get_embodiment_token(self, base_embodiment_keys, obs_aug_metadata, act_aug_metadata):
        """
        Get embodiment tokens based on augmentation metadata

        Args:
            base_embodiment_keys: List[str] | None, base embodiment keys for each sample
            obs_aug_metadata: dict with 'params' containing observation augmentation parameters
            act_aug_metadata: dict with 'params' containing action augmentation parameters

        Returns:
            e_token: [batch_size, num_tokens, hidden_dim]
        """
        import torch
        import numpy as np
        from ..models_pytorch.preprocessing_pytorch import (
            _CROP_SCALES, _CROP_POS, _ROT_DEGS, _FLIP,
            _ACTION_TRANS_SCALES, _ACTION_ROT_SCALES
        )

        def find_closest_index(value, tuple_values):
            """Find index of closest value in tuple, handling floating point precision."""
            min_diff = float('inf')
            best_idx = 0
            for i, v in enumerate(tuple_values):
                diff = abs(value - v)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            return best_idx

        device = self.prefix_token_bank.weight.device
        batch_size = len(base_embodiment_keys)

        idx_list = []
        for i in range(batch_size):
            crop_scale = obs_aug_metadata['params'][i]['base_0_rgb']['crop_scale']
            crop_pos = obs_aug_metadata['params'][i]['base_0_rgb']['crop_pos']
            rotation_deg = obs_aug_metadata['params'][i]['base_0_rgb']['rotation_deg']
            flip = obs_aug_metadata['params'][i]['base_0_rgb']['flip']
            transition = act_aug_metadata['params'][i]['action']['transition']
            rotation = act_aug_metadata['params'][i]['action']['rotation']

            # Use find_closest_index for numeric values to handle floating point precision
            crop_scale_idx = find_closest_index(crop_scale, _CROP_SCALES)
            rotation_deg_idx = find_closest_index(rotation_deg, _ROT_DEGS)
            transition_idx = find_closest_index(transition, _ACTION_TRANS_SCALES)
            rotation_idx = find_closest_index(rotation, _ACTION_ROT_SCALES)

            # String/integer values can use exact matching
            crop_pos_idx = _CROP_POS.index(crop_pos)
            flip_idx = find_closest_index(flip, _FLIP)  # flip is also numeric, use find_closest_index

            idx = (crop_scale_idx, crop_pos_idx, rotation_deg_idx, flip_idx, transition_idx, rotation_idx)
            idx = np.array(idx)
            idx = np.ravel_multi_index(idx, (len(_CROP_SCALES), len(_CROP_POS), len(_ROT_DEGS), len(_FLIP), len(_ACTION_TRANS_SCALES), len(_ACTION_ROT_SCALES)))
            idx_list.append(idx)

        idx_list = torch.tensor(idx_list, device=device)
        e_tokens = self.prefix_token_bank(idx_list)
        e_tokens = e_tokens.view(batch_size, self.peft_num_tokens, self.token_hidden_dim)

        return e_tokens

    # @staticmethod
    # def build_embodiment_keys(base_keys: list[tuple], obs_aug_metadata: dict, act_aug_metadata: dict | None = None):
    #     """Build final embodiment_keys by appending augmentation key per sample.

    #     Requirements:
    #     - base_keys must be a list of tuples, length == batch size.
    #     - obs_aug_metadata["keys"]: list[str], non-wrist geometric parts concatenated and sorted by view
    #     - act_aug_metadata["keys"]: list[str], e.g., "act_sc=1"
    #     """
    #     img_keys = obs_aug_metadata.get("keys", []) if isinstance(obs_aug_metadata, dict) else []
    #     act_keys = act_aug_metadata.get("keys", []) if isinstance(act_aug_metadata, dict) else []
    #     batch_size = len(img_keys) if img_keys else (len(act_keys) if act_keys else 0)

    #     if not isinstance(base_keys, list):
    #         raise TypeError("base_keys must be a list of tuples with length == batch size")
    #     if len(base_keys) != batch_size:
    #         raise ValueError(f"embodiment_keys length {len(base_keys)} != batch_size {batch_size}")
    #     for i, bk in enumerate(base_keys):
    #         if not isinstance(bk, tuple):
    #             raise TypeError(f"base_keys[{i}] must be a tuple, got {type(bk)}")

    #     final_keys = []
    #     for i in range(batch_size):
    #         parts = []
    #         if img_keys:
    #             parts.append(img_keys[i])
    #         if act_keys:
    #             parts.append(act_keys[i])
    #         aug_key = "|".join([p for p in parts if p]) if parts else ""
    #         final_keys.append(base_keys[i] + (aug_key,))
    #     return final_keys
    @staticmethod
    def build_embodiment_keys(
        base_keys,
        obs_aug_metadata: dict | None,
        act_aug_metadata: dict | None,
        *,
        join_with: str = "_",
        act_sep: str = "|",
    ) -> list[str]:
        """Merge base keys, observation aug keys, and action aug suffixes into final embodiment keys.

        - base_keys: list/iterable of per-sample base keys, a single key, or None
        - obs_aug_metadata: expects {'keys': List[str]}
        - act_aug_metadata: expects {'key_suffixes': List[str]}
        """
        obs_keys = None
        if isinstance(obs_aug_metadata, dict):
            obs_keys = obs_aug_metadata.get("keys")

        act_suffixes = None
        if isinstance(act_aug_metadata, dict):
            act_suffixes = act_aug_metadata.get("key_suffixes")

        # Determine batch size
        b = None
        if isinstance(obs_keys, list):
            b = len(obs_keys)
        elif isinstance(act_suffixes, list):
            b = len(act_suffixes)
        elif isinstance(base_keys, list):
            b = len(base_keys)
        if b is None:
            raise ValueError("Cannot infer batch size for building embodiment keys")

        # Normalize inputs to lists of strings
        def to_list(x, n):
            if x is None:
                return [""] * n
            if isinstance(x, list):
                return [str(v) for v in x]
            # single scalar -> repeat
            return [str(x)] * n

        base_list = to_list(base_keys, b)
        obs_list = to_list(obs_keys, b)
        act_list = to_list(act_suffixes, b)

        # Build per-sample augmentation key (obs + action suffix)
        aug_list = []
        for o, a in zip(obs_list, act_list, strict=True):
            if a:
                aug_list.append(f"{o}{act_sep}{a}" if o else a)
            else:
                aug_list.append(o)

        # Combine base with augmentation key
        final = []
        for base, aug in zip(base_list, aug_list, strict=True):
            if base and aug:
                final.append(join_with.join((base, aug)))
            elif base:
                final.append(base)
            else:
                final.append(aug)
        return final


    def __len__(self):
        return self.num_embeddings
    
    def get_random_embodiment_token(self, base_embodiment_keys):
        """
        Get random embodiment tokens for initialization. 
        This is used for the case that given base keys and sample a random token with aug params.
        Args:
            base_embodiment_keys: List[str], base embodiment keys for each sample
        Returns:
            e_token: [batch_size, num_tokens, hidden_dim]
            obs_aug_metadata: dict with 'params' containing observation augmentation parameters
            act_aug_metadata: dict with 'params' containing action augmentation parameters
        """
        import torch
        import numpy as np
        import random
        from ..models_pytorch.preprocessing_pytorch import (
            _CROP_SCALES, _CROP_POS, _ROT_DEGS, _FLIP,
            _ACTION_TRANS_SCALES, _ACTION_ROT_SCALES
        )

        device = self.prefix_token_bank.weight.device
        batch_size = len(base_embodiment_keys)

        # Prepare lists for augmentation parameters
        obs_aug_params = []
        act_aug_params = []
        idx_list = []

        for i in range(batch_size):
            # 输入base key
            base_key = base_embodiment_keys[i]

            # 找到base key 对应的 flatten idx范围
            # Each base key has NO * NA possible combinations
            # base_key_id = self.base_key_to_id.get(base_key, 0)
            # start_idx = base_key_id * self._NO * self._NA
            # end_idx = (base_key_id + 1) * self._NO * self._NA

            # 范围内抽取一个随机的flatten idx
            # random_flat_idx = random.randint(start_idx, end_idx - 1)

            # 简化版本：随机采样各个维度的 idx（不考虑 base key 的范围）
            crop_scale_idx = random.randint(0, len(_CROP_SCALES) - 1)
            crop_pos_idx = random.randint(0, len(_CROP_POS) - 1)
            rotation_deg_idx = random.randint(0, len(_ROT_DEGS) - 1)
            flip_idx = random.randint(0, len(_FLIP) - 1)
            transition_idx = random.randint(0, len(_ACTION_TRANS_SCALES) - 1)
            rotation_idx = random.randint(0, len(_ACTION_ROT_SCALES) - 1)

            # 计算 flatten idx
            idx = np.ravel_multi_index(
                (crop_scale_idx, crop_pos_idx, rotation_deg_idx, flip_idx, transition_idx, rotation_idx),
                (len(_CROP_SCALES), len(_CROP_POS), len(_ROT_DEGS), len(_FLIP),
                 len(_ACTION_TRANS_SCALES), len(_ACTION_ROT_SCALES))
            )
            idx_list.append(idx)

            # 下面是得到 aug params的过程
            # 根据多维 idx 得到 aug params
            crop_scale = _CROP_SCALES[crop_scale_idx]
            crop_pos = _CROP_POS[crop_pos_idx]
            rotation_deg = _ROT_DEGS[rotation_deg_idx]
            flip = _FLIP[flip_idx]
            transition = _ACTION_TRANS_SCALES[transition_idx]
            rotation = _ACTION_ROT_SCALES[rotation_idx]

            # Construct obs aug params
            obs_param = {
                'base_0_rgb': {
                    'crop_scale': crop_scale,
                    'crop_pos': crop_pos,
                    'rotation_deg': rotation_deg,
                    'flip': flip,
                    'cj_preset': 0,  # Fixed to 0
                }
            }
            obs_aug_params.append(obs_param)

            # Construct act aug params
            act_param = {
                'action': {
                    'transition': transition,
                    'rotation': rotation,
                }
            }
            act_aug_params.append(act_param)

        # 从 embedding 中获取 e_tokens
        idx_list = torch.tensor(idx_list, device=device)
        e_tokens = self.prefix_token_bank(idx_list)
        e_tokens = e_tokens.view(batch_size, self.peft_num_tokens, self.width)

        # 返回 e token 和 obs/act aug params
        obs_aug_metadata = {'params': obs_aug_params}
        act_aug_metadata = {'params': act_aug_params}

        return e_tokens, obs_aug_metadata, act_aug_metadata