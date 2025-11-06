from collections.abc import Sequence
import logging

import torch
import kornia.geometry.transform as KT
import kornia.enhance as KE

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)

############################################
# 离散参数定义（有限集合，1:1 可复现）
############################################

_CROP_SCALES = (1.00, 0.95, 0.90)
_CROP_POS = ("C", "U", "D", "L", "R")  # Center / Up / Down / Left / Right
_ROT_DEGS = (-10.0, 0.0, 10.0)
_FLIP = (0, 1)

# 颜色预设（有限档位，便于离散回放）；索引即 preset id
_COLOR_PRESETS = (
    {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "hue": 0.0},  # p0: no-op
    {"brightness": 1.10, "contrast": 1.00, "saturation": 1.00, "hue": 0.00},
    {"brightness": 0.90, "contrast": 1.00, "saturation": 1.00, "hue": 0.00},
    {"brightness": 1.00, "contrast": 1.10, "saturation": 1.00, "hue": 0.00},
    {"brightness": 1.00, "contrast": 0.90, "saturation": 1.00, "hue": 0.00},
    {"brightness": 1.00, "contrast": 1.00, "saturation": 1.10, "hue": 0.00},
    {"brightness": 1.00, "contrast": 1.00, "saturation": 0.90, "hue": 0.00},
    {"brightness": 1.00, "contrast": 1.00, "saturation": 1.00, "hue": 0.05},
)


def _sample_discrete(options: Sequence, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, len(options), (batch_size,), device=device)
    values = torch.tensor([options[i] for i in range(len(options))], device=device, dtype=torch.float32)
    return values[idx], idx  # values per sample, and indices


# Action scaling (discrete, per-sample, grouped)
_ACTION_TRANS_SCALES = (0.95, 1.00, 1.05)
_ACTION_ROT_SCALES = (0.95, 1.00, 1.05)


def sample_action_grouped_scales(
    batch_size: int,
    device: torch.device,
    *,
    prob_enable: float = 0.10,
    trans_scales: Sequence[float] = _ACTION_TRANS_SCALES,
    rot_scales: Sequence[float] = _ACTION_ROT_SCALES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample grouped action scales for translation (x,y,z) and rotation (roll,pitch,yaw).

    Returns:
        trans_values: [B] selected translation scale values
        trans_idx:    [B] indices into trans_scales
        rot_values:   [B] selected rotation scale values
        rot_idx:      [B] indices into rot_scales
        enabled_mask: [B] 1 if augmentation enabled, 0 otherwise
    """
    b = batch_size
    # Bernoulli mask for enabling augmentation per-sample
    if prob_enable <= 0:
        enabled_mask = torch.zeros((b,), device=device)
    elif prob_enable >= 1:
        enabled_mask = torch.ones((b,), device=device)
    else:
        enabled_mask = (torch.rand((b,), device=device) < float(prob_enable)).to(torch.float32)

    # Prepare discrete sets
    trans_vals_all = torch.tensor(list(trans_scales), device=device, dtype=torch.float32)
    rot_vals_all = torch.tensor(list(rot_scales), device=device, dtype=torch.float32)

    if trans_vals_all.numel() == 0 or rot_vals_all.numel() == 0:
        raise ValueError("trans_scales and rot_scales must be non-empty")

    # Sample indices for all samples
    trans_idx = torch.randint(0, trans_vals_all.numel(), (b,), device=device)
    rot_idx = torch.randint(0, rot_vals_all.numel(), (b,), device=device)

    trans_values = trans_vals_all[trans_idx]
    rot_values = rot_vals_all[rot_idx]

    # For disabled samples, force 1.0 and index-of-1.0
    def _idx_of_one(vals: torch.Tensor) -> int:
        diffs = torch.abs(vals - 1.0)
        return int(torch.argmin(diffs).item())

    one_idx_trans = _idx_of_one(trans_vals_all)
    one_idx_rot = _idx_of_one(rot_vals_all)

    if (enabled_mask == 0).any():
        mask = (enabled_mask == 0)
        trans_values = torch.where(mask, torch.tensor(1.0, device=device), trans_values)
        rot_values = torch.where(mask, torch.tensor(1.0, device=device), rot_values)
        trans_idx = torch.where(mask, torch.tensor(one_idx_trans, device=device), trans_idx)
        rot_idx = torch.where(mask, torch.tensor(one_idx_rot, device=device), rot_idx)

    return trans_values, trans_idx, rot_values, rot_idx, enabled_mask


def apply_action_scale_grouped(
    actions: torch.Tensor,
    trans_scales: torch.Tensor,
    rot_scales: torch.Tensor,
) -> torch.Tensor:
    """Apply grouped action scaling to translation (x,y,z) and rotation (roll,pitch,yaw).

    actions:       [B, T, D]
    trans_scales:  [B]
    rot_scales:    [B]

    Notes:
        - Does not modify gripper (dim 6) or any dims >= 7.
        - If action dim < 6, scales as many as available.
    """
    if actions.ndim != 3:
        raise ValueError(f"actions must be [B,T,D], got shape {actions.shape}")
    b, _, d = actions.shape
    scale_vec = torch.ones((b, 1, d), device=actions.device, dtype=actions.dtype)
    # translation: dims 0,1,2
    for idx in range(min(3, d)):
        scale_vec[:, :, idx] = trans_scales[:, None]
    # rotation: dims 3,4,5
    for idx in range(3, min(6, d)):
        scale_vec[:, :, idx] = rot_scales[:, None]
    # gripper (dim 6) and beyond remain 1.0
    return actions * scale_vec


def _boxes_from_scale_pos(scales: torch.Tensor, pos_indices: torch.Tensor) -> torch.Tensor:
    """构造每个样本的裁剪 box 角点（归一化坐标 [B, 4, 2]）。

    角点顺序: top-left, top-right, bottom-right, bottom-left
    pos_indices: 0:C,1:U,2:D,3:L,4:R
    """
    b = scales.shape[0]
    # 默认中心裁剪边界（x1,y1,x2,y2）
    x1 = (1 - scales) / 2
    y1 = (1 - scales) / 2
    x2 = 1 - x1
    y2 = 1 - y1

    # 根据位置对齐覆盖边界
    u_mask = pos_indices == 1  # Up
    if u_mask.any():
        y1 = torch.where(u_mask, torch.zeros_like(y1), y1)
        y2 = torch.where(u_mask, scales, y2)

    d_mask = pos_indices == 2  # Down
    if d_mask.any():
        y1 = torch.where(d_mask, 1.0 - scales, y1)
        y2 = torch.where(d_mask, torch.ones_like(y2), y2)

    l_mask = pos_indices == 3  # Left
    if l_mask.any():
        x1 = torch.where(l_mask, torch.zeros_like(x1), x1)
        x2 = torch.where(l_mask, scales, x2)

    r_mask = pos_indices == 4  # Right
    if r_mask.any():
        x1 = torch.where(r_mask, 1.0 - scales, x1)
        x2 = torch.where(r_mask, torch.ones_like(x2), x2)

    # 组装角点 [B, 4, 2]
    tl = torch.stack([x1, y1], dim=1)
    tr = torch.stack([x2, y1], dim=1)
    br = torch.stack([x2, y2], dim=1)
    bl = torch.stack([x1, y2], dim=1)
    points = torch.stack([tl, tr, br, bl], dim=1)
    return points.clamp(0, 1)


def _apply_color_presets(x: torch.Tensor, preset_indices: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W], 取 0..1
    b = x.shape[0]
    out = x
    for i in range(b):
        p = int(preset_indices[i].item())
        preset = _COLOR_PRESETS[p]
        if preset["brightness"] != 1.0:
            out[i] = KE.adjust_brightness(out[i], preset["brightness"])  # type: ignore[arg-type]
        if preset["contrast"] != 1.0:
            out[i] = KE.adjust_contrast(out[i], preset["contrast"])  # type: ignore[arg-type]
        if preset["saturation"] != 1.0:
            out[i] = KE.adjust_saturation(out[i], preset["saturation"])  # type: ignore[arg-type]
        if preset["hue"] != 0.0:
            out[i] = KE.adjust_hue(out[i], preset["hue"])  # type: ignore[arg-type]
    return out.clamp(0, 1)


def _format_geom_key(scale: float, pos_idx: int, rot_deg: float, flip: int) -> str:
    pos_letter = _CROP_POS[pos_idx]
    return f"crop=s{scale:.2f}@{pos_letter}|rot={int(rot_deg):+d}|flip={flip}"


def build_embodiment_keys(base_keys: list[tuple], obs_aug_metadata: dict, act_aug_metadata: dict | None = None):
    """Build final embodiment_keys by appending augmentation key per sample.

    Requirements:
    - base_keys must be a list of tuples, length == batch size.
    - obs_aug_metadata["keys"]: list[str], non-wrist geometric parts concatenated and sorted by view
    - act_aug_metadata["keys"]: list[str], e.g., "act_sc=1"
    """
    img_keys = obs_aug_metadata.get("keys", []) if isinstance(obs_aug_metadata, dict) else []
    act_keys = act_aug_metadata.get("keys", []) if isinstance(act_aug_metadata, dict) else []
    batch_size = len(img_keys) if img_keys else (len(act_keys) if act_keys else 0)

    if not isinstance(base_keys, list):
        raise TypeError("base_keys must be a list of tuples with length == batch size")
    if len(base_keys) != batch_size:
        raise ValueError(f"embodiment_keys length {len(base_keys)} != batch_size {batch_size}")
    for i, bk in enumerate(base_keys):
        if not isinstance(bk, tuple):
            raise TypeError(f"base_keys[{i}] must be a tuple, got {type(bk)}")

    final_keys = []
    for i in range(batch_size):
        parts = []
        if img_keys:
            parts.append(img_keys[i])
        if act_keys:
            parts.append(act_keys[i])
        aug_key = "|".join([p for p in parts if p]) if parts else ""
        final_keys.append(base_keys[i] + (aug_key,))
    return final_keys


# 以上定义用于“参数驱动 + 函数式变换”的离散增广，不再依赖 Kornia 的随机管线


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    aug_config: dict | list[dict] | None = None,  # Kept for backward compatibility, not used with Kornia
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    return_aug_params: bool = False,  # New parameter to return augmentation parameters
    aug_enable_prob: float | None = None,  # Probability to enable obs augmentation per-sample
):
    """Preprocess observation with discrete, parameter-driven per-sample augmentation.

    Args:
        observation: Input observation
        train: Whether in training mode (if True, applies discrete augmentations)
        aug_config: Deprecated, kept for backward compatibility (not used)
        image_keys: Keys for images to process
        image_resolution: Target image resolution
        return_aug_params: If True, return per-sample, per-view discrete augmentation parameters and keys

    Returns:
        If return_aug_params=False:
            Preprocessed observation with augmentations applied
        If return_aug_params=True:
            (processed_obs, {
                "params": List[Dict[str, descriptor]],  # 每个样本的视角 -> 增广参数
                "keys": List[str],  # 聚合后的字符串 key，用于构建 embodiment_keys
            })
    """
    # 根据实际observation中的图像动态设置image_keys
    if observation.images:
        image_keys = list(observation.images.keys())

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]
    out_images = {}
    per_sample_view_params: list[dict[str, dict[str, float | int]]] | None = None
    per_sample_key_components: list[list[str]] | None = None

    for key in image_keys:
        image = observation.images[key]
        is_wrist_view = "wrist" in key

        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # 转 0..1，并使用 channels-first 以便几何函数
            image = image / 2.0 + 0.5
            image = image.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]

            b = image.shape[0]
            device = image.device
            enable_mask = None
            if aug_enable_prob is not None and 0.0 <= float(aug_enable_prob) < 1.0:
                enable_mask = (torch.rand((b,), device=device) < float(aug_enable_prob))
            elif aug_enable_prob is not None and float(aug_enable_prob) >= 1.0:
                enable_mask = torch.ones((b,), dtype=torch.bool, device=device)
            elif aug_enable_prob is not None and float(aug_enable_prob) <= 0.0:
                enable_mask = torch.zeros((b,), dtype=torch.bool, device=device)

            # 采样离散参数（腕部：固定几何，不采样几何）
            if is_wrist_view:
                crop_scales = torch.full((b,), 1.0, device=device)
                crop_pos_idx = torch.zeros((b,), dtype=torch.long, device=device)  # C
                rot_degs = torch.zeros((b,), device=device)
                flip_vals = torch.zeros((b,), device=device)
            else:
                crop_scales, _ = _sample_discrete(_CROP_SCALES, b, device)
                _, crop_pos_idx = _sample_discrete(range(len(_CROP_POS)), b, device)
                rot_degs, _ = _sample_discrete(_ROT_DEGS, b, device)
                flip_vals, _ = _sample_discrete(_FLIP, b, device)

            # 颜色 preset（所有视角都可）
            _, cj_idx = _sample_discrete(range(len(_COLOR_PRESETS)), b, device)

            # 保存原图，用于按概率禁用增广
            image_orig = image

            # 裁剪 + 缩放
            boxes = _boxes_from_scale_pos(crop_scales, crop_pos_idx)
            image_aug = KT.crop_and_resize(image, boxes, IMAGE_RESOLUTION)

            # 旋转（度 -> 弧度）
            angles_rad = rot_degs * torch.pi / 180.0
            image_aug = KT.rotate(image_aug, angles_rad)

            # 水平翻转（按掩码）
            flip_mask = flip_vals > 0.5
            if flip_mask.any():
                image_aug[flip_mask] = torch.flip(image_aug[flip_mask], dims=(3,))

            # 颜色扰动（preset）
            image_aug = _apply_color_presets(image_aug, cj_idx)

            # 按概率选择是否采用增广图像
            if enable_mask is not None:
                mask = enable_mask.view(b, 1, 1, 1)
                image = torch.where(mask, image_aug, image_orig)
            else:
                image = image_aug

            # 记录参数与 key 组件
            if return_aug_params:
                if per_sample_view_params is None:
                    per_sample_view_params = [{} for _ in range(b)]
                    per_sample_key_components = [[] for _ in range(b)]

                for i in range(b):
                    # 如果该样本未启用增广，则记录 no-op 参数并不加入 key 组件
                    if enable_mask is not None and not bool(enable_mask[i].item()):
                        descriptor = {
                            "crop_scale": 1.0,
                            "crop_pos": "C",
                            "rotation_deg": 0.0,
                            "flip": 0,
                            "cj_preset": 0,
                        }
                        per_sample_view_params[i][key] = descriptor
                    else:
                        descriptor = {
                            "crop_scale": float(crop_scales[i].item()),
                            "crop_pos": _CROP_POS[int(crop_pos_idx[i].item())],
                            "rotation_deg": float(rot_degs[i].item()),
                            "flip": int(flip_vals[i].item()),
                            "cj_preset": int(cj_idx[i].item()),
                        }
                        per_sample_view_params[i][key] = descriptor
                        if not is_wrist_view:
                            per_sample_key_components[i].append(
                                f"{key}:{_format_geom_key(descriptor['crop_scale'], int(crop_pos_idx[i].item()), descriptor['rotation_deg'], descriptor['flip'])}"
                            )

            # 回到 [B,H,W,C]，并转回 [-1,1]
            image = image.permute(0, 2, 3, 1)
            image = torch.clamp(image, 0, 1)
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        # if is_channels_first:
        image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]
    
    # HACK
    # Ensure state is float32 to avoid unintended float64 promotion downstream
    state = observation.state
    if isinstance(state, torch.Tensor) and state.dtype != torch.float32:
        state = state.to(torch.float32)

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    processed_obs = SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )

    if return_aug_params:
        if per_sample_view_params is None:
            if out_images:
                batch_size = next(iter(out_images.values())).shape[0]
            else:
                state_shape = observation.state.shape
                batch_size = state_shape[0] if len(state_shape) > 0 else 1
            per_sample_view_params = [{} for _ in range(batch_size)]
            per_sample_key_components = [[] for _ in range(batch_size)]

        keys = [
            "|".join(sorted(components)) if components else "no_geom_aug"
            for components in per_sample_key_components
        ]

        aug_metadata = {
            "params": per_sample_view_params,
            "keys": keys,
        }
        return processed_obs, aug_metadata

    return processed_obs


def preprocess_actions_pytorch(
    actions: torch.Tensor,
    *,
    return_aug_params: bool = False,
    action_aug_prob: float = 0.10,
):
    """Preprocess actions with grouped augmentation for translation and rotation.

    Args:
        actions: [B, T, D]
        action_aug_prob: Probability to enable augmentation per sample (0..1)
        return_aug_params: If True, also return per-sample metadata dict

    Returns:
        If return_aug_params=False:
            Tensor actions_out
        If return_aug_params=True:
            (actions_out, {
               'params': List[{'action': {'transition': float, 'rotation': float}}],
               'key_suffixes': List[str],  # e.g., 'act_tx=1|act_rt=2'
            })
    """
    if actions.ndim != 3:
        raise ValueError(f"actions must be shape [B,T,D], got {actions.shape}")

    bsz = actions.shape[0]
    device = actions.device

    if action_aug_prob <= 0.0:
        if return_aug_params:
            # No augmentation: scales are 1.0, use index closest to 1.0
            trans_vals_all = torch.tensor(list(_ACTION_TRANS_SCALES), device=device, dtype=torch.float32)
            rot_vals_all = torch.tensor(list(_ACTION_ROT_SCALES), device=device, dtype=torch.float32)

            def _idx_of_one(vals: torch.Tensor) -> int:
                diffs = torch.abs(vals - 1.0)
                return int(torch.argmin(diffs).item())

            one_idx_trans = _idx_of_one(trans_vals_all)
            one_idx_rot = _idx_of_one(rot_vals_all)
            params = [{"action": {"transition": 1.0, "rotation": 1.0}} for _ in range(bsz)]
            suffixes = [f"act_tx={one_idx_trans}|act_rt={one_idx_rot}" for _ in range(bsz)]
            return actions, {"params": params, "key_suffixes": suffixes}
        return actions

    # Sample and apply grouped scales
    trans_scales, trans_idx, rot_scales, rot_idx, enabled_mask = sample_action_grouped_scales(
        bsz, device, prob_enable=action_aug_prob
    )
    actions_out = apply_action_scale_grouped(actions, trans_scales, rot_scales)

    if not return_aug_params:
        return actions_out

    params = []
    suffixes = []
    for i in range(bsz):
        params.append({
            "action": {
                "transition": float(trans_scales[i].item()),
                "rotation": float(rot_scales[i].item()),
            }
        })
        suffixes.append(f"act_tx={int(trans_idx[i].item())}|act_rt={int(rot_idx[i].item())}")

    return actions_out, {"params": params, "key_suffixes": suffixes}


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
