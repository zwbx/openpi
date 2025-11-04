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


# 以上定义用于“参数驱动 + 函数式变换”的离散增广，不再依赖 Kornia 的随机管线


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    aug_config: dict | list[dict] | None = None,  # Kept for backward compatibility, not used with Kornia
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    return_aug_params: bool = False,  # New parameter to return augmentation parameters
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

            # 裁剪 + 缩放
            boxes = _boxes_from_scale_pos(crop_scales, crop_pos_idx)
            image = KT.crop_and_resize(image, boxes, IMAGE_RESOLUTION)

            # 旋转（度 -> 弧度）
            angles_rad = rot_degs * torch.pi / 180.0
            image = KT.rotate(image, angles_rad)

            # 水平翻转（按掩码）
            flip_mask = flip_vals > 0.5
            if flip_mask.any():
                image[flip_mask] = torch.flip(image[flip_mask], dims=(3,))

            # 颜色扰动（preset）
            image = _apply_color_presets(image, cj_idx)

            # 记录参数与 key 组件
            if return_aug_params:
                if per_sample_view_params is None:
                    per_sample_view_params = [{} for _ in range(b)]
                    per_sample_key_components = [[] for _ in range(b)]

                for i in range(b):
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
