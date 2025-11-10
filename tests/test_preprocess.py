import torch

from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
from PIL import Image
import numpy as np


class DummyObservation:
    def __init__(self, images, image_masks, state):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = None
        self.tokenized_prompt_mask = None
        self.token_ar_mask = None
        self.token_loss_mask = None


def _make_gradient_image(b: int = 1, h: int = 224, w: int = 224, device: str = "cpu"):
    """Create a smooth RGB gradient image in [-1, 1], BHWC format."""
    xs = torch.linspace(0.0, 1.0, w, device=device)
    ys = torch.linspace(0.0, 1.0, h, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = xx  # horizontal gradient
    g = yy  # vertical gradient
    bch = (xx + yy) * 0.5  # diagonal gradient
    img = torch.stack([r, g, bch], dim=-1)  # HWC in [0,1]
    img = img.unsqueeze(0).repeat(b, 1, 1, 1)  # BHWC
    img = img * 2.0 - 1.0  # [-1, 1]
    return img.to(torch.float32)


def _blockiness_score(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Measure how close the image is to being piecewise constant per blocks.

    Returns a per-sample scalar: lower means more blocky. Expects BCHW in [-1,1].
    """
    # Convert to [0,1] and work on luminance-like channel (simple mean over channels)
    x01 = (x * 0.5 + 0.5).clamp(0, 1)
    y = x01.mean(dim=1, keepdim=True)  # [B,1,H,W]
    # Average pool to blocks, then upsample back and compare
    pooled = torch.nn.functional.avg_pool2d(y, kernel_size=block, stride=block, ceil_mode=False)
    up = torch.nn.functional.interpolate(pooled, size=y.shape[-2:], mode="nearest")
    diff = (y - up) ** 2
    # Normalize by variance to avoid scale issues
    var = y.var(dim=(-2, -1), keepdim=False).clamp_min(1e-8)  # [B,1]
    mse = diff.mean(dim=(-2, -1))  # [B,1]
    score = (mse / var).squeeze(1).mean(dim=-1)  # [B]
    return score


def test_preprocess_observation_pytorch_no_blocky_artifacts():
    device = "cpu"
    b, h, w = 1, 224, 224

    # 加载 CoA.png 图像
    img_pil = Image.open("CoA.png").convert("RGB").resize((w, h))
    img_np = np.array(img_pil).astype(np.float32) / 127.5 - 1.0  # 归一化到 [-1,1]
    img_bhwc = torch.from_numpy(img_np).unsqueeze(0).to(device)  # BHWC

    # Build dummy observation
    images = {"base_0_rgb": img_bhwc}
    image_masks = {"base_0_rgb": torch.ones((b,), dtype=torch.bool, device=device)}
    state = torch.zeros((b, 8), dtype=torch.float32, device=device)
    obs = DummyObservation(images=images, image_masks=image_masks, state=state)

    # Deterministic augmentation parameters per-sample per-view
    use_aug_params = [
        {
            "base_0_rgb": {
                "crop_scale": 0.70,
                "crop_pos": "C",
                "rotation_deg": 50.0,
                "flip": 1,
                # choose a color preset that is non-trivial but safe
                "cj_preset": 1,
            }
        }
    ]

    processed, meta = preprocess_observation_pytorch(
        obs,
        train=True,
        return_aug_params=True,
        aug_enable_prob=1.0,
        use_aug_params=use_aug_params,
        
    )

    out = processed.images["base_0_rgb"]  # [B,C,H,W] in [-1,1]

    # Basic sanity checks
    assert out.ndim == 4 and out.shape[0] == b and out.shape[1] == 3
    assert out.min().item() >= -1.0 - 1e-4 and out.max().item() <= 1.0 + 1e-4

    # The augmentation should not collapse the image into large color blocks.
    # A very blocky image will have a low score here.
    score = _blockiness_score(out, block=32)
    # Threshold chosen empirically: natural/augmented smooth gradients should be well above 1e-3
    assert torch.all(score > 1e-3), f"Blockiness score too low: {score}"

    # Also ensure the output is not close to a constant image (degenerate case)
    per_sample_std = out.float().reshape(b, -1).std(dim=1)
    assert torch.all(per_sample_std > 1e-3), f"Output image variance too low: {per_sample_std}"

