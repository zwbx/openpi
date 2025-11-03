"""
独立调试模型脚本 - 使用保存的输入数据测试模型 forward pass

用法:
    python test_model_debug.py

这个脚本会:
1. 加载训练时保存的模型输入数据
2. 创建模型实例
3. 运行 forward pass 并打印详细信息
4. 方便单步调试和检查中间结果
"""

import logging
import sys
from pathlib import Path

import torch

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config


def init_logging():
    """初始化日志"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def main():
    init_logging()

    # 配置路径
    debug_inputs_path = Path("checkpoints/pi05_simpler_debug/pytorch_test/debug_model_inputs.pt")

    if not debug_inputs_path.exists():
        logging.error(f"调试输入文件不存在: {debug_inputs_path}")
        logging.error("请先运行训练脚本生成调试输入:")
        logging.error("  uv run scripts/train_pytorch.py debug --exp_name pytorch_test")
        sys.exit(1)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 加载保存的输入数据
    logging.info(f"加载调试输入数据: {debug_inputs_path}")
    inputs = torch.load(debug_inputs_path, map_location=device, weights_only=False)

    observation = inputs["observation"]
    actions = inputs["actions"]
    next_obs = inputs["next_obs"]

    # 打印输入数据形状
    logging.info("=" * 80)
    logging.info("输入数据形状:")
    logging.info("-" * 80)

    logging.info(f"observation 类型: {type(observation)}")
    if hasattr(observation, 'image'):
        logging.info(f"  observation.image 键: {list(observation.image.keys())}")
        for key, img in observation.image.items():
            logging.info(f"    {key}: {img.shape} ({img.dtype})")
    if hasattr(observation, 'state'):
        logging.info(f"  observation.state: {observation.state.shape} ({observation.state.dtype})")
    if hasattr(observation, 'text'):
        logging.info(f"  observation.text: {observation.text.shape} ({observation.text.dtype})")

    logging.info(f"actions: {actions.shape} ({actions.dtype})")

    if next_obs is not None:
        logging.info(f"next_obs 类型: {type(next_obs)}")
        if hasattr(next_obs, 'image'):
            logging.info(f"  next_obs.image 键: {list(next_obs.image.keys())}")
            for key, img in next_obs.image.items():
                logging.info(f"    {key}: {img.shape} ({img.dtype})")
        if hasattr(next_obs, 'state'):
            logging.info(f"  next_obs.state: {next_obs.state.shape} ({next_obs.state.dtype})")
    else:
        logging.info("next_obs: None")

    logging.info("=" * 80)

    # 创建模型配置 (使用 debug config)
    logging.info("创建模型配置...")

    # 从 config 加载完整配置
    config = _config.get_config("pi05_simpler_debug")

    # 创建模型
    if not isinstance(config.model, pi0_config.Pi0Config):
        model_cfg = pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    logging.info(f"模型配置: {model_cfg}")

    # 创建模型实例
    logging.info("创建模型实例...")
    model = pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    model.train()  # 设置为评估模式,方便调试

    # 启用梯度检查点 (可选)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("启用梯度检查点")

    logging.info("=" * 80)
    logging.info("开始 forward pass...")
    logging.info("=" * 80)


    try:
        # Forward pass
        with torch.no_grad():  # 不需要梯度,方便调试
            losses = model(observation, actions, next_obs=next_obs)

        logging.info("Forward pass 成功!")
        logging.info(f"losses: {losses}")

        if isinstance(losses, dict):
            logging.info("损失详情:")
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    logging.info(f"  {key}: {value.item():.6f}")
                else:
                    logging.info(f"  {key}: {value}")
        elif isinstance(losses, torch.Tensor):
            logging.info(f"总损失: {losses.item():.6f}")

    except Exception as e:
        logging.error(f"Forward pass 失败: {e}")
        import traceback
        traceback.print_exc()

        # 在错误时也进入调试模式
        import pdb; pdb.post_mortem()


if __name__ == "__main__":
    main()
