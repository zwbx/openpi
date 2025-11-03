"""
独立调试数据加载脚本 - 使用 pi05_simpler_debug 配置测试 dataload 模块（支持 MultiLeRobotDataset）

用法:
    python test_multidata_debug.py

这个脚本会:
1. 加载 pi05_simpler_debug 配置
2. 构造 MultiLeRobotDataset（将 repo_id 扩展为列表）
3. 创建 TorchDataLoader 并取出一个 batch
4. 打印详细的张量形状与类型，并保存调试输入到 checkpoint 目录
"""

import logging
import sys
from pathlib import Path
import dataclasses

import torch

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.models.model as _model


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


def print_observation_info(observation: _model.Observation, actions: torch.Tensor, next_obs: _model.Observation | None):
    logging.info("=" * 80)
    logging.info("输入数据形状:")
    logging.info("-" * 80)

    logging.info(f"observation 类型: {type(observation)}")
    if hasattr(observation, "image") and observation.image is not None:
        logging.info(f"  observation.image 键: {list(observation.image.keys())}")
        for key, img in observation.image.items():
            logging.info(f"    {key}: {img.shape} ({img.dtype})")
    if hasattr(observation, "state") and observation.state is not None:
        logging.info(f"  observation.state: {observation.state.shape} ({observation.state.dtype})")
    if hasattr(observation, "text") and observation.text is not None:
        logging.info(f"  observation.text: {getattr(observation.text, 'shape', None)} ({getattr(observation.text, 'dtype', None)})")

    logging.info(f"actions: {actions.shape} ({actions.dtype})")

    if next_obs is not None:
        logging.info(f"next_obs 类型: {type(next_obs)}")
        if hasattr(next_obs, "image") and next_obs.image is not None:
            logging.info(f"  next_obs.image 键: {list(next_obs.image.keys())}")
            for key, img in next_obs.image.items():
                logging.info(f"    {key}: {img.shape} ({img.dtype})")
        if hasattr(next_obs, "state") and next_obs.state is not None:
            logging.info(f"  next_obs.state: {next_obs.state.shape} ({next_obs.state.dtype})")
    else:
        logging.info("next_obs: None")

    logging.info("=" * 80)


def main():
    init_logging()

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 加载完整配置
    logging.info("加载pi05_simpler_debug配置...")
    config = _config.get_config("pi05_simpler_debug")

    # 由工厂创建原始 DataConfig
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"原始 DataConfig.repo_id: {data_config.repo_id}")

    # 将 repo_id 扩展为列表以启用 MultiLeRobotDataset
    # 为避免下载其他数据集，默认重复使用同一 repo 两次进行组合调试
    if isinstance(data_config.repo_id, str):
        multi_repo = [data_config.repo_id, data_config.repo_id]
    elif isinstance(data_config.repo_id, (list, tuple)):
        multi_repo = list(data_config.repo_id)
    else:
        logging.error("DataConfig.repo_id 未设置，无法创建数据集")
        sys.exit(1)

    data_config_multi = dataclasses.replace(data_config, repo_id=multi_repo)
    logging.info(f"Multi repo_id: {data_config_multi.repo_id}")

    # 创建 TorchDataLoader（PyTorch 框架下迭代张量）
    logging.info("创建 TorchDataLoader...")
    data_loader = _data_loader.create_torch_data_loader(
        data_config_multi,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="pytorch",
    )

    # 取出一个 batch
    logging.info("迭代一个 batch 进行检查...")
    try:
        it = iter(data_loader)
        observation, actions, next_obs = next(it)

        # 将张量移动到设备（仅在 PyTorch 模式下）
        if hasattr(observation, "image") and observation.image is not None:
            for k in list(observation.image.keys()):
                observation.image[k] = observation.image[k].to(device)
        if hasattr(observation, "state") and observation.state is not None:
            observation.state = observation.state.to(device)
        actions = actions.to(device)
        if next_obs is not None and hasattr(next_obs, "image") and next_obs.image is not None:
            for k in list(next_obs.image.keys()):
                next_obs.image[k] = next_obs.image[k].to(device)
        if next_obs is not None and hasattr(next_obs, "state") and next_obs.state is not None:
            next_obs.state = next_obs.state.to(device)

        # 打印详细信息
        print_observation_info(observation, actions, next_obs)

        # 保存调试输入
        save_path = Path("checkpoints/pi05_simpler_debug/multidata_test/debug_multidata_inputs.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"保存调试输入到: {save_path}")
        torch.save({"observation": observation, "actions": actions, "next_obs": next_obs}, save_path)

        logging.info("数据加载 debug 成功!")
    except StopIteration:
        logging.error("数据集为空，无法迭代 batch")
    except Exception as e:
        logging.error(f"数据加载/迭代失败: {e}")
        import traceback
        traceback.print_exc()
        import pdb; pdb.post_mortem()


if __name__ == "__main__":
    main()