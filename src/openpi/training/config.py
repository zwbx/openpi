"""See _CONFIGS for the list of available configs."""

from collections.abc import Sequence
import dataclasses
import difflib
import pathlib
from typing import Any, Protocol

import tyro

import openpi.models.common as common
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_small as pi0_small
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
from openpi.shared import delta_actions
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


def default_dataset_root() -> str:
    """Default location for the dataset cache."""
    return str(download.get_cache_dir() / "datasets")


@dataclasses.dataclass
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Contains precomputed normalization stats.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Indicates where the cached dataset should be stored.
    dataset_root: str | None = dataclasses.field(default_factory=default_dataset_root)

    # If true, will disable syncing the dataset from the huggingface hub. Allows training on local-only datasets.
    local_files_only: bool = False


class DataConfigFactory(Protocol):
    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        """Create a data config."""


class FakeDataConfig(DataConfigFactory):
    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        return DataConfig(repo_id="fake")


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # The LeRobot repo id.
    repo_id: str
    # The delta action mask. Each value corresponds to an action dimension and indicates if it should be converted to a delta action.
    # If None, absolute actions are used.
    delta_action_mask: Sequence[bool] | None = None
    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # If true, will adapt the joint and gripper values to match the pi runtime. This useful when
    # fine-tuning a pretrained model.
    adapt_to_pi: bool = False
    # Repack transforms. Default is used if not provided.
    repack_transforms: _transforms.Group | None = None
    # If true, will disable syncing the dataset from the huggingface hub.
    local_files_only: bool = False

    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        norm_stats_path = metadata_dir / self.repo_id / "norm_stats.json"
        norm_stats = _normalize.deserialize_json(norm_stats_path.read_text()) if norm_stats_path.exists() else None

        repack_transforms = self.repack_transforms or _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )

        return DataConfig(
            repo_id=self.repo_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=_transforms.Group(
                inputs=[
                    aloha_policy.AlohaInputs(
                        action_dim=model.action_dim,
                        delta_action_mask=self.delta_action_mask,
                        adapt_to_pi=self.adapt_to_pi,
                    ),
                ],
                outputs=[
                    aloha_policy.AlohaOutputs(
                        delta_action_mask=self.delta_action_mask,
                        adapt_to_pi=self.adapt_to_pi,
                    ),
                ],
            ),
            model_transforms=_transforms.Group(
                inputs=[
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model.max_token_len),
                        default_prompt=self.default_prompt,
                    ),
                ]
            ),
            local_files_only=self.local_files_only,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Number of action dimensions.
    action_dim: int = 24
    # Number of action steps in the horizon.
    action_horizon: int = 50
    # Maximum token length for the prompt.
    max_token_len: int = 48

    # The Flax module representing the neural network implementation; must adhere to the BaseModule interface. We can put
    # it directly into the config like this because unbound Flax modules are just dataclasses.
    module: common.BaseModule = dataclasses.field(default_factory=pi0.Module)
    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for metadata (e.g., norm stats).
    metadata_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often to log training metrics.
    log_interval: int = 100
    # How often to save checkpoints.
    save_interval: int = 1000
    # How often to keep checkpoints.
    keep_interval: int = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # Keyword arguments to pass to the policy's sample method.
    sample_kwargs: dict[str, Any] | None = None

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def metadata_dir(self) -> pathlib.Path:
        """Get the metadata directory for this config."""
        return (pathlib.Path(self.metadata_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def create_model(self) -> _model.Model:
        """Create a model for this config."""
        return _model.Model(
            module=self.module,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
        )

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_CONFIGS = [
    #
    # pi0 configs.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    #
    # Additional configs.
    #
    TrainConfig(
        name="pi0_paligemma",
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
    ),
    TrainConfig(
        name="pi0_small",
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
    ),
    #
    # Example configs.
    #
    TrainConfig(
        name="aloha_static_cups_open",
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_static_cups_open",
            delta_action_mask=delta_actions.make_bool_mask(6, -1, 6, -1),
            adapt_to_pi=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            # Set this to true if you are using a dataset that is not on the huggingface hub.
            local_files_only=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6
        ),
    ),
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        batch_size=2,
        module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        batch_size=2,
        module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f"Did you mean '{closest[0]}'? " if closest else ""
        if closest:
            raise ValueError(f"Config '{config_name}' not found. Did you mean '{closest_str}'?")
        raise ValueError(f"Config '{config_name}' not found.")

    return _CONFIGS_DICT[config_name]
