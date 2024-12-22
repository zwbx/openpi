from collections.abc import Sequence
import dataclasses
import enum
import logging
from typing import Any

import tyro

from openpi import transforms
from openpi.models import exported as _exported
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import calvin_policy
from openpi.policies import droid_policy
from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import delta_actions
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    CALVIN = "calvin"
    LIBERO = "libero"


@dataclasses.dataclass
class Exported:
    """Load an exported checkpoint."""

    # Checkpoint directory (e.g., "s3://openpi-assets/exported/pi0_aloha/model").
    dir: str
    # Processor name to load the norm stats from. If not provided, the default processor for the environment will be used.
    processor: str | None = None


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for.
    env: EnvMode = EnvMode.ALOHA_SIM
    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Exported | None = None

    # If provided, overrides the default prompt for the policy.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False


def repack_from_env(env: EnvMode) -> transforms.Group:
    """Creates environment specific repack transforms."""
    # TODO(ury): Move this to the runtime.
    match env:
        case EnvMode.ALOHA:
            return transforms.Group(
                inputs=[aloha_policy.ActInputsRepack()],
                outputs=[aloha_policy.ActOutputsRepack()],
            )
        case EnvMode.ALOHA_SIM:
            return transforms.Group(
                inputs=[aloha_policy.ActInputsRepack()],
                outputs=[aloha_policy.ActOutputsRepack()],
            )
        case _:
            return transforms.Group()


# Default exported models.
DEFAULT_EXPORTED: dict[EnvMode, Exported] = {
    EnvMode.ALOHA: Exported(
        dir="s3://openpi-assets/exported/pi0_aloha/model",
        processor="trossen_biarm_single_base_cam_24dim",
    ),
    EnvMode.ALOHA_SIM: Exported(
        dir="s3://openpi-assets/exported/pi0_aloha_sim/model",
        processor="huggingface_aloha_sim_transfer_cube",
    ),
    EnvMode.DROID: Exported(
        dir="s3://openpi-assets/exported/pi0_droid/model",
        processor="openx_droid",
    ),
    EnvMode.CALVIN: Exported(
        dir="s3://openpi-assets/exported/pi0_calvin/model",
        processor="calvin",
    ),
    EnvMode.LIBERO: Exported(
        dir="s3://openpi-assets/exported/pi0_libero/model",
        processor="libero",
    ),
}


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, exported: Exported | None = None
) -> _policy.Policy:
    model: _model.BaseModel
    config: _policy_config.PolicyConfig

    default_exported = DEFAULT_EXPORTED[env]
    if exported:
        checkpoint_dir = exported.dir
        processor = exported.processor or default_exported.processor
    else:
        checkpoint_dir = default_exported.dir
        processor = default_exported.processor
    assert processor, "Default processor must be always set"

    logging.info("Loading model...")
    model = _exported.PiModel.from_checkpoint(checkpoint_dir)

    def make_policy_config(
        input_layers: Sequence[transforms.DataTransformFn],
        output_layers: Sequence[transforms.DataTransformFn],
        sample_kwargs: dict[str, Any] | None = None,
    ):
        sample_kwargs = sample_kwargs or {"num_steps": 10}
        return _policy_config.PolicyConfig(
            model=model,
            norm_stats=model.norm_stats(processor),
            default_prompt=default_prompt,
            input_layers=input_layers,
            output_layers=output_layers,
            sample_kwargs=sample_kwargs,
        )

    logging.info("Creating policy...")
    match env:
        case EnvMode.ALOHA:
            delta_action_mask = delta_actions.make_bool_mask(6, -1, 6, -1)
            config = make_policy_config(
                input_layers=[
                    aloha_policy.ActInputsRepack(),
                    aloha_policy.AlohaInputs(
                        action_dim=model.action_dim,
                        delta_action_mask=delta_action_mask,
                        adapt_to_pi=True,
                    ),
                ],
                output_layers=[
                    aloha_policy.AlohaOutputs(
                        delta_action_mask=delta_action_mask,
                        adapt_to_pi=True,
                    ),
                    aloha_policy.ActOutputsRepack(),
                ],
            )
        case EnvMode.ALOHA_SIM:
            config = make_policy_config(
                input_layers=[
                    aloha_policy.ActInputsRepack(),
                    aloha_policy.AlohaInputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    aloha_policy.AlohaOutputs(),
                    aloha_policy.ActOutputsRepack(),
                ],
            )
        case EnvMode.DROID:
            config = make_policy_config(
                input_layers=[
                    droid_policy.DroidInputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    droid_policy.DroidOutputs(),
                    transforms.SubsampleActions(stride=5),
                ],
            )
        case EnvMode.CALVIN:
            config = make_policy_config(
                input_layers=[
                    calvin_policy.CalvinInputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    calvin_policy.CalvinOutputs(),
                ],
            )
        case EnvMode.LIBERO:
            config = make_policy_config(
                input_layers=[
                    libero_policy.LiberoInputs(action_dim=model.action_dim),
                ],
                output_layers=[
                    libero_policy.LiberoOutputs(),
                ],
            )
        case _:
            raise ValueError(f"Unknown environment mode: {env}")

    return _policy_config.create_policy(config)


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                repack_transforms=repack_from_env(args.env),
                default_prompt=args.default_prompt,
            )
        case Exported():
            return create_default_policy(args.env, default_prompt=args.default_prompt, exported=args.policy)
        case None:
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    logging.info("Creating server...")
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port)

    logging.info("Serving...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
