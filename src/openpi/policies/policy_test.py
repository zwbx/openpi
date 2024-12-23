from openpi_client import action_chunk_broker

from openpi.models import exported as _exported
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config


def create_policy_config() -> _policy_config.PolicyConfig:
    model = _exported.PiModel.from_checkpoint("s3://openpi-assets/exported/pi0_aloha_sim/model")

    return _policy_config.PolicyConfig(
        model=model,
        norm_stats=model.norm_stats("huggingface_aloha_sim_transfer_cube"),
        input_layers=[
            aloha_policy.ActInputsRepack(),
            aloha_policy.AlohaInputs(
                action_dim=model.action_dim,
                delta_action_mask=None,
                adapt_to_pi=False,
            ),
        ],
        output_layers=[
            aloha_policy.AlohaOutputs(
                delta_action_mask=None,
                adapt_to_pi=False,
            ),
            aloha_policy.ActOutputsRepack(),
        ],
    )


def test_infer():
    config = create_policy_config()
    policy = _policy_config.create_policy(config)

    example = aloha_policy.make_aloha_example()
    outputs = policy.infer(example)

    assert outputs["qpos"].shape == (config.model.action_horizon, 14)


def test_broker():
    config = create_policy_config()
    policy = _policy_config.create_policy(config)

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["qpos"].shape == (14,)
