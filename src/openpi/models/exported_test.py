import pathlib

import jax
import jax.numpy as jnp

import openpi.models.exported as exported
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.training.checkpoints as _checkpoints


def test_sample_actions():
    model = exported.PiModel.from_checkpoint("s3://openpi-assets/exported/pi0_aloha_sim/model")
    actions = model.sample_actions(jax.random.key(0), model.fake_obs(), num_steps=10)

    assert actions.shape == (1, model.action_horizon, model.action_dim)


def test_exported_as_pi0():
    pi_model = exported.PiModel.from_checkpoint("s3://openpi-assets/exported/pi0_aloha_sim/model")
    model = pi_model.set_module(pi0.Module(), param_path="decoder")

    key = jax.random.key(0)
    obs = model.fake_obs()

    pi_actions = pi_model.sample_actions(key, obs, num_steps=10)
    actions = model.sample_actions(key, obs, num_steps=10)

    assert pi_actions.shape == (1, model.action_horizon, model.action_dim)
    assert actions.shape == (1, model.action_horizon, model.action_dim)

    diff = jnp.max(jnp.abs(pi_actions - actions))
    assert diff < 10.0


def test_processor_loading():
    pi_model = exported.PiModel.from_checkpoint("s3://openpi-assets/exported/pi0_aloha_sim/model")
    assert pi_model.processor_names() == ["huggingface_aloha_sim_transfer_cube"]

    norm_stats = pi_model.norm_stats("huggingface_aloha_sim_transfer_cube")
    assert sorted(norm_stats) == ["actions", "state"]


def test_convert_to_openpi(tmp_path: pathlib.Path):
    output_dir = tmp_path / "output"

    exported.convert_to_openpi(
        "s3://openpi-assets/exported/pi0_aloha_sim/model",
        "huggingface_aloha_sim_transfer_cube",
        output_dir,
    )

    # Make sure that we can load the params and norm stats.
    _ = _model.restore_params(output_dir / "params")
    _ = _checkpoints.load_norm_stats(output_dir / "assets")
