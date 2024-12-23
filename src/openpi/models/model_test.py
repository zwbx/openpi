import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0
from openpi.shared import download


def make_from_spec(spec: jax.ShapeDtypeStruct):
    return jnp.zeros(shape=spec.shape, dtype=spec.dtype)


def create_pi0_model():
    return _model.Model(module=pi0.Module(), action_dim=24, action_horizon=50, max_token_len=48)


def test_model():
    model = create_pi0_model()

    batch_size = 2
    obs, act = model.fake_obs(batch_size), model.fake_act(batch_size)

    rng = jax.random.key(0)
    model = model.set_params(model.init_params(rng, obs, act))

    loss = model.compute_loss(rng, obs, act)
    assert loss.shape == ()

    actions = model.sample_actions(rng, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_model_restore():
    model = create_pi0_model()

    batch_size = 2
    obs, act = model.fake_obs(batch_size), model.fake_act(batch_size)

    params = _model.restore_params(download.maybe_download("s3://openpi-assets/exported/pi0_aloha_sim/model"))
    model = model.set_params(params)

    rng = jax.random.key(0)
    loss = model.compute_loss(rng, obs, act)
    assert loss.shape == ()

    actions = model.sample_actions(rng, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
