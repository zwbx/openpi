import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from typing_extensions import override

from openpi.models import common
import openpi.models.transformer as _transformer
import openpi.models.vit as _vit
from openpi.shared import array_typing as at


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class ViTEncoder(nn.Module):
    """ViT encoder from the Google vision_transformer codebase."""

    dtype: str = "bfloat16"

    @nn.compact
    @at.typecheck
    def __call__(self, image: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b seq emb"]:
        vit = _vit.VisionTransformer(
            name="VisionTransformer",
            dtype=self.dtype,
            # Removes class token.
            num_classes=None,
            classifier="unpooled",
            # R26+ViT-S_32 config.
            patches=ml_collections.ConfigDict({"size": (1, 1)}),
            transformer=ml_collections.ConfigDict({"mlp_dim": 1536, "num_heads": 6, "num_layers": 12}),
            hidden_size=384,
            resnet=ml_collections.ConfigDict({"num_layers": (2, 2, 2, 2), "width_factor": 1}),
        )

        # VisionTransformer expects images in [0, 1] range.
        image = (image + 1) / 2
        return vit(image, train=False)


class Encoder(nn.Module):
    """Transformer encoder that combines ViTEncoders for each image, plus state information."""

    variant: _transformer.Variant = "small"
    dtype: str = "bfloat16"

    @nn.compact
    @at.typecheck
    def __call__(self, obs: common.Observation) -> _transformer.TokenSequence:
        transformer, embed_dim = _transformer.get_variant(self.variant, dtype=self.dtype)

        image_tokens: list[_transformer.TokenSequence] = []
        for name in obs.images:
            zimg = ViTEncoder(name=f"backbone_{name}", dtype=self.dtype)(obs.images[name])
            zimg = nn.Dense(embed_dim, name=f"proj_{name}")(zimg)
            posemb = self.param(f"posemb_image_{name}", nn.initializers.normal(0.02), (embed_dim,))
            image_tokens.append(
                _transformer.TokenSequence(
                    tokens=zimg,
                    pos=jnp.broadcast_to(posemb, zimg.shape),
                    mask=jnp.broadcast_to(obs.image_masks[name][:, None], zimg.shape[:-1]),
                )
            )

        state_token = _transformer.TokenSequence(
            tokens=nn.Dense(embed_dim, name="state_proj")(obs.state)[:, None],
            pos=self.param("posemb_state", nn.initializers.normal(0.02), (embed_dim,))[None],
        )

        input_tokens = _transformer.TokenSequence.concatenate(*image_tokens, state_token)

        return transformer(input_tokens)


class Decoder(nn.Module):
    variant: _transformer.Variant = "small"
    dtype: str = "bfloat16"

    @nn.compact
    @at.typecheck
    def __call__(
        self,
        noisy_actions: at.Float[at.Array, "b ah ad"],
        timestep: at.Float[at.Array, " b"],
        cond_tokens: _transformer.TokenSequence,
    ) -> at.Float[at.Array, "b ah ad"]:
        transformer, embed_dim = _transformer.get_variant(self.variant, dtype=self.dtype)

        tokens = _transformer.TokenSequence(
            # project actions to embedding dimension
            tokens=nn.Dense(embed_dim, name="in_proj")(noisy_actions),
            # use learned positional embedding for actions
            pos=self.param("posemb_actions", nn.initializers.normal(0.02), (noisy_actions.shape[1], embed_dim)),
        )

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, embed_dim, min_period=4e-3, max_period=4.0)
        # time MLP
        time_emb = nn.Dense(embed_dim, name="time_mlp_in")(time_emb)
        time_emb = nn.swish(time_emb)
        time_emb = nn.Dense(embed_dim, name="time_mlp_out")(time_emb)

        output_tokens = transformer(tokens, xattn_cond=cond_tokens, adaln_cond=time_emb)
        return nn.Dense(noisy_actions.shape[-1], name="out_proj")(output_tokens.tokens)


class Module(common.BaseModule):
    encoder: Encoder = Encoder()
    decoder: Decoder = Decoder()

    @at.typecheck
    @override
    def compute_loss(
        self,
        obs: common.Observation,
        target_actions: common.Actions,
        *,
        timestep: at.Float[at.Array, " b"] | None = None,
    ) -> at.Float[at.Array, "b ah"]:
        batch_size = target_actions.shape[0]

        noise = jax.random.normal(self.make_rng("loss"), target_actions.shape)
        if timestep is None:
            timestep = jax.random.beta(self.make_rng("loss"), 1.5, 1, (batch_size,)) * 0.999 + 0.001

        time_expanded = timestep[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * target_actions
        u_t = noise - target_actions
        pred = self.decoder(x_t, timestep, self.encoder(obs))
        return jnp.mean(jnp.square(pred - u_t), axis=2)

    @at.typecheck
    @override
    def sample_actions(
        self,
        action_horizon: int,
        action_dim: int,
        obs: common.Observation,
        *,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        num_steps: int = 10,
    ) -> common.Actions:
        dt = -1.0 / num_steps
        batch_size = obs.state.shape[0]
        if noise is None:
            noise = jax.random.normal(self.make_rng("sample"), (batch_size, action_horizon, action_dim))

        cond_tokens = self.encoder(obs)

        @at.typecheck
        def sample_step(
            module: Module,
            carry: tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, ""]],
        ) -> tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, ""]]:
            x_t, time = carry
            time_batched = einops.repeat(time, "-> b", b=batch_size)
            v_t = module.decoder(x_t, time_batched, cond_tokens)
            # Euler step
            x_tilde = x_t + dt * v_t
            return x_tilde, time + dt

        @at.typecheck
        def cond_fn(
            module: Module,
            carry: tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, ""]],
        ) -> at.Bool[at.Array, ""]:
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        time = jnp.array(1.0, dtype=jnp.float32)
        x_0, _ = nn.while_loop(cond_fn, sample_step, self, (noise, time))
        return x_0
