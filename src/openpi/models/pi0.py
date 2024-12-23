import logging
from typing import Literal

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import common
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


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


class Module(common.BaseModule):
    """Pi0 module (transfusion-style decoder-only flow matching)."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

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
        pred = self.forward(obs, x_t, timestep, mode="train")
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
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> common.Actions:
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = obs.state.shape[0]
        if noise is None:
            noise = jax.random.normal(self.make_rng("sample"), (batch_size, action_horizon, action_dim))

        # first fill KV cache (in-place)
        self.forward(obs, None, None, mode="fill_cache")

        @at.typecheck
        def sample_step(
            module: Module,
            carry: tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, ""]],
        ) -> tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, ""]]:
            x_t, time = carry
            time_batched = einops.repeat(time, "-> b", b=batch_size)
            v_t = module.forward(obs, x_t, time_batched, mode="decode")
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

    @nn.compact
    @at.typecheck
    def forward(
        self,
        obs: common.Observation,
        noisy_actions: at.Float[at.Array, "b ah ad"] | None,
        timestep: at.Float[at.Array, " b"] | None,
        mode: Literal["train", "fill_cache", "decode"],
    ):
        """Main forward pass of the transformer. It operates in 3 modes:

        1. mode="train": This is full forward pass, used during training.
        2. mode="fill_cache": This is used to compute the KV cache for the prefix (image + language inputs).
        3. mode="decode": This is used to perform a flow matching integration step; it uses the KV cache computed in the
            fill_cache mode.
        """
        paligemma_scope = self.scope.push("PaliGemma")
        llm_scope = paligemma_scope.push("llm")
        img_scope = paligemma_scope.push("img")

        paligemma_config = _gemma.get_config(self.paligemma_variant)
        action_expert_config = _gemma.get_config(self.action_expert_variant)
        gemma = _gemma.Module(
            configs=[paligemma_config, action_expert_config],
            embed_dtype=self.dtype,
            parent=llm_scope,
        )
        siglip = _siglip.Module(
            num_classes=paligemma_config.width,
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm=self.dtype,
            parent=img_scope,
        )

        batch_size = obs.state.shape[0]

        input_mask: list[at.Bool[at.Array, "b s"]] = []
        ar_mask: list[int] = []

        if mode in ["train", "fill_cache"]:
            prefix_tokens: list[at.Float[at.Array, "b s emb"]] = []
            # embed images
            for name in obs.images:
                image_tokens, _ = siglip(obs.images[name], train=False)

                prefix_tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(
                        obs.image_masks[name],
                        "b -> b s",
                        s=image_tokens.shape[1],
                    )
                )
                # image tokens attend to each other
                ar_mask += [0] * image_tokens.shape[1]

            # add language (aka tokenized inputs)
            if obs.tokenized_prompt is not None:
                # run gemma in embed-only mode
                tokenized_inputs = gemma(tokens=obs.tokenized_prompt, embedded=None)
                prefix_tokens.append(tokenized_inputs)
                input_mask.append(obs.tokenized_prompt_mask)
                # full attention between image and language inputs
                ar_mask += [0] * tokenized_inputs.shape[1]
            prefix_tokens = jnp.concatenate(prefix_tokens, axis=1)
            prefix_len = prefix_tokens.shape[1]

        if mode in ["train", "decode"]:
            assert noisy_actions is not None

            suffix_tokens: list[at.Float[at.Array, "b s emb"]] = []
            # add a single state token
            state_token = nn.Dense(action_expert_config.width, name="state_proj")(obs.state)
            suffix_tokens.append(state_token[:, None, :])
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [1]

            action_horizon = noisy_actions.shape[1]
            # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
            time_emb = posemb_sincos(timestep, action_expert_config.width, min_period=4e-3, max_period=4.0)
            # mix timestep + action information using an MLP
            action_tokens = nn.Dense(action_expert_config.width, name="action_in_proj")(noisy_actions)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = nn.Dense(action_expert_config.width, name="action_time_mlp_in")(action_time_tokens)
            action_time_tokens = nn.swish(action_time_tokens)
            action_time_tokens = nn.Dense(action_expert_config.width, name="action_time_mlp_out")(action_time_tokens)
            # add to input tokens
            suffix_tokens.append(action_time_tokens)
            input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
            # image/language/state inputs do not attend to action tokens
            ar_mask += [1] + ([0] * (action_horizon - 1))

            suffix_tokens = jnp.concatenate(suffix_tokens, axis=1)
            suffix_len = suffix_tokens.shape[1]

        if mode == "train":
            # due to prefix-lm decoding, it is very important that the prefix cannot attend to the suffix
            assert ar_mask[prefix_len] == 1

        # create attention mask (shared between prefix and suffix)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = np.array(ar_mask, dtype=np.int32)

        ar_mask = einops.repeat(ar_mask, "s -> b s", b=batch_size)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        if mode in ["train", "decode"]:
            out_proj = nn.Dense(noisy_actions.shape[-1], name="action_out_proj")

        if mode == "train":
            # full forward pass on prefix + suffix at once
            positions = jnp.cumsum(input_mask, axis=1) - 1
            _, out = gemma(
                tokens=None,
                embedded=[prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                decode=False,
            )
            return out_proj(out[:, -action_horizon:])
        if mode == "fill_cache":
            # fill the KV cache using the prefix tokens. this mutates the "cache" variable in place.
            self.put_variable("cache", "prefix_mask", input_mask.astype(bool))
            positions = jnp.cumsum(input_mask, axis=-1) - 1
            gemma(
                tokens=None,
                embedded=[prefix_tokens, None],
                positions=positions,
                mask=attn_mask,
                decode=True,
            )
            return None
        if mode == "decode":
            # decode using the existing KV cache
            prefix_len = gemma.variables["cache"]["layers"]["attn"]["k_cache"].shape[2]
            # `prefix_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_mask = self.get_variable("cache", "prefix_mask")
            assert prefix_mask.shape == (batch_size, prefix_len)
            prefix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            combined_mask = jnp.concatenate([prefix_mask, attn_mask], axis=-1)
            assert combined_mask.shape == (
                batch_size,
                suffix_len,
                prefix_len + suffix_len,
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = (
                jnp.sum(self.get_variable("cache", "prefix_mask"), axis=-1)[:, None]
                + jnp.cumsum(input_mask, axis=-1)
                - 1
            )
            unused, out = gemma(
                tokens=None,
                embedded=[None, suffix_tokens],
                mask=combined_mask,
                positions=positions,
                decode=True,
            )
            assert unused is None
            return out_proj(out[:, -action_horizon:])
        raise ValueError(f"Invalid mode: {mode}")
