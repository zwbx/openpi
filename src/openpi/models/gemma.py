# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Callable, Sequence
import dataclasses
import logging
import math
from typing import Literal

import einops
import flax.linen as nn
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float = 0.0
    # https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    rank_annotation: str = "L"

    def __post_init__(self):
        if self.rank != int(self.alpha):
            logging.warning(
                "Rank and alpha are not the same, this will cause accuracy error when merging LoRA params currently."
            )


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    projection_lora: LoRAConfig | None = None
    projection_kv_lora: LoRAConfig | None = None
    output_lora: LoRAConfig | None = None


Variant = Literal["dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            projection_lora=LoRAConfig(rank=64, alpha=64.0),
            projection_kv_lora=LoRAConfig(rank=64, alpha=64.0),
            output_lora=LoRAConfig(rank=64, alpha=64.0),
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class Einsum(nn.Module):
    shape: tuple[int, ...]
    init_fn: nn.initializers.Initializer

    @nn.compact
    def __call__(self, eqn, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w = self.param("w", self.init_fn, self.shape).astype(dtype)
        return jnp.einsum(eqn, x, w)


_LORA_A_KEY = "lora_a"
_LORA_B_KEY = "lora_b"


@at.typecheck
class LoRAEinsum(nn.Module):
    base: Einsum
    lora_config: LoRAConfig
    merge_eqn: str
    lora_a_init_fn: nn.initializers.Initializer
    lora_b_init_fn: nn.initializers.Initializer

    def setup(self):
        nn.share_scope(self, self.base)

    @nn.compact
    def __call__(self, eqn, x, *, deterministic=True):
        orig_x = x
        eqn_lora_a, eqn_lora_b = self._get_lora_eqn(eqn, self.merge_eqn)
        if self.lora_config.dropout > 0.0:
            x = nn.Dropout(rate=self.lora_config.dropout, deterministic=deterministic)(x)
        lora_a_shape, lora_b_shape = self._parse_shape(self.merge_eqn)
        lora_a = self.param(_LORA_A_KEY, self.lora_a_init_fn, lora_a_shape).astype(x.dtype)
        lora_b = self.param(_LORA_B_KEY, self.lora_b_init_fn, lora_b_shape).astype(x.dtype)
        lora_a = jnp.einsum(eqn_lora_a, x, lora_a)
        lora_b = jnp.einsum(eqn_lora_b, lora_a, lora_b)

        # TODO: scaling_value should ideally be a self.variable however currently base model doesn't allow any
        # auxilary variables.
        scaling_value = (
            self.lora_config.alpha / self.lora_config.rank
            if not self.lora_config.rslora
            else self.lora_config.alpha / math.sqrt(self.lora_config.rank)
        )

        return self.base(eqn, orig_x) + lora_b * scaling_value

    def _get_lora_eqn(self, eqn: str, lora_merge_eqn: str) -> tuple[str, str]:
        """Figure out lora_a and lora_b eqn from eqn and lora_merge_eqn.
        input:
        eqn: x,w->y
        lora_merge_eqn: lora_a,lora_b->w

        output:
        lora_a_eqn: x,lora_a->?
        lora_b_eqn: ?,lora_b->y
        """
        (x_repr, w_repr), y_repr = _parse_einops_eqn(eqn)
        (lora_a_repr, lora_b_repr), w_repr_p = _parse_einops_eqn(lora_merge_eqn)
        assert len(w_repr) == len(self.base.shape), f"w_repr={w_repr}, shape={self.base.shape}"
        assert w_repr == w_repr_p, f"w_repr={w_repr}, w_repr_p={w_repr_p} should be the same."

        # figure out x,lora_a's output annotation by using y and lora_b
        # the way to do this is to:
        # 1. remove the common prefix and suffix from lora_b and y
        # 2. then the ? will be (common prefix) (stripped y) (stripped lora_b)
        # the equation will look like:
        # [(prefix) (stripped y) (lora b)], [(prefix) (lora b) (suffix)] -> [(prefix) (y) (suffix)]
        prefix, _, y_repr_stripped, lora_b_repr_stripped = self._remove_common_prefix_suffix(y_repr, lora_b_repr)
        lora_intermediate_repr = prefix + y_repr_stripped + lora_b_repr_stripped

        eqn_lora_a_lhs = ", ".join([x_repr, lora_a_repr])
        eqn_lora_b_lhs = ", ".join([lora_intermediate_repr, lora_b_repr])
        return eqn_lora_a_lhs + " -> " + lora_intermediate_repr, eqn_lora_b_lhs + " -> " + y_repr

    def _remove_common_prefix_suffix(self, str1, str2):
        # Get the common prefix
        prefix = ""
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                prefix += str1[i]
            else:
                break

        # Get the common suffix
        suffix = ""
        for i in range(1, min(len(str1), len(str2)) + 1):
            if str1[-i] == str2[-i]:
                suffix = str1[-i] + suffix
            else:
                break

        return prefix, suffix, str1[len(prefix) : -len(suffix)], str2[len(prefix) : -len(suffix)]

    def _parse_shape(self, lora_merge_eqn: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        (lora_lhs_part_0, lora_lhs_part_1), lora_rhs = _parse_einops_eqn(lora_merge_eqn)
        ann_to_dim = dict(zip(lora_rhs, self.base.shape, strict=True))
        ann_to_dim[self.lora_config.rank_annotation] = self.lora_config.rank
        return tuple(ann_to_dim[ann] for ann in lora_lhs_part_0), tuple(ann_to_dim[ann] for ann in lora_lhs_part_1)


def merge_lora_params(lora_params: at.PyTree, get_lora_transform_eqn: Callable[[str], str]) -> at.PyTree:
    params = lora_params["params"]
    flattened_params = traverse_util.flatten_dict(params, sep="/")
    merged_params = {}
    for k in flattened_params:
        if _LORA_A_KEY not in k:
            continue
        lora_b_key = k.replace(_LORA_A_KEY, _LORA_B_KEY)
        orig_w_key = k.replace(_LORA_A_KEY, "w")
        assert lora_b_key in flattened_params
        assert orig_w_key in flattened_params
        lora_merge = jnp.einsum(get_lora_transform_eqn(k), flattened_params[k], flattened_params[lora_b_key])
        # TODO: Currently we don't handling lora scaling value here due to the base model doesn't support auxilary
        # variables.
        merged_params[orig_w_key] = flattened_params[orig_w_key] + lora_merge
    for k in flattened_params:
        if _LORA_A_KEY in k or _LORA_B_KEY in k:
            continue
        if k not in merged_params:
            merged_params[k] = flattened_params[k]
    return {"params": traverse_util.unflatten_dict(merged_params, sep="/")}


def _parse_einops_eqn(eqn: str) -> tuple[tuple[str, str], str]:
    lhs, rhs = eqn.split("->")
    lhs_parts = lhs.split(",")
    assert len(lhs_parts) == 2

    def strip_space(s):
        return s.replace(" ", "")

    lhs_parts[0] = strip_space(lhs_parts[0])
    lhs_parts[1] = strip_space(lhs_parts[1])
    rhs = strip_space(rhs)
    return ((lhs_parts[0], lhs_parts[1]), rhs)


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        normed_inputs = normed_inputs * (
            1 + scale
        )  # scale by learned parameter in float32 (matches Flax implementation)
        return normed_inputs.astype(dtype)  # return in original dtype


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, decode: bool):  # noqa: FBT001
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                )
                if config.projection_lora is not None:
                    qkv_einsum = LoRAEinsum(
                        qkv_einsum,
                        config.projection_lora,
                        merge_eqn="3KDL,3KLKH->3KDH",
                        lora_a_init_fn=nn.initializers.lecun_normal(in_axis=-3, out_axis=-1, batch_axis=(0, 1, 3)),
                        lora_b_init_fn=nn.initializers.lecun_normal(in_axis=-3, out_axis=-1, batch_axis=(0, 1, 3)),
                    )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                )
                if config.projection_lora is not None:
                    q_einsum = LoRAEinsum(
                        q_einsum,
                        config.projection_lora,
                        merge_eqn="NDL,NLNH->NDH",
                        lora_a_init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                        lora_b_init_fn=nn.initializers.lecun_normal(in_axis=-3, out_axis=-1, batch_axis=(0, 2)),
                    )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                )
                if config.projection_kv_lora is not None:
                    kv_einsum = LoRAEinsum(
                        kv_einsum,
                        config.projection_kv_lora,
                        merge_eqn="2KDL,2KLKH->2KDH",
                        lora_a_init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                        lora_b_init_fn=nn.initializers.lecun_normal(in_axis=-3, out_axis=-1, batch_axis=(0, 1, 3)),
                    )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if decode:
            if not self.has_variable("cache", "k_cache"):
                # initial prefill
                self.put_variable("cache", "k_cache", k)
                self.put_variable("cache", "v_cache", v)
            else:
                # decoding
                k = jnp.concatenate([self.get_variable("cache", "k_cache"), k], axis=1)
                v = jnp.concatenate([self.get_variable("cache", "v_cache"), v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                )
                if config.projection_lora is not None:
                    out_einsum = LoRAEinsum(
                        out_einsum,
                        config.projection_lora,
                        merge_eqn="NHNL,NLD->NHD",
                        lora_a_init_fn=nn.initializers.lecun_normal(in_axis=(-4, -3), out_axis=(-2, -1)),
                        lora_b_init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: Sequence[Config]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, unused_scan_arg, positions, attn_mask, decode, deterministic=True):  # noqa: FBT002
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                x = RMSNorm(name=_name("pre_attention_norm", i))(x)  # noqa: PLW2901
            pre_attn.append(x)

        post_attn = attn(pre_attn, positions, attn_mask, decode)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)

        out = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x = RMSNorm(name=_name("pre_ffw_norm", i))(x)  # noqa: PLW2901
                x = FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                )(x)
            out.append(x)

        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = jax.tree.map(lambda x, y: x + y, xs, out)

        return xs, unused_scan_arg


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.

    @nn.compact
    @at.typecheck
    def __call__(
        self,
        *,
        tokens: at.Int[at.Array, "b t"] | None,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None] | None,
        positions: at.Int[at.Array, "b t"] | None = None,
        mask: at.Bool[at.Array, "b t s"] | None = None,
        decode: bool = False,
        deterministic: bool = True,
    ) -> at.Float[at.Array, "b t d"] | Sequence[at.Float[at.Array, "b _t _d"] | None]:
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        # embedder for first expert only
        embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )

        if tokens is not None:
            # embed only
            assert embedded is None, "Cannot pass both tokens and embedded"
            return embedder.encode(tokens).astype(self.embed_dtype)

        assert embedded is not None
        assert positions is not None
        assert mask is not None

        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)

        mask = jnp.asarray(mask)[:, None, :, :]

        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5, 6),  # 0=self, 5=decode, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        block = nn.scan(
            block_cls,
            # cache has axis 1 since we want leading dimension to be batch size.
            variable_axes={"params": 0, "cache": 1},
            split_rngs={"params": True, "dropout": True},
            in_axes=nn.broadcast,
            length=self.configs[0].depth,
        )(
            parent=self.scope.push("layers"),
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )

        embedded, _ = block(embedded, (), positions, mask, decode, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [RMSNorm(name=_name("final_norm", i))(e) if e is not None else e for i, e in enumerate(embedded)]


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"
