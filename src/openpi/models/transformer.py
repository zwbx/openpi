from collections.abc import Callable
import enum
import functools as ft
import logging
from typing import Literal

import einops
from flax import struct
import flax.linen as nn
from flax.linen import dtypes
import jax.ad_checkpoint
import jax.numpy as jnp

import openpi.shared.array_typing as at

logger = logging.getLogger(__name__)

AFTER_ATTN_CHECKPOINT_NAME = "after_attn"
AFTER_XATTN_CHECKPOINT_NAME = "after_xattn"
QKV_CHECKPOINT_NAME = "qkv"


def _custom_dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    broadcast_dropout: bool = True,  # noqa
    dropout_rng=None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,  # noqa
    dtype=None,
    precision=None,
    module=None,
):
    """Mostly copied from nn.dot_product_attention, except for adding checkpointing logic, and enforcing float32 logits
    for stability.
    """
    assert module is None
    assert dropout_rate == 0.0
    assert dropout_rng is None
    assert bias is None

    query, key, value = dtypes.promote_dtype(query, key, value, dtype=dtype)

    # save post-projection query, key, value for backward pass
    query = jax.ad_checkpoint.checkpoint_name(query, QKV_CHECKPOINT_NAME)
    key = jax.ad_checkpoint.checkpoint_name(key, QKV_CHECKPOINT_NAME)
    value = jax.ad_checkpoint.checkpoint_name(value, QKV_CHECKPOINT_NAME)

    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    assert query.dtype == dtype

    # calculate logits in float32 for stability
    logits = jnp.einsum("...qhd,...khd->...hqk", query, key, precision=precision, preferred_element_type=jnp.float32)
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(jnp.float32).min
        logits = jnp.where(mask, logits, big_neg)

    # normalize the attention weights and cast back to the original dtype (if not float32)
    attn_weights = jax.nn.softmax(logits).astype(dtype)

    # return weighted sum over values for each query position
    out = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value, precision=precision)

    assert out.dtype == dtype
    return out


@at.typecheck
@struct.dataclass
class TokenSequence:
    """Holds a sequence of tokens alongside positional information."""

    tokens: at.Float[at.ArrayLike, "b seq emb"]
    # pos may or may not have a batch dimension
    pos: at.Float[at.Array, "b seq emb"] | at.Float[at.Array, "seq emb"]
    # optional masking information
    mask: at.Bool[at.Array, "b seq"] | None = None

    def __len__(self):
        return self.tokens.shape[1]

    @property
    def emb_dim(self):
        return self.tokens.shape[-1]

    @staticmethod
    def concatenate(*sequences: "TokenSequence") -> "TokenSequence":
        """Concatenates multiple sequences along the sequence dimension."""
        tokens = jnp.concatenate([seq.tokens for seq in sequences], axis=1)
        # if any sequence's pos has a batch dimension, broadcast the others to have one
        if any(seq.pos.ndim == 3 for seq in sequences):
            batch_size = next(seq.pos.shape[0] for seq in sequences if seq.pos.ndim == 3)
            pos = jnp.concatenate(
                [
                    seq.pos if seq.pos.ndim == 3 else jnp.broadcast_to(seq.pos, (batch_size, *seq.pos.shape))
                    for seq in sequences
                ],
                axis=1,
            )
        else:
            pos = jnp.concatenate([seq.pos for seq in sequences], axis=0)

        # if any sequence has a mask, create True masks for the others
        if any(seq.mask is not None for seq in sequences):
            mask = jnp.concatenate(
                [
                    seq.mask
                    if seq.mask is not None
                    else jnp.ones((seq.tokens.shape[0], seq.tokens.shape[1]), dtype=jnp.bool_)
                    for seq in sequences
                ],
                axis=1,
            )
        else:
            mask = None

        return TokenSequence(tokens=tokens, pos=pos, mask=mask)


class PosembStrategy(enum.Enum):
    """Controls how positional embeddings are incorporated into the transformer. Configured separately for the
    input sequence and the cross-attention sequence. Note that for cross-attention, ADD_AT_ATTN and
    ADD_AT_BEGINNING are very similar, since the key and value token sequences are the same for every
    attention operation. The only difference is that ADD_AT_ATTN adds the positional embeddings to the key
    sequence only, while ADD_AT_BEGINNING adds them to both the key and value sequences.

    NONE:
        Ignore positional embeddings.
    ADD_AT_BEGINNING:
        Adds the positional embeddings to the token sequence at the beginning of the transformer call.
    ADD_AT_ATTN:
        Adds the positional embeddings to the query and key (but not value) sequences at every attention
        operation.
    """

    NONE = enum.auto()
    ADD_AT_BEGINNING = enum.auto()
    ADD_AT_ATTN = enum.auto()


class AttentionBlock(nn.Module):
    """Implements either self-attention (if q == kv) or cross-attention (if q != kv)."""

    num_heads: int
    normalize_qk: bool = True

    @nn.compact
    @at.typecheck
    def __call__(
        self,
        *,
        q: at.Float[at.Array, "b q_seq q_emb"],
        kv: at.Float[at.Array, "b kv_seq kv_emb"],
        q_pos: at.Float[at.Array, "*bq q_seq q_emb"],
        kv_pos: at.Float[at.Array, "*bkv kv_seq kv_emb"],
        mask: at.Bool[at.Array, "b q_seq kv_seq"] | None = None,
        dtype: at.DTypeLike,
    ) -> at.Float[at.Array, "b q_seq q_emb"]:
        # broadcast mask to have a head dimension
        if mask is not None:
            mask = einops.repeat(mask, "b q_seq kv_seq -> b n q_seq kv_seq", n=self.num_heads)
        # we add posembs to queries and keys, but not values
        q = q + q_pos
        k = kv + kv_pos
        v = kv
        return nn.MultiHeadAttention(
            num_heads=self.num_heads,
            normalize_qk=self.normalize_qk,
            use_bias=False,
            kernel_init=nn.initializers.lecun_normal(),
            attention_fn=_custom_dot_product_attention,
            dtype=dtype,
        )(q, k, v, mask=mask)


class MLPBlock(nn.Module):
    dim: int

    @nn.compact
    @at.typecheck
    def __call__(self, x: at.Float[at.Array, "b seq emb"], *, dtype: at.DTypeLike) -> at.Float[at.Array, "b seq emb"]:
        embed_dim = x.shape[-1]
        # SwiGLU MLP.
        # fuse the first 2 matmuls into one in case it's more efficient
        out = nn.DenseGeneral((2, self.dim), use_bias=False, kernel_init=nn.initializers.lecun_normal(), dtype=dtype)(x)
        gating, hidden = einops.rearrange(out, "b seq n emb -> n b seq emb")
        return nn.Dense(embed_dim, use_bias=False, kernel_init=nn.initializers.lecun_normal(), dtype=dtype)(
            nn.swish(gating) * hidden
        )


class AdaLNGeneral(nn.Module):
    """Generalized LayerNorm module, optionally adaptive based on conditioning information.

    If `cond` is None, applies standard LayerNorm with learned scale and bias. If `cond` is not None, applies
    adaptive LayerNorm (AdaLN):

    >>> out = LayerNorm(x) * (1 + scale) + shift

    Where `scale` and `shift` are learned from conditioning information and initialized to always be 0 (so
    that the output is initially equal to LayerNorm(x)), and LayerNorm here is the version without learned
    parameters.

    If `fn` is not None, this module applies normalization, `fn`, and then a residual connection. For example,
    with `cond == None`:

    >>> out = x + fn(LayerNorm(x))

    With `cond != None`, this becomes AdaLNZero (from the DiT paper):

    >>> out = x + gate * fn(LayerNorm(x) * (1 + scale) + shift)

    where `gate`, `scale`, and `shift` are once again initialized to always be 0, so the output is initially
    equal to the input.
    """

    fn: Callable | None = None

    @nn.compact
    @at.typecheck
    def __call__(
        self,
        x: at.Float[at.Array, "b seq emb"],
        cond: at.Float[at.Array, "b cond_emb"] | at.Float[at.Array, "b seq cond_emb"] | None = None,
        *,
        dtype: at.DTypeLike,
    ) -> at.Float[at.Array, "b seq emb"]:
        if cond is None:
            if self.fn is None:
                return nn.LayerNorm(dtype=dtype)(x)
            return x + self.fn(nn.LayerNorm(dtype=dtype)(x))
        # number of learned AdaLN vectors
        n = 2 if self.fn is None else 3
        adaln = nn.DenseGeneral(
            features=(n, x.shape[-1]),
            kernel_init=nn.zeros,
            dtype=dtype,
        )(nn.swish(cond))
        if cond.ndim == 2:
            adaln = einops.rearrange(adaln, "b n emb -> n b 1 emb")
        elif cond.ndim == 3:
            adaln = einops.rearrange(adaln, "b seq n emb -> n b seq emb")
        else:
            raise ValueError(f"Invalid number of dimensions for cond: {cond.ndim}")

        modulated = nn.LayerNorm(use_bias=False, use_scale=False, dtype=dtype)(x) * (1 + adaln[0]) + adaln[1]

        if self.fn is None:
            return modulated
        return x + adaln[2] * self.fn(modulated)


class TransformerBlock(nn.Module):
    """Transformer block (no attention mask) with optional AdaLN and cross-attention conditioning."""

    attn: AttentionBlock
    mlp: MLPBlock

    @nn.compact
    @at.typecheck
    def __call__(
        self,
        x: TokenSequence,
        xattn_cond: TokenSequence | None = None,
        adaln_cond: at.Float[at.Array, "b adaln_emb"] | at.Float[at.Array, "b seq adaln_emb"] | None = None,
        self_attn_mask: at.Bool[at.Array, "b seq seq"] | None = None,
        *,
        dtype: at.DTypeLike,
    ) -> TokenSequence:
        # if x.mask is not None, apply it to the self-attention mask
        if x.mask is not None:
            if self_attn_mask is None:
                self_attn_mask = jnp.ones((x.tokens.shape[0], x.tokens.shape[1], x.tokens.shape[1]), dtype=jnp.bool_)
            # take the outer product of x.mask with itself to form a full (b, seq, seq) attention mask and then combine
            # it with the existing attention mask
            self_attn_mask = jnp.logical_and(self_attn_mask, jnp.logical_and(x.mask[:, None, :], x.mask[:, :, None]))

        def self_attn_fn(y):
            return self.attn.copy(name="self_attn")(
                q=y, kv=y, q_pos=x.pos, kv_pos=x.pos, mask=self_attn_mask, dtype=dtype
            )

        # self-attention
        tokens = AdaLNGeneral(self_attn_fn)(x.tokens, adaln_cond, dtype=dtype)

        tokens = jax.ad_checkpoint.checkpoint_name(tokens, AFTER_ATTN_CHECKPOINT_NAME)

        # cross-attention
        if xattn_cond is not None:
            # if xattn_cond.mask is not None, generate a cross-attention mask
            if xattn_cond.mask is not None:
                xattn_mask = einops.repeat(xattn_cond.mask, "b xseq -> b seq xseq", seq=x.tokens.shape[1])
            else:
                xattn_mask = None

            def xattn_fn(y):
                return self.attn.copy(name="cross_attn")(
                    q=y, kv=xattn_cond.tokens, q_pos=x.pos, kv_pos=xattn_cond.pos, mask=xattn_mask, dtype=dtype
                )

            tokens = AdaLNGeneral(xattn_fn)(tokens, adaln_cond, dtype=dtype)

        tokens = jax.ad_checkpoint.checkpoint_name(tokens, AFTER_XATTN_CHECKPOINT_NAME)

        # mlp
        tokens = AdaLNGeneral(ft.partial(self.mlp, dtype=dtype))(tokens, adaln_cond, dtype=dtype)

        return x.replace(tokens=tokens)


class Transformer(nn.Module):
    """Transformer stack with optional AdaLN and cross-attention conditioning.

    AdaLN conditioning is a single vector. Cross-attention conditioning is a sequence of vectors, where the
    sequence length may be different from the input sequence length. The input, adaln conditioning, and cross-
    attention conditioning may all have different embedding dimensions.
    """

    num_layers: int
    transformer_block: TransformerBlock
    self_attn_posemb_strategy: PosembStrategy = PosembStrategy.ADD_AT_BEGINNING
    xattn_posemb_strategy: PosembStrategy = PosembStrategy.NONE
    dtype: str = "bfloat16"

    @nn.compact
    @at.typecheck
    def __call__(
        self,
        x: TokenSequence,
        xattn_cond: TokenSequence | None = None,
        adaln_cond: at.Float[at.Array, "b adaln_emb"] | at.Float[at.Array, "b seq adaln_emb"] | None = None,
        self_attn_mask: at.Bool[at.Array, "b seq seq"] | None = None,
    ) -> TokenSequence:
        orig_pos = x.pos  # save because we always want to include it in the output sequence
        # the transformer block always adds positional embeddings, so we disable ADD_AT_ATTN by zeroing them
        # out here
        if self.self_attn_posemb_strategy == PosembStrategy.ADD_AT_BEGINNING:
            x = x.replace(tokens=x.tokens + x.pos)
        if self.self_attn_posemb_strategy != PosembStrategy.ADD_AT_ATTN:
            x = x.replace(pos=jnp.zeros_like(x.pos, dtype=self.dtype))
        x = x.replace(tokens=x.tokens.astype(self.dtype))

        if xattn_cond is not None:
            if self.xattn_posemb_strategy == PosembStrategy.ADD_AT_BEGINNING:
                xattn_cond = xattn_cond.replace(tokens=xattn_cond.tokens + xattn_cond.pos)
            if self.xattn_posemb_strategy != PosembStrategy.ADD_AT_ATTN:
                xattn_cond = xattn_cond.replace(pos=jnp.zeros_like(xattn_cond.pos, dtype=self.dtype))
            xattn_cond = xattn_cond.replace(tokens=xattn_cond.tokens.astype(self.dtype))

        if adaln_cond is not None:
            adaln_cond = adaln_cond.astype(self.dtype)

        def block_call(module, x):
            return module(x, xattn_cond, adaln_cond, self_attn_mask, dtype=self.dtype), None

        # Enables rematerialization (aka gradient checkpointing). This configuration saves only the post-projection
        # query, key, and value tensors, as well as the activations after the full attention and cross-attention blocks.
        # This is based on seqax.
        block_call_remat = nn.remat(
            block_call,
            policy=jax.checkpoint_policies.save_only_these_names(
                (AFTER_ATTN_CHECKPOINT_NAME, AFTER_XATTN_CHECKPOINT_NAME, QKV_CHECKPOINT_NAME)
            ),
        )
        # scanning over layers significantly speeds up compilation time
        x, _ = nn.scan(
            block_call_remat,
            length=self.num_layers,
            variable_axes={"params": 0},  # create new parameters for each iteration
            split_rngs={"params": True},
        )(self.transformer_block, x)

        x = x.replace(tokens=AdaLNGeneral(name="final_norm")(x.tokens, adaln_cond, dtype=self.dtype))

        # restore original posemb for downstream use
        return x.replace(pos=orig_pos)


Variant = Literal["dummy", "tiny", "small", "base", "large"]


def get_variant(variant: Variant, **kwargs) -> tuple[Transformer, int]:
    if variant == "dummy":
        return Transformer(
            num_layers=2,
            transformer_block=TransformerBlock(
                attn=AttentionBlock(num_heads=2),
                mlp=MLPBlock(dim=4),
            ),
            **kwargs,
        ), 4
    if variant == "tiny":
        return Transformer(
            num_layers=4,
            transformer_block=TransformerBlock(
                attn=AttentionBlock(num_heads=2),
                mlp=MLPBlock(dim=512),
            ),
            **kwargs,
        ), 128
    if variant == "small":
        return Transformer(
            num_layers=12,
            transformer_block=TransformerBlock(
                attn=AttentionBlock(num_heads=6),
                mlp=MLPBlock(dim=1536),
            ),
            **kwargs,
        ), 384
    if variant == "base":
        return Transformer(
            num_layers=12,
            transformer_block=TransformerBlock(
                attn=AttentionBlock(num_heads=12),
                mlp=MLPBlock(dim=3072),
            ),
            **kwargs,
        ), 768
    if variant == "large":
        return Transformer(
            num_layers=24,
            transformer_block=TransformerBlock(
                attn=AttentionBlock(num_heads=16),
                mlp=MLPBlock(dim=4096),
            ),
            **kwargs,
        ), 1024
    raise ValueError(f"Invalid transformer variant: {variant}")
