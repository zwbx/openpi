import collections
import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params: ...


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/pi0_aloha_sim/exp/1000/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/pi0_base/params"

    Will use EMA parameters if available.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        return _model.restore_params(download.maybe_download(self.params_path))


def _recover_tree(d: dict) -> dict:
    """Recover a tree from a flat dict delimited by '/'. Only used for big_vision weights."""
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in d.items():
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        tree[k] = _recover_tree(dict(kv_pairs))
    return tree


_MODULE_NUM_RE = re.compile(r"(.*)_\d+$")


def _convert_pre_linen(params: at.Params) -> at.Params:
    """Copied from big_vision."""
    if not isinstance(params, dict):
        return params

    params_renamed = {}
    counts = {}
    names = sorted(params.keys())
    for name in names:
        value = params[name]
        match = _MODULE_NUM_RE.match(name)
        if match:
            module = match.group(1)
            num = counts.get(module, 0)
            name = f"{module}_{num}"  # noqa: PLW2901
            counts[module] = num + 1
        params_renamed[name] = _convert_pre_linen(value)

    return params_renamed


def _fix_groupnorm(params: at.Params) -> at.Params:
    """Copied from big_vision."""
    regex = re.compile(r"gn(\d+|_root|_proj)$")

    def fix_gn(args):
        path, array = args
        if len(path) > 1 and regex.match(path[-2]) and path[-1] in ("bias", "scale"):
            array = array.squeeze()
        return (path, array)

    return flax.traverse_util.unflatten_dict(dict(map(fix_gn, flax.traverse_util.flatten_dict(params).items())))


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint. Compatible with the Pi0 model. Weights from the PaliGemma
    expert will be overwritten whereas weights from the action expert will remain untouched.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        # The weights are stored in a special big_vision format, so we need a special function to unflatten them.
        paligemma_params = _recover_tree(flat_params)["params"]

        # Now we will do our own flattening to merge the PaliGemma weights with the action expert weights.
        leaves, treedef = jax.tree_util.tree_flatten_with_path(params["PaliGemma"])
        leaves = dict(leaves)
        for kp, v in jax.tree_util.tree_flatten_with_path(paligemma_params)[0]:
            if kp in leaves:
                logger.info(f"Overwriting {jax.tree_util.keystr(kp)}")
                leaves[kp] = v

        new_paligemma_params = jax.tree_util.tree_unflatten(treedef, leaves.values())
        return {**params, "PaliGemma": new_paligemma_params}


@dataclasses.dataclass(frozen=True)
class GoogleViTWeightLoader(WeightLoader):
    """Loads weights from a Google ViT checkpoint. Compatible with the Pi0-small model. Only overwrites weights from the
    image backbones in the encoder.
    """

    # The Google ViT can take any resolution, including non-square ones. The resolution specified here must match the
    # resolution of the images passed into the model.
    target_resolution: tuple[int, int] = (224, 224)

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
            gs={"token": "anon"},
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        pretrained_params = _fix_groupnorm(_convert_pre_linen(_recover_tree(flat_params)))

        # convert the weights to scanned format
        encoderblocks = [
            pretrained_params["Transformer"].pop(f"encoderblock_{i}")
            for i in range(sum("encoderblock" in k for k in pretrained_params["Transformer"]))
        ]
        pretrained_params["Transformer"]["encoderblock"] = jax.tree.map(lambda *x: jnp.stack(x), *encoderblocks)

        # -- INTERPOLATE POSITION EMBEDDINGS --
        # retrieve pretrained position embeddings
        old_posemb = pretrained_params["Transformer"]["posembed_input"]["pos_embedding"]  # (1, num_tokens, emb_dim)
        if old_posemb.shape[0] != 1:
            raise ValueError(f"Invalid shape: {old_posemb.shape=}")
        old_posemb = old_posemb[0]  # (num_tokens, emb_dim)
        if "cls" in pretrained_params:
            old_posemb = old_posemb[1:]  # remove class token
        # pretrained models all take square inputs
        size = np.sqrt(len(old_posemb))
        if not np.isclose(size, round(size)):
            raise ValueError(f"Number of tokens {len(old_posemb)} is not a perfect square")
        old_h = old_w = round(size)

        # remove cls token and head
        del pretrained_params["cls"]
        del pretrained_params["head"]

        for key in params["encoder"]:
            if not key.startswith("backbone_"):
                continue
            # retrieve newly initialized position embeddings
            vit_params = params["encoder"][key]["VisionTransformer"]
            new_posemb = vit_params["Transformer"]["posembed_input"]["pos_embedding"]  # (1, num_tokens, emb_dim)
            if new_posemb.shape[0] != 1:
                raise ValueError(f"Invalid shape: {new_posemb.shape=}")
            new_posemb = new_posemb[0]  # (num_tokens, emb_dim)
            if "cls" in vit_params:
                raise ValueError("Initialized Google ViT should not have a class token")
            # we need the target resolution just to compute the aspect ratio, because the google_vit model is
            # implemented in a way that does not expose this information
            aspect_ratio = self.target_resolution[0] / self.target_resolution[1]
            # compute the original height and width that was flattened into new_n_tokens
            new_h, new_w = np.sqrt(len(new_posemb) / aspect_ratio), np.sqrt(len(new_posemb) * aspect_ratio)
            if not np.isclose(new_h, round(new_h)) or not np.isclose(new_w, round(new_w)):
                raise ValueError(
                    f"Target resolution {self.target_resolution} with aspect ratio {aspect_ratio} "
                    f"does not match number of tokens {len(new_posemb)}"
                )
            new_h, new_w = round(new_h), round(new_w)

            # perform the interpolation
            logger.info(f"Interpolating position embeddings from {old_h}x{old_w} to {new_h}x{new_w} for {key=}")
            old_posemb = old_posemb.reshape(old_h, old_w, -1)
            zoom = (new_h / old_h, new_w / old_w, 1)
            new_posemb = scipy.ndimage.zoom(old_posemb, zoom, order=1)  # bilinear interpolation
            new_posemb = new_posemb.reshape(1, new_h * new_w, -1)
            assert new_posemb.shape == vit_params["Transformer"]["posembed_input"]["pos_embedding"].shape

            # copy over the weights
            pretrained_params_for_key = jax.tree.map(lambda x: x, pretrained_params)
            pretrained_params_for_key["Transformer"]["posembed_input"]["pos_embedding"] = new_posemb
            params["encoder"][key]["VisionTransformer"] = pretrained_params_for_key

        return params
