from collections.abc import Callable
import re
from typing import Any

from flax import struct
import jax
import optax

from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: at.Params
    opt_state: at.PyTree
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: at.Params | None


@at.typecheck
def mask_from_regex(regex: str, pytree: at.PyTree) -> at.PyTree[bool]:
    """Returns a PyTree of the same structure as `pytree` where each leaf is `True` if the leaf's keypath matches the regex.

    Keypaths are generated using `jax.tree_util.keystr`, so they'll typically look something like `['a']['b']['c']['d']`
    (for a plain dictionary).
    """
    compiled = re.compile(regex)
    return jax.tree_util.tree_map_with_path(
        lambda path, _: compiled.fullmatch(jax.tree_util.keystr(path)) is not None, pytree
    )


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
