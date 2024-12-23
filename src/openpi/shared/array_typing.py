import contextlib
import functools as ft
import inspect

import beartype
import jax
import jax.core
from jaxtyping import Array  # noqa: F401
from jaxtyping import ArrayLike
from jaxtyping import Bool  # noqa: F401
from jaxtyping import DTypeLike  # noqa: F401
from jaxtyping import Float
from jaxtyping import Int  # noqa: F401
from jaxtyping import Key  # noqa: F401
from jaxtyping import Num  # noqa: F401
from jaxtyping import PyTree
from jaxtyping import Real  # noqa: F401
from jaxtyping import Shaped
from jaxtyping import UInt8  # noqa: F401
from jaxtyping import config
from jaxtyping import jaxtyped
import jaxtyping._decorator

# patch jaxtyping to handle https://github.com/patrick-kidger/jaxtyping/issues/277.
# the problem is that custom PyTree nodes are sometimes initialized with arbitrary types (e.g., `jax.ShapeDtypeStruct`,
# `jax.Sharding`, or even <object>) due to JAX tracing operations. this patch skips typechecking when the stack trace
# contains `jax._src.tree_util`, which should only be the case during tree unflattening.
_original_check_dataclass_annotations = jaxtyping._decorator._check_dataclass_annotations  # noqa: SLF001


def _check_dataclass_annotations(self, typechecker):
    if not any(frame.frame.f_globals["__name__"] == "jax._src.tree_util" for frame in inspect.stack()):  # noqa: RET503
        return _original_check_dataclass_annotations(self, typechecker)


jaxtyping._decorator._check_dataclass_annotations = _check_dataclass_annotations  # noqa: SLF001

KeyArrayLike = jax.typing.ArrayLike

Params = PyTree[Float[ArrayLike, "..."]]
Batch = PyTree[Shaped[ArrayLike, "b ..."]]

# runtime type-checking decorator
typecheck = ft.partial(jaxtyped, typechecker=beartype.beartype)


@contextlib.contextmanager
def disable_typechecking():
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)  # noqa: FBT003
    yield
    config.update("jaxtyping_disable", initial)
