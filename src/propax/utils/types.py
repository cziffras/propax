from typing import TypeAlias

from jaxtyping import Array, Bool, Float, Int

jaxFloat: TypeAlias = float | Float[Array, ""] | Float[Array, "..."]  # noqa: F722
jaxInt: TypeAlias = int | Int[Array, ""] | Int[Array, "..."]  # noqa: F722
jaxBool: TypeAlias = bool | Bool[Array, ""] | Bool[Array, "..."]  # noqa: F722
