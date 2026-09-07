import jax.numpy as jnp

from .types import jaxBool, jaxFloat


def pick(condition: jaxBool, when_true, when_false) -> jnp.ndarray:
    return jnp.asarray(jnp.where(condition, when_true, when_false))


def relative_scale(x: jaxFloat) -> jnp.ndarray:
    """
    `width / x` stops meaning anything as x approaches zero, so below unity the
    comparison falls back to absolute.
    """
    return jnp.maximum(jnp.abs(jnp.asarray(x)), 1.0)


def safe_div(numerator: jaxFloat, denominator: jaxFloat) -> jnp.ndarray:
    """
    NOTE :
    Reverse mode differentiates both arms of a `where`, so guarding the result
    still leaves a 0/0 in the arm not taken and NaN in the cotangent.
    """
    return jnp.asarray(
        numerator / jnp.where(jnp.asarray(denominator) == 0.0, 1.0, denominator)
    )
