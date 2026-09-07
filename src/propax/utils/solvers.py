from functools import lru_cache, partial

import jax
import jax.numpy as jnp

from .numerics import pick, relative_scale, safe_div


def newton_loop(func, lower, upper, args, *, x0=None, max_steps, rtol):
    """A Newton that cannot leave its bracket.

    `func` returns `(value, derivative)`

    Each step tightens the enclosure on the sign first, then takes
    the Newton step only if it lands strictly inside the tightened bracket;
    otherwise it halves.

    `x0` starts it somewhere other than the middle, for a caller holding a
    guess. It is clipped into the bracket, never trusted to be inside it.

    NOTE :
    - This solver as is is undifferentiable and unjitted: the
    callers supply their own derivative from the implicit function
    theorem, so nothing inside needs one.
    - It stops on the step, not on the bracket width: a Newton approaches
    from one side, so the far endpoint never moves and a width test would
    run every lane to `max_steps` which under `vmap` is paid by the whole
    batch.
    - Tried the Halley step that required a second order derivative; the gain
    is not susbtantial and in some cases slows down computation since there
    is an additional computation at each step.
    """
    f_lo = func(lower, args)[0]
    f_hi = func(upper, args)[0]
    bracket = jnp.sign(f_lo) * jnp.sign(f_hi)

    def keep_going(state):
        lo, hi, x, step, moved, f = state
        # scaling factor for rtol
        denom = relative_scale(x)
        settled = moved / denom <= rtol
        exhausted = (hi - lo) / denom <= rtol
        return (step < max_steps) & jnp.logical_not(settled | exhausted)

    def body_fn(state):
        lo, hi, x, step, _, _ = state
        f, df = func(x, args)

        same = f * f_lo > 0.0
        lo_new = jnp.where(same, x, lo)
        hi_new = jnp.where(same, hi, x)

        # A Newton step that overshoots the bracket is still pointing the right
        # way; rejecting it for the midpoint throws away every bit of progress
        # made so far, and on a strongly convex residual makes the method
        # degenerate into the bisection it was meant to improve on : we decided
        # to clip it but still keeping the direction the step is pointing at point
        # is also not to cost extra evaluations (Armijo rule etc...)
        #
        # Update: made the clip unconditional, we used to guard only steps leaving the brackets
        # however some were hitting the bounds and alternating between them without
        # firing while outrageously diverging !
        candidate = x - safe_div(f, df)
        damped = jnp.clip(candidate, x - 0.5 * (x - lo_new), x + 0.5 * (hi_new - x))
        x_new = jnp.where(jnp.isfinite(candidate), damped, 0.5 * (lo_new + hi_new))
        return lo_new, hi_new, x_new, step + 1, jnp.abs(x_new - x), f

    start = 0.5 * (lower + upper) if x0 is None else jnp.clip(x0, lower, upper)
    _, _, x, _, _, _ = jax.lax.while_loop(
        keep_going,
        body_fn,
        (
            lower,
            upper,
            start,
            jnp.asarray(0),
            jnp.asarray(jnp.inf),
            jnp.asarray(jnp.inf),
        ),
    )
    # an endpoint that already is the root needs no iteration, and the loop
    # above cannot converge onto it: nothing tells it to move that way
    x = pick(f_lo == 0.0, lower, pick(f_hi == 0.0, upper, x))
    return x, bracket


@lru_cache(maxsize=None)
def get_bisect_core(max_steps, rtol, atol):
    """Build (and remember through caching) the jitted core for one set of stopping rules.

    Cached on the arguments so the same settings return the same function
    object: `jax.jit` keys its cache on identity.
    """

    @partial(jax.custom_jvp, nondiff_argnums=(0,))
    def _bisect_core(func, lower, upper, args):
        sign_lo_init = jnp.sign(func(lower, args))
        # the endpoints' sign product decides whether a root is bracketed at
        # all; without it the width test below reports success on any interval
        bracket = sign_lo_init * jnp.sign(func(upper, args))

        def keep_going(state):
            lo, hi, _, step = state
            width = hi - lo
            mid = 0.5 * (lo + hi)
            denom = relative_scale(mid)
            narrow = (width <= atol) | (width / denom <= rtol)
            return (step < max_steps) & jnp.logical_not(narrow)

        def body_fn(state):
            lo, hi, sign_lo, step = state
            mid = (lo + hi) / 2.0
            sign_mid = jnp.sign(func(mid, args))
            same_sign = sign_lo == sign_mid
            return (
                jnp.where(same_sign, mid, lo),
                jnp.where(same_sign, hi, mid),
                jnp.where(same_sign, sign_mid, sign_lo),
                step + 1,
            )

        # The interval is halved until it meets the
        # tolerance, and `max_steps` is only a cap
        # keeping a pathological lane from running forever
        final_lo, final_hi, _, _ = jax.lax.while_loop(
            keep_going, body_fn, (lower, upper, sign_lo_init, jnp.asarray(0))
        )

        sol = (final_lo + final_hi) / 2.0
        width = final_hi - final_lo

        return sol, width, bracket

    @_bisect_core.defjvp
    def _through_the_root(func, primals, tangents):
        """Implicit function theorem: f(x(a), a) = 0 fixes dx/da = -(df/da)/(df/dx)."""
        *_, args = primals
        *_, pushed = tangents

        root, width, bracket = _bisect_core(func, *primals)

        def at_the_root(moved_args):
            return func(root, moved_args)

        slope = jax.grad(func, argnums=0)(root, args)
        _, carried = jax.jvp(at_the_root, (args,), (pushed,))
        held = (jnp.zeros_like(width), jnp.zeros_like(bracket))

        return (root, width, bracket), (-carried / slope, *held)

    return _bisect_core


@partial(jax.jit, static_argnums=(0,), static_argnames=("max_steps", "rtol", "atol"))
def bisect(func, lower, upper, args, *, max_steps, rtol, atol):
    """The root of `func` in [lower, upper], and whether it was really found."""
    core_func = get_bisect_core(max_steps, rtol, atol)

    sol, width, bracket = core_func(func, lower, upper, args)

    denom = relative_scale(sol)
    narrow = (width <= atol) | (width / denom <= rtol)
    converged = narrow & (bracket <= 0.0)

    return sol, converged
