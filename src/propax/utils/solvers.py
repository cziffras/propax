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
