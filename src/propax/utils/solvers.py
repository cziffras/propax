import equinox as eqx
import jax
import jax.numpy as jnp

from .numerics import pick, relative_scale, safe_div


@eqx.filter_custom_jvp
def _newton_loop(params, *, consts, carry, residual, lower, upper, x0, max_steps, rtol):
    f_lo = residual(lower, consts, params, carry)[0]
    f_hi = residual(upper, consts, params, carry)[0]

    def keep_going(state):
        lo, hi, x, step, moved, _ = state
        # scaling factor for rtol
        denom = relative_scale(x)
        settled = moved / denom <= rtol
        exhausted = (hi - lo) / denom <= rtol
        return (step < max_steps) & jnp.logical_not(settled | exhausted)

    def body_fn(state):
        lo, hi, x, step, _, carry = state
        f, df, carry = residual(x, consts, params, carry)

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
        return lo_new, hi_new, x_new, step + 1, jnp.abs(x_new - x), carry

    start = 0.5 * (lower + upper) if x0 is None else jnp.clip(x0, lower, upper)
    _, _, x, _, _, carry = jax.lax.while_loop(
        keep_going,
        body_fn,
        (
            lower,
            upper,
            start,
            jnp.asarray(0),
            jnp.asarray(jnp.inf),
            carry,
        ),
    )
    # an endpoint that already is the root needs no iteration, and the loop
    # above cannot converge onto it: nothing tells it to move that way
    x = pick(f_lo == 0.0, lower, pick(f_hi == 0.0, upper, x))
    return x, carry


@_newton_loop.def_jvp
def _newton_loop_jvp(
    primals, tangents, *, consts, carry, residual, lower, upper, x0, max_steps, rtol
):
    (params,), (params_dot,) = primals, tangents

    def _is_none(leaf):
        return leaf is None

    # for jax grad compatibility : eqx returns None for inactive tangents, we must use partition and combine
    # params and params_dot have the same Pytree structure
    no_tangent = jax.tree.map(_is_none, params_dot, is_leaf=_is_none)
    nondiff, diff = eqx.partition(params, no_tangent, is_leaf=_is_none)
    diff_dot = eqx.filter(params_dot, no_tangent, is_leaf=_is_none, inverse=True)

    x, carry = _newton_loop(
        params,
        consts=consts,
        carry=carry,
        residual=residual,
        lower=lower,
        upper=upper,
        x0=x0,
        max_steps=max_steps,
        rtol=rtol,
    )

    _, f_dot, dfdx = jax.jvp(
        lambda diff: residual(x, consts, eqx.combine(diff, nondiff), carry)[:2],
        (diff,),
        (diff_dot,),
        has_aux=True,
    )
    carry_dot = jax.tree.map(jnp.zeros_like, carry)
    # dx = -(dres/dparams . dparams) / (dres/dx)
    return (x, carry), (-safe_div(f_dot, dfdx), carry_dot)


def newton_loop(params, *, consts, carry, residual, lower, upper, x0, max_steps, rtol):
    """A differentiable Newton that cannot leave its bracket.

    `residual` returns `(value, derivative)`

    Each step tightens the enclosure on the sign first, then takes
    the Newton step only if it lands strictly inside the tightened bracket;
    otherwise it halves.

    `x0` starts it somewhere other than the middle, for a caller holding a
    guess. It is clipped into the bracket, never trusted to be inside it.

    NOTE :
    - This solver only allows differentiating wrt `params` (only positional
    argument in _newton_loop, see eqx.filter_custom_jvp documentation)
    - It stops on the step, not on the bracket width: a Newton approaches
    from one side, so the far endpoint never moves and a width test would
    run every lane to `max_steps` which under `vmap` is paid by the whole
    batch.
    - Tried the Halley step that required a second order derivative; the gain
    is not susbtantial and in some cases slows down computation since there
    is an additional computation at each step.
    """
    # If any of the kwargs passed to the eqx defined jvp has a tangent wrt to
    # the differentiable variable, eqx raises (see extract below):
    # t_args, t_kwargs = jtu.tree_map(_drop_nondiff, tangents, dynamic)
    #   if len(jtu.tree_leaves(t_kwargs)) > 0:
    #       raise ValueError("Received keyword tangent")
    # --> we apply stop gradient on all kwargs

    lower, upper, consts, carry, x0 = jax.tree.map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
        (lower, upper, consts, carry, x0),
    )
    return _newton_loop(
        params,
        consts=consts,
        carry=carry,
        residual=residual,
        lower=lower,
        upper=upper,
        x0=x0,
        max_steps=max_steps,
        rtol=rtol,
    )
