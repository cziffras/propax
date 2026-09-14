from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from propax.fluids.generic import HelmholtzEOS
from propax.utils.solvers import newton_loop
from propax.utils.types import jaxBool

from ...utils.numerics import pick, relative_scale
from ..config import ThermoVar
from ..domain import T_bounds, node_is_valid, rho_bounds
from ..saturation import Superancillary
from ..tolerances import TOL

# What a residual returns where the EOS has nothing to say
_OUT_OF_DOMAIN = 1e6


class _IsobarWalk(NamedTuple):
    """What `_solve_along_isobar` carries from one step to the next."""

    lo: Array  # the bracket, tightened at every step
    hi: Array
    T: Array  # where the next residual is evaluated
    T_last: Array  # the two temperatures already visited and the densities
    rho_last: Array  # found there, which the secant extrapolates through
    T_before: Array
    rho_before: Array
    step: Array
    moved: Array


def _is_a_root(scaled_residual: Array, target: Array, scale: float) -> jaxBool:
    reference = jnp.maximum(jnp.abs(target) / scale, 1.0)
    return jnp.abs(scaled_residual) <= TOL.acc.root_atol * reference


def _one_dim_slope(key: str, solve_for_T: bool, props, derivs) -> Array:
    """d(target)/d(unknown), from the EOS call the residual already made
    for faster newton solving.

    At fixed rho:
    - du/dT = c_v,
    - ds/dT = c_v/T,
    - dh/dT = c_v + dP_dT/rho

    At fixed T:
    - dP/drho outright,
    - ds/drho|_T = -dP_dT/rho^2
    """
    if not solve_for_T:
        if key == ThermoVar.P.internal_key:
            return derivs["dP_drho"]
        # (T, S) also solves for density, and its slope runs the other way
        rho = props[ThermoVar.D]
        return -derivs["dP_dT"] / (rho * rho)
    cv = props[ThermoVar.CVMASS]
    if key == ThermoVar.U.internal_key:
        return cv
    if key == ThermoVar.S.internal_key:
        return cv / props[ThermoVar.T]
    if key == ThermoVar.H.internal_key:
        return cv + derivs["dP_dT"] / props[ThermoVar.D]
    else:
        raise KeyError(f"Key {key} is not 1d bracketable. ")


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _one_dim_root(
    key: str,
    scale: float,
    solve_for_T: bool,
    eos: HelmholtzEOS,
    known: Array,
    target: Array,
    lo: Array,
    hi: Array,
) -> Array:
    def residual(x, _):
        rho, T = (known, x) if solve_for_T else (x, known)
        props, derivs = eos.props_rhoT(rho, T, with_derivatives=True)
        f = (props[key] - target) / scale
        df = _one_dim_slope(key, solve_for_T, props, derivs) / scale
        return pick(jnp.isfinite(f), f, _OUT_OF_DOMAIN), df

    x, _ = newton_loop(
        residual,
        lo,
        hi,
        None,
        max_steps=TOL.caps.newton_steps,
        rtol=TOL.acc.newton_rtol,
    )
    return jnp.asarray(x)


@_one_dim_root.defjvp
def _one_dim_root_jvp(key, scale, solve_for_T, primals, tangents):
    eos, known, target, lo, hi = primals
    x = _one_dim_root(key, scale, solve_for_T, eos, known, target, lo, hi)

    def value(x_, known_):
        rho, T = (known_, x_) if solve_for_T else (x_, known_)
        return eos.props_rhoT(rho, T)[key]

    d_dx, d_dknown = jax.jacfwd(value, argnums=(0, 1))(x, known)
    # implicit function theorem on y(x, known) = target
    x_dot = (tangents[2] - d_dknown * tangents[1]) / d_dx
    return x, x_dot


def _bracket_for_T(
    eos: HelmholtzEOS, saturation: Superancillary, rho: Array
) -> Tuple[Array, Array]:
    """[T_lo, T_hi] holding the single root of y(rho, T) = y, and no other.

    c_v > 0 makes the target monotone in T, but only outside the dome. The floor
    is therefore T_sat(rho), which the superancillary inverts directly, and the
    whole range for a density that never meets the curve.

    Just below T_sat rather than on it: a saturated state has its root exactly
    there, and a floor solved to tolerance can land above, outside its own
    bracket.
    """
    T_lo, T_hi = (jnp.asarray(b) for b in T_bounds(eos))
    T_sat, on_dome = saturation.T_of_rho(jax.lax.stop_gradient(rho))
    lo = pick(on_dome, T_sat * (1.0 - TOL.acc.sat_edge_nudge), T_lo)
    return jax.lax.stop_gradient(lo), jnp.asarray(T_hi)


def _bracket_for_rho(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    T: Array,
    tvar_other: ThermoVar,
    val_other: Array,
) -> Tuple[Array, Array]:
    """[rho_lo, rho_hi] on one side of the dome, where dP/drho >= 0 holds.

    Which side `val_other` says: against P_sat(T) when it is the pressure, and
    against the midpoint of the saturated pair when it is caloric. Above T_crit
    there is no branch to anchor on, and anchoring would invert the bracket.

    Colder than the curve is fitted for, the pair is read at its floor instead.
    rho_L falls with T and rho_V rises, so both ends then sit outside the true
    ones and the bracket still holds its root.
    """
    rho_lo, rho_hi = (jnp.asarray(b) for b in rho_bounds(saturation))
    on_dome = T < saturation.T_crit
    saturated = saturation.state_T(jnp.clip(T, saturation.T_min, saturation.T_crit))

    if tvar_other == ThermoVar.P:
        is_liquid = val_other > saturated.P
    else:
        is_liquid = val_other < 0.5 * (
            saturated.L[tvar_other] + saturated.V[tvar_other]
        )

    return (
        pick(on_dome & is_liquid, saturated.L[ThermoVar.D], rho_lo),
        pick(on_dome & jnp.logical_not(is_liquid), saturated.V[ThermoVar.D], rho_hi),
    )


def solve_1d(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    tvar_known: ThermoVar,
    val_known: Array,
    tvar_other: ThermoVar,
    val_other: Array,
    is_inactive: jaxBool,
) -> Tuple[jaxBool, Array, Array]:
    """Single-phase solve when one of (D, T) is given, so one unknown remains.

    `_bracket_for_T` and `_bracket_for_rho` build the interval, `_one_dim_root`
    walks it and carries the derivative.
    """
    solve_for_T = tvar_known == ThermoVar.D
    scale = ThermoVar(tvar_other).spec.scale
    key = tvar_other.internal_key

    def as_rho_T(unknown, known):
        # rho first, T second, always
        return (known, unknown) if solve_for_T else (unknown, known)

    def residual(x, args):
        eos_, known, target = args
        res = (eos_.props_rhoT(*as_rho_T(x, known))[key] - target) / scale
        return pick(jnp.isfinite(res), res, _OUT_OF_DOMAIN)

    inputs_finite = jnp.isfinite(val_other) & jnp.isfinite(val_known)
    route_dummy = is_inactive | jnp.logical_not(inputs_finite)
    fallback = eos.rho_crit_mass if solve_for_T else eos.T_crit
    known_safe = pick(jnp.isfinite(val_known) & (val_known > 0.0), val_known, fallback)

    if solve_for_T:
        lo, hi = _bracket_for_T(eos, saturation, known_safe)
    else:
        lo, hi = _bracket_for_rho(eos, saturation, known_safe, tvar_other, val_other)

    # an inactive lane must still be handed a target its bracket can hold
    dummy_target = jax.lax.stop_gradient(
        eos.props_rhoT(*as_rho_T(0.5 * (lo + hi), known_safe))[key]
    )
    target_safe = pick(route_dummy, dummy_target, val_other)
    residual_args = (
        eos,
        jax.lax.stop_gradient(known_safe),
        jax.lax.stop_gradient(target_safe),
    )

    x_final = _one_dim_root(
        key,
        scale,
        solve_for_T,
        eos,
        known_safe,
        target_safe,
        jax.lax.stop_gradient(lo),
        jax.lax.stop_gradient(hi),
    )
    rho_final, T_final = as_rho_T(x_final, known_safe)
    op_status = (
        inputs_finite
        & _is_a_root(residual(x_final, residual_args), target_safe, scale)
        & node_is_valid(saturation, rho_final, T_final)
        & jnp.logical_not(is_inactive)
    )
    return op_status, jnp.asarray(rho_final), jnp.asarray(T_final)


# -------------------------------------------------------
# /!\ Nested bracketed solve, for the pairs holding P /!\
# -------------------------------------------------------


def _density_at_TP(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    T: Array,
    P: Array,
    seed: Optional[Array] = None,
) -> Array:
    """The density at (T, P), mechanical stability brackets.

    dP/drho >= 0 holds per branch, so the bracket has to name one:
    at (T, P) the state is a liquid exactly when P exceeds P_sat(T).
    A branch decided at T_sat(P) instead is wrong wherever that saturation
    state does not exist.

    `seed` need not be converged, only be on the right branch. With none, the
    branch supplies it: the saturated liquid density for a liquid, which is
    nearly incompressible, and the ideal gas for a vapour, which is exact in the
    dilute limit.
    """
    rho_lo, rho_hi = (jnp.asarray(b) for b in rho_bounds(saturation))
    on_dome = T < saturation.T_crit
    saturated = saturation.state_T(jnp.clip(T, saturation.T_min, saturation.T_crit))
    rho_L, rho_V = saturated.L[ThermoVar.D], saturated.V[ThermoVar.D]

    is_liquid = P > saturated.P
    lo = pick(on_dome & is_liquid, rho_L, rho_lo)
    hi = pick(on_dome & jnp.logical_not(is_liquid), rho_V, rho_hi)

    if seed is None:
        seed = pick(on_dome & is_liquid, rho_L, P / (eos.R_spec * T))

    def residual(rho, args):
        eos_, T_, P_ = args
        props, derivs = eos_.props_rhoT(rho, T_, with_derivatives=True)
        f = (props[ThermoVar.P] - P_) / ThermoVar.P.spec.scale
        return (
            pick(jnp.isfinite(f), f, _OUT_OF_DOMAIN),
            derivs["dP_drho"] / ThermoVar.P.spec.scale,
        )

    # `newton_loop`, whose bracket tightens at every step: a wrong density
    # here flips the sign of the enclosing residual, which a bisection cannot
    # survive, having already discarded the half holding the root
    rho, _ = newton_loop(
        residual,
        lo,
        hi,
        (eos, T, P),
        x0=seed,
        max_steps=TOL.caps.warm_newton_steps,
        rtol=TOL.acc.newton_rtol,
    )
    return jnp.asarray(rho)


def isobar_bounds(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    tvar_other: ThermoVar,
    P: Array,
    y: Array,
) -> Tuple[Array, Array]:
    """The stretch of the isobar P that can hold a single-phase state of y.

    T_sat(P) cuts the isobar in two and the caloric variable says which half:
    below the saturated liquid value a compressed liquid, above the saturated
    vapour value a superheated gas. Supercritically no cut is made.

    An empty interval says no single-phase state on this isobar carries this y,
    which the two-phase branch then answers or nobody does. The endpoint is
    nudged off the line rather than laid on it: there `P > P_sat(T)` is a tie,
    and a root on the line belongs to `solve_two_phase` anyway.
    """
    T_lo, T_hi = (jnp.asarray(b) for b in T_bounds(eos))
    saturated = saturation.state_P(P)
    subcritical = (P < eos.P_crit) & saturated.is_valid
    is_liquid = y < saturated.L[tvar_other]

    off = saturated.T * TOL.acc.sat_edge_nudge
    lo = pick(subcritical & jnp.logical_not(is_liquid), saturated.T + off, T_lo)
    hi = pick(subcritical & is_liquid, saturated.T - off, T_hi)
    return lo, hi


def _isobar_slope(key: str, props, derivs) -> Array:
    """dy/dT at fixed P, read off the EOS call the residual already made.

    - dh/dT|_P = c_p
    - ds/dT|_P = c_p/T from dh = T ds along the isobar
    - du/dT|_P = c_p + (P/rho^2) drho/dT|_P,
      with drho/dT|_P = -dP_dT/dP_drho by the implicit function theorem
    """
    cp = props[ThermoVar.CPMASS]
    if key == ThermoVar.H.internal_key:
        return cp
    if key == ThermoVar.S.internal_key:
        return cp / props[ThermoVar.T]
    if key == ThermoVar.U.internal_key:
        rho = props[ThermoVar.D]
        return cp - (props[ThermoVar.P] / (rho * rho)) * (
            derivs["dP_dT"] / derivs["dP_drho"]
        )
    else:
        raise KeyError(f"Key {key} carries no isobar. ")


def _solve_along_isobar(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    key: str,
    scale: float,
    lo: Array,
    hi: Array,
    P: Array,
    y: Array,
) -> Array:
    """Find T on the isobar P where y(T) = y, carrying the density along.

    A damped Newton, whose step is free: dy/dT|_P is c_p and the same EOS call
    returns it. Each evaluation runs a whole density solve, which is most of
    what the flash spends, so taking a handful of steps rather than a
    bisection's forty-odd is the difference.

    Written here rather than handed to `newton_loop` because it carries state:
    the densities at the previous two temperatures, extrapolated by a secant to
    seed the next help convergence.
    """

    def residual(T, rho_seed):
        rho = _density_at_TP(eos, saturation, T, P, seed=rho_seed)
        props, derivs = eos.props_rhoT(rho, T, with_derivatives=True)
        out = (props[key] - y) / scale
        finite = jnp.isfinite(out)
        return (
            pick(finite, out, _OUT_OF_DOMAIN),
            _isobar_slope(key, props, derivs) / scale,
            rho,
        )

    # the first evaluation has nothing to carry, so it solves its density cold
    f_lo, _, rho_first = residual(lo, None)

    # No early exit on a bracket whose ends share a sign: the ends are where
    # the enclosed density solve is least reliable, so the caller judges the
    # answer by its residual instead.

    def keep_going(walk: _IsobarWalk):
        denom = relative_scale(walk.T)
        settled = walk.moved / denom <= TOL.acc.outer_rtol
        exhausted = (walk.hi - walk.lo) / denom <= TOL.acc.outer_rtol
        return (walk.step < TOL.caps.isobar_steps) & jnp.logical_not(
            settled | exhausted
        )

    def body(walk: _IsobarWalk) -> _IsobarWalk:
        # Secant through the last two temperatures rather than reuse of the
        # last one alone: rho varies smoothly along an isobar, so a first-order
        # extrapolation starts the inner Newton closer.
        # It may point outside the density bracket --> the inner solve already clips it
        span = walk.T_last - walk.T_before
        seed = pick(
            span == 0.0,
            walk.rho_last,
            walk.rho_last
            + (walk.rho_last - walk.rho_before)
            * (walk.T - walk.T_last)
            / pick(span == 0.0, 1.0, span),
        )
        f, dfdT, rho = residual(walk.T, seed)

        # tighten first, then judge the step against the tightened bracket
        same = f * f_lo > 0.0
        lo_new = pick(same, walk.T, walk.lo)
        hi_new = pick(same, walk.hi, walk.T)

        # Always clip to avoid the newton step to bump into the limits of the bisection
        # and lose track of the actual root
        candidate = walk.T - f / pick(dfdT == 0.0, 1.0, dfdT)
        damped = jnp.clip(
            candidate,
            walk.T - 0.5 * (walk.T - lo_new),
            walk.T + 0.5 * (hi_new - walk.T),
        )
        T_new = pick(jnp.isfinite(candidate), damped, 0.5 * (lo_new + hi_new))
        return _IsobarWalk(
            lo=lo_new,
            hi=hi_new,
            T=T_new,
            T_last=walk.T,
            rho_last=rho,
            T_before=walk.T_last,
            rho_before=walk.rho_last,
            step=walk.step + 1,
            moved=jnp.abs(T_new - walk.T),
        )

    walk = jax.lax.while_loop(
        keep_going,
        body,
        _IsobarWalk(
            lo=lo,
            hi=hi,
            T=0.5 * (lo + hi),
            T_last=lo,
            rho_last=rho_first,
            T_before=lo,
            rho_before=rho_first,
            step=jnp.asarray(0),
            moved=jnp.asarray(jnp.inf),
        ),
    )
    return walk.T


# At the solution (rho, T) obeys P(rho, T) = P and
# y(rho, T) = y, so the implicit function theorem
# gives both as one 2x2 solve on the EOS Jacobian,
# there's no need to write painful layer by layer
# differentiation
def _nested_root_primal(
    key: str,
    scale: float,
    eos: HelmholtzEOS,
    saturation: Superancillary,
    lo: Array,
    hi: Array,
    P: Array,
    y: Array,
) -> Array:
    T_sol = _solve_along_isobar(eos, saturation, key, scale, lo, hi, P, y)
    return jnp.stack([_density_at_TP(eos, saturation, T_sol, P), jnp.asarray(T_sol)])


_nested_root = jax.custom_jvp(_nested_root_primal, nondiff_argnums=(0, 1))


@_nested_root.defjvp
def _nested_root_jvp(key, scale, primals, tangents):
    eos, saturation, lo, hi, P, y = primals
    out = _nested_root(key, scale, eos, saturation, lo, hi, P, y)
    rho_sol, T_sol = out[0], out[1]

    def constraints(rho, T):
        props = eos.props_rhoT(rho, T)
        return jnp.stack([props[ThermoVar.P.internal_key], props[key]])

    d_drho, d_dT = jax.jacfwd(constraints, argnums=(0, 1))(rho_sol, T_sol)
    jacobian = jnp.stack([d_drho, d_dT], axis=1)
    targets_dot = jnp.stack([tangents[-2], tangents[-1]])
    return out, jnp.linalg.solve(jacobian, targets_dot)


def solve_nested(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    tvar_other: ThermoVar,
    val_P: Array,
    val_other: Array,
    is_inactive: jaxBool,
) -> Tuple[jaxBool, Array, Array]:
    """(P, X) for X in {H, S, U}, by a bracket on T over a bracket on rho.

    Neither P nor the caloric variable is natural, so no single bracket exists.
    Two stages do: along an isobar T is the only freedom left, and

        dh/dT|_P = c_p > 0        ds/dT|_P = c_p / T > 0

    are thermal stability, proven. (P, U) rides on the same construction with
    du/dT|_P = c_p - P v alpha, which is not proven but was monotone on every
    isobar measured (docs/flash.md).

    T_sat(P) splits the isobar in two, and which half holds the state is read
    off the saturated values at that temperature  below the liquid one it is
    a compressed liquid, above the vapour one a superheated gas, and between
    them the state is two-phase and this is not asked.

    Each outer evaluation costs an inner solve. That is the price of knowing
    which root was found, which a 2D Newton cannot say.
    """
    key = tvar_other.internal_key
    scale = float(tvar_other.spec.scale)

    inputs_finite = jnp.isfinite(val_P) & jnp.isfinite(val_other)
    route_dummy = is_inactive | jnp.logical_not(inputs_finite)
    P_safe = pick(route_dummy, eos.P_crit * 0.5, val_P)
    # an inactive lane needs a target its bracket can hold: the critical point
    # is in domain for every fluid and costs one EOS call
    dummy_target = jax.lax.stop_gradient(
        eos.props_rhoT(eos.rho_crit_mass, eos.T_crit)[key]
    )
    other_safe = pick(route_dummy, dummy_target, val_other)
    lo, hi = isobar_bounds(eos, saturation, tvar_other, P_safe, other_safe)

    def residual(T, args):
        eos_, sat_, P_, target = args
        rho = _density_at_TP(eos_, sat_, T, P_)
        out = (eos_.props_rhoT(rho, T)[key] - target) / scale
        return pick(jnp.isfinite(out), out, _OUT_OF_DOMAIN)

    state = _nested_root(key, scale, eos, saturation, lo, hi, P_safe, other_safe)
    rho_sol, T_sol = state[0], state[1]
    args = (eos, saturation, P_safe, other_safe)

    op_status = (
        _is_a_root(residual(T_sol, args), other_safe, scale)
        & inputs_finite
        & jnp.logical_not(is_inactive)
        & node_is_valid(saturation, rho_sol, T_sol)
    )
    return jnp.asarray(op_status), rho_sol, T_sol
