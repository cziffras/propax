from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from propax.fluids.generic import HelmholtzEOS
from propax.utils.solvers import newton_loop
from propax.utils.types import jaxBool

from ...utils.numerics import pick
from ..config import ThermoVar
from ..domain import T_bounds, node_is_valid, rho_bounds
from ..saturation import Superancillary
from ..tolerances import TOL
from .dispatch import both_natural_pair, is_nested_pair, one_dim_known_var

_OUT_OF_DOMAIN = 1e6


def _is_a_root(scaled_residual: Array, target: Array, scale: float) -> jaxBool:
    reference = jnp.maximum(jnp.abs(target) / scale, 1.0)
    return jnp.abs(scaled_residual) <= TOL.acc.root_atol * reference


##################################################


def _bracket_for_rho(
    saturation: Superancillary,
    T: Array,
    tvar_other: ThermoVar,
    val_other: Array,
) -> Tuple[Array, Array]:
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
    solve_for_T = tvar_known == ThermoVar.D
    scale = tvar_other.spec.scale
    key = tvar_other.internal_key

    def as_rho_T(unknown, known):
        return (known, unknown) if solve_for_T else (unknown, known)

    inputs_finite = jnp.isfinite(val_other) & jnp.isfinite(val_known)
    route_dummy = is_inactive | jnp.logical_not(inputs_finite)
    fallback = eos.rho_crit_mass if solve_for_T else eos.T_crit
    known_safe = pick(jnp.isfinite(val_known) & (val_known > 0.0), val_known, fallback)

    if solve_for_T:
        T_lo, T_hi = (jnp.asarray(b) for b in T_bounds(eos))
        T_sat, on_dome = saturation.T_of_rho(jax.lax.stop_gradient(known_safe))
        lo = pick(on_dome, T_sat * (1.0 - TOL.acc.sat_edge_nudge), T_lo)
        hi = T_hi
    else:
        lo, hi = _bracket_for_rho(saturation, known_safe, tvar_other, val_other)

    dummy_target = jax.lax.stop_gradient(
        eos.props_rhoT(*as_rho_T(0.5 * (lo + hi), known_safe))[key]
    )
    target_safe = pick(route_dummy, dummy_target, val_other)

    def residual(x, eos_, params, carry):
        known_, target_ = params
        rho, T = as_rho_T(x, known_)
        props, derivs = eos_.props_rhoT(rho, T, with_derivatives=True)
        f = (props[key] - target_) / scale

        if not solve_for_T:
            if key == ThermoVar.P.internal_key:
                slope = derivs["dP_drho"]
            else:
                slope = -derivs["dP_dT"] / (rho * rho)
        elif key == ThermoVar.U.internal_key:
            slope = props[ThermoVar.CVMASS]
        elif key == ThermoVar.S.internal_key:
            slope = props[ThermoVar.CVMASS] / T
        elif key == ThermoVar.H.internal_key:
            slope = props[ThermoVar.CVMASS] + derivs["dP_dT"] / rho
        else:
            raise KeyError(f"Key {key} is not 1d bracketable. ")

        return pick(jnp.isfinite(f), f, _OUT_OF_DOMAIN), slope / scale, carry

    x_final, _ = newton_loop(
        (known_safe, target_safe),
        consts=eos,
        carry=None,
        residual=residual,
        lower=lo,
        upper=hi,
        x0=None,
        max_steps=TOL.caps.newton_steps,
        rtol=TOL.acc.newton_rtol,
    )
    rho_final, T_final = as_rho_T(x_final, known_safe)

    props = eos.props_rhoT(rho_final, T_final)
    op_status = (
        inputs_finite
        & _is_a_root((props[key] - target_safe) / scale, target_safe, scale)
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
    seed: Array,
) -> Array:
    """The density at (T, P), mechanical stability brackets.

    dP/drho >= 0 holds per branch, so the bracket has to name one:
    at (T, P) the state is a liquid exactly when P exceeds P_sat(T).
    A branch decided at T_sat(P) instead is wrong wherever that saturation
    state does not exist.
    """
    lo, hi = _bracket_for_rho(saturation, T, ThermoVar.P, P)

    def residual(rho, eos_, params, carry):
        T_, P_ = params
        props, derivs = eos_.props_rhoT(rho, T_, with_derivatives=True)
        f = (props[ThermoVar.P] - P_) / ThermoVar.P.spec.scale
        return (
            pick(jnp.isfinite(f), f, _OUT_OF_DOMAIN),
            derivs["dP_drho"] / ThermoVar.P.spec.scale,
            carry,
        )

    rho, _ = newton_loop(
        (T, P),
        consts=eos,
        carry=None,
        residual=residual,
        lower=lo,
        upper=hi,
        x0=seed,
        max_steps=TOL.caps.warm_newton_steps,
        rtol=TOL.acc.newton_rtol,
    )
    return jnp.asarray(rho)


def solve_nested(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    tvar_other: ThermoVar,
    val_P: Array,
    val_other: Array,
    is_inactive: jaxBool,
) -> Tuple[jaxBool, Array, Array]:
    """
    Nested solves happen for flashes combining P and an energetic variable,
    we call them nested because the outer solve searches for the temperature
    while the inner solve finds the corresponding density for each T-P pair.

    Put differently this method finds T along the isobar where y(T) = y by
    searching for the density at each solve step and querying the EOS in rho, T.
    """

    key = tvar_other.internal_key
    scale = float(tvar_other.spec.scale)

    inputs_finite = jnp.isfinite(val_P) & jnp.isfinite(val_other)
    inactive = is_inactive | jnp.logical_not(inputs_finite)
    y_crit = jax.lax.stop_gradient(eos.props_rhoT(eos.rho_crit_mass, eos.T_crit)[key])
    P = pick(inactive, 0.5 * eos.P_crit, val_P)
    y = pick(inactive, y_crit, val_other)

    sat = saturation.state_P(P)
    T_min, T_max = (jnp.asarray(b) for b in T_bounds(eos))
    has_dome = (P < eos.P_crit) & sat.is_valid
    liquid = has_dome & (y < sat.L[tvar_other])
    vapour = has_dome & jnp.logical_not(liquid)
    lo = pick(vapour, sat.T * (1.0 + TOL.acc.sat_edge_nudge), T_min)
    hi = pick(liquid, sat.T * (1.0 - TOL.acc.sat_edge_nudge), T_max)

    def residual(T, consts, params, rho_prev):
        eos_, sat_ = consts
        P_, y_ = params

        rho = _density_at_TP(eos_, sat_, T, P_, seed=rho_prev)

        props, derivs = eos_.props_rhoT(rho, T, with_derivatives=True)
        f = (props[key] - y_) / scale

        cp = props[ThermoVar.CPMASS]
        if key == ThermoVar.H.internal_key:
            slope = cp
        elif key == ThermoVar.S.internal_key:
            slope = cp / T
        elif key == ThermoVar.U.internal_key:
            slope = cp - (props[ThermoVar.P] / (rho * rho)) * (
                derivs["dP_dT"] / derivs["dP_drho"]
            )
        else:
            raise KeyError(f"Key {key} carries no isobar. ")

        return pick(jnp.isfinite(f), f, _OUT_OF_DOMAIN), slope / scale, rho

    T0 = 0.5 * (lo + hi)
    rho_branch, _ = _bracket_for_rho(saturation, T0, ThermoVar.P, P)
    rho_min, _ = rho_bounds(saturation)
    rho_guess = pick(rho_branch > rho_min, rho_branch, P / (eos.R_spec * T0))

    T_sol, rho_prev = newton_loop(
        (P, y),
        consts=(eos, saturation),
        carry=rho_guess,
        residual=residual,
        lower=lo,
        upper=hi,
        x0=T0,
        max_steps=TOL.caps.isobar_steps,
        rtol=TOL.acc.outer_rtol,
    )
    rho_sol = _density_at_TP(eos, saturation, T_sol, P, seed=rho_prev)

    props = eos.props_rhoT(rho_sol, T_sol)
    P_scale = ThermoVar.P.spec.scale
    op_status = (
        _is_a_root((props[ThermoVar.P] - P) / P_scale, P, P_scale)
        & _is_a_root((props[key] - y) / scale, y, scale)
        & inputs_finite
        & jnp.logical_not(is_inactive)
        & node_is_valid(saturation, rho_sol, T_sol)
    )
    return jnp.asarray(op_status), rho_sol, T_sol


def solve_single_phase(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    tvar1: ThermoVar,
    val1: Array,
    tvar2: ThermoVar,
    val2: Array,
    is_inactive: jaxBool,
) -> Tuple[jaxBool, Array, Array]:
    if both_natural_pair(tvar1, tvar2):
        rho, T = (val1, val2) if tvar1 == ThermoVar.D else (val2, val1)
        ok = (
            jnp.isfinite(rho)
            & jnp.isfinite(T)
            & (rho > 0.0)
            & jnp.logical_not(is_inactive)
        )
        return ok, rho, T

    if is_nested_pair(tvar1, tvar2):
        val_P, tvar_other, val_other = (
            (val1, tvar2, val2) if tvar1 == ThermoVar.P else (val2, tvar1, val1)
        )
        return solve_nested(eos, saturation, tvar_other, val_P, val_other, is_inactive)

    known = one_dim_known_var(tvar1, tvar2)
    if known is None:
        raise ValueError(f"({tvar1.value}, {tvar2.value}) has no single-phase solver")
    val_known, tvar_other, val_other = (
        (val1, tvar2, val2) if tvar1 == known else (val2, tvar1, val1)
    )
    return solve_1d(
        eos, saturation, known, val_known, tvar_other, val_other, is_inactive
    )
