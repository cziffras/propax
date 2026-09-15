from typing import Tuple

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import Array

from propax.fluids.generic import HelmholtzEOS
from propax.utils.types import jaxBool

from ...utils.numerics import pick
from ..config import ThermoVar
from ..domain import clamp_quality, is_two_phase
from ..saturation import Superancillary
from ..tolerances import TOL
from .results import as_mixed, get_sat_bounds


def _saturation_at(eos: HelmholtzEOS, saturation: Superancillary, anchor, value):
    if anchor == ThermoVar.T:
        in_range = (value >= eos.T_triple) & (value < eos.T_crit)
        return saturation.state_T(value), in_range
    in_range = (value >= eos.P_triple) & (value < eos.P_crit)
    return saturation.state_P(value), in_range


def _split(anchor: ThermoVar, tvar1, val1, tvar2, val2):
    """(the anchor's value, the other variable, its value)."""
    return (val1, tvar2, val2) if tvar1 == anchor else (val2, tvar1, val1)


def solve_saturated(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    tvar1: ThermoVar,
    val1: Array,
    tvar2: ThermoVar,
    val2: Array,
) -> Tuple[jaxBool, Array, Array]:
    """The state read from a quality and either the pressure or the temperature.

    Returns (is_valid, T_sat, quality). The quality is clipped into [0, 1]
    because the lever rule may not see anything else, but a quality outside it
    was never a mixture and the flag says so.
    """
    anchor = ThermoVar.T if ThermoVar.T in (tvar1, tvar2) else ThermoVar.P
    known, _, x = _split(anchor, tvar1, val1, tvar2, val2)
    sat, in_range = _saturation_at(eos, saturation, anchor, known)
    is_valid = (
        in_range & sat.is_valid & jnp.isfinite(x) & is_two_phase(x, jnp.asarray(True))
    )
    return is_valid, sat.T, clamp_quality(x)


def solve_two_phase(
    eos: HelmholtzEOS,
    saturation: Superancillary,
    tvar1: ThermoVar,
    val1: Array,
    tvar2: ThermoVar,
    val2: Array,
    is_inactive: jaxBool,
    slack: float = 0.0,
) -> Tuple[jaxBool, Array, Array]:
    """
    Check whether the given state lies in the two-phase region, and return
    (is_biphasic, T_sat, quality).

    `slack` widens the quality window the flag accepts, for a caller that wants
    a ring of states around the dome rather than the dome alone, that is useful
    for table building, a flash never does need it.

    Three cases, on which the body dispatches: P given (T_sat read from the
    saturation curve), T given (T_sat is the input), or neither; then T_sat is
    found by a Newton solve on the temperature at which the quality induced by
    one input reconstructs the other.

    Why this cannot be a flash on the EOS (simply reverting the EOS)?
    a(rho, T) = u - Ts is the free energy of a homogeneous fluid: rho presupposes
    matter uniformly spread. Inside the dome the equilibrium is not homogeneous
    but consists in two, regions at rho_L and rho_V; so no (rho, T) on the EOS
    surface reproduces it, and the surface cannot be inverted for it.
    The equilibrium sits on the common tangent below the surface of slope
    -P_sat (since P = -da /dv), not on the surface itself.

    What is solved instead is the position along that tangent, meaning the vapour
    quality q, the two tangency points being fixed by the saturation solver
    through equal pressure and equal fugacity (chemical equilibrium, since
    ln f_L - ln f_V = (g_L - g_V)/RT).
    """
    vars_set = {tvar1, tvar2}

    # (P, T) pins the saturation state itself: no mixture can be named by it
    if vars_set == {ThermoVar.P, ThermoVar.T}:
        T_guess = jnp.asarray((eos.T_triple + eos.T_crit) / 2.0)
        return jnp.array(False), T_guess, jnp.array(0.5)

    # P or T given: T_sat comes off the curve, the other variable places x on it
    anchor = next((v for v in (ThermoVar.P, ThermoVar.T) if v in vars_set), None)
    if anchor is not None:
        known, other_tvar, other_val = _split(anchor, tvar1, val1, tvar2, val2)
        sat, in_range = _saturation_at(eos, saturation, anchor, known)
        L, V = get_sat_bounds(sat, other_tvar)
        x = (as_mixed(other_tvar, other_val) - L) / (V - L)
        # a saturation state the module flags as invalid must not silently
        # become a converged two-phase answer
        is_valid_flag = in_range & is_two_phase(
            x, sat.is_valid, slack=max(slack, TOL.acc.quality_slack)
        )
        return is_valid_flag, sat.T, clamp_quality(x, slack)

    # Neither: the residual is the variable the lever rule is linear in, rebuilt
    # from the quality the other one induces. Taken the other way round the
    # Jacobian is ill-conditioned, (U, D) = (-2e4, 2.0) gave NaNs up to c796380
    if tvar1.spec.linear_in_x:
        linear, other, linear_val, other_val = tvar1, tvar2, val1, val2
    else:
        linear, other, linear_val, other_val = tvar2, tvar1, val2, val1

    linear_norm = as_mixed(linear, linear_val)
    other_norm = as_mixed(other, other_val)
    s_linear = jnp.asarray(linear.spec.scale, dtype=other_norm.dtype)

    def residual(T, args):
        linear_norm, other_norm, s_linear = args
        sat = saturation.state_T(jnp.clip(T, saturation.T_min, saturation.T_crit))
        L_other, V_other = get_sat_bounds(sat, other)
        denom = pick(
            jnp.abs(V_other - L_other) < TOL.acc.lever_denom_atol,
            1.0,
            V_other - L_other,
        )
        x_induced = (other_norm - L_other) / denom
        L_lin, V_lin = get_sat_bounds(sat, linear)
        res = (L_lin + x_induced * (V_lin - L_lin) - linear_norm) / s_linear
        # A non-finite residual anywhere in a vmapped batch aborts the
        # whole batch inside lineax; return a large finite value instead
        # so only this lane fails to converge.
        return pick(jnp.isfinite(res), res, 1e6)

    # Neither P nor T pins T_sat, so this Newton is the only thing standing
    # between the caller and a wrong phase, and it needs a high quality seed:
    # the residual is walked along the curve, uniformly in s = sqrt(1 - T/T_crit)
    # since the dome closes like sqrt(theta), and a uniform scan in T would spend
    # its last cell on the whole critical region and step over the crossing there
    T_dummy = (saturation.T_min + saturation.T_crit) / 2.0
    T_guess, crossed = _provide_guess_and_flag_for_third_case(
        residual, saturation, other, T_dummy, linear_norm, other_norm, s_linear
    )

    # Inactive lanes, non-finite inputs (1/0 from `as_mixed`) and lanes the scan
    # found no crossing for are handed a trivial root at T_dummy: the Newton stops
    # at once and the flag below turns them down
    sat_dummy = saturation.state_T(T_dummy)
    L_lin_dummy, V_lin_dummy = get_sat_bounds(sat_dummy, linear)
    L_oth_dummy, V_oth_dummy = get_sat_bounds(sat_dummy, other)
    inputs_finite = jnp.isfinite(linear_norm) & jnp.isfinite(other_norm)
    route_dummy = is_inactive | jnp.logical_not(inputs_finite & crossed)
    args = (
        pick(route_dummy, 0.5 * (L_lin_dummy + V_lin_dummy), linear_norm),
        pick(route_dummy, 0.5 * (L_oth_dummy + V_oth_dummy), other_norm),
        s_linear,
    )

    # well_posed=False (least-squares) for both the forward Newton and its
    # implicit adjoint: it tolerates a singular Jacobian so a
    # lane fails to converge instead of raising, and so the reverse-mode
    # adjoint solve stays finite
    linear_solver = lx.AutoLinearSolver(well_posed=False)
    solver = optx.Newton(**TOL.acc.equilibrium, linear_solver=linear_solver)
    sol = optx.root_find(
        fn=residual,
        solver=solver,
        y0=T_guess,
        args=args,
        max_steps=TOL.caps.equilibrium_steps,
        throw=False,
        adjoint=optx.ImplicitAdjoint(linear_solver=linear_solver),
    )
    success = sol.result == optx.RESULTS.successful
    T_final = pick(success, sol.value, T_guess)

    sat_final = saturation.state_T(T_final)
    L_other_f, V_other_f = get_sat_bounds(sat_final, other)
    # degenerate when coming to the critical point
    well_posed = jnp.abs(V_other_f - L_other_f) >= TOL.acc.lever_denom_atol
    x_final = (other_norm - L_other_f) / pick(well_posed, V_other_f - L_other_f, 1.0)

    # `linear` is then what has to come back, ignore the solver's flag
    L_lin_f, V_lin_f = get_sat_bounds(sat_final, linear)
    lin_rebuilt = L_lin_f + x_final * (V_lin_f - L_lin_f)
    reproduces_linear = (
        jnp.abs(lin_rebuilt - linear_norm) / s_linear <= TOL.acc.root_atol
    )

    in_solver_range = (T_final >= eos.T_triple) & (T_final < saturation.T_crit)
    is_biphasic_flag = (
        crossed
        & success
        & inputs_finite
        & in_solver_range
        & well_posed
        & reproduces_linear
        & is_two_phase(x_final, jnp.asarray(True), slack=slack)
    )
    return is_biphasic_flag, T_final, clamp_quality(x_final, slack)


def _provide_guess_and_flag_for_third_case(
    residual, saturation, other_meta, T_dummy, linear_norm, other_norm, s_linear
):

    T_min_scan = jnp.asarray(saturation.T_min)
    # s is a decreasing function of T
    s_scan = jnp.linspace(0.0, saturation.s_of_T(T_min_scan), TOL.caps.n_seed_scan)
    T_scan = saturation.T_of_s(s_scan)

    scan_args = (linear_norm, other_norm, s_linear)
    r_scan = jnp.asarray(jax.vmap(lambda t: residual(t, scan_args))(T_scan))
    r_scan = pick(jnp.isfinite(r_scan), r_scan, jnp.inf)

    # where the branches meet the residual has no lever rule and an arbitrary
    # sign, so a crossing touching such a point is taken only if no other exists
    L_other, V_other = get_sat_bounds(jax.vmap(saturation.state_T)(T_scan), other_meta)
    guarded = jnp.abs(V_other - L_other) < TOL.acc.lever_denom_atol

    crosses = (r_scan[:-1] * r_scan[1:]) <= 0.0
    clean = crosses & jnp.logical_not(guarded[:-1] | guarded[1:])
    k = pick(clean.any(), jnp.argmax(clean), jnp.argmax(crosses))

    r_k = r_scan[k]
    r_kp1 = r_scan[k + 1]
    s_k = s_scan[k]
    s_kp1 = s_scan[k + 1]

    # using a linear interpolation allows to greatly improve the guess
    # interpolates on s for reliability close to the critical point
    s_interp = pick(
        r_k < r_kp1,
        jnp.interp(0.0, jnp.array([r_k, r_kp1]), jnp.array([s_k, s_kp1])),
        jnp.interp(0.0, jnp.array([r_kp1, r_k]), jnp.array([s_kp1, s_k])),
    )
    T_interp = saturation.T_of_s(s_interp)

    T_guess = pick(crosses.any(), T_interp, T_dummy)
    T_guess = pick(jnp.isfinite(T_guess), T_guess, T_dummy)

    return jax.lax.stop_gradient(T_guess), crosses.any()
