from typing import Tuple

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import Array

from propax.fluids.generic import HelmholtzEOS
from propax.utils.types import jaxBool

from ..config import ThermoVar
from ..domain import clamp_quality, is_two_phase
from ..saturation import Superancillary
from ..tolerances import TOL
from .results import get_sat_bounds


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
    is_T = tvar1 == ThermoVar.T or tvar2 == ThermoVar.T
    known = val1 if tvar1 in (ThermoVar.P, ThermoVar.T) else val2
    x = val2 if tvar1 in (ThermoVar.P, ThermoVar.T) else val1

    sat = saturation.state_T(known) if is_T else saturation.state_P(known)
    # closed at the triple point, where the curve starts and the superancillary
    # answers; open at the critical one, where the lever rule loses its denominator
    in_range = (
        (known >= eos.T_triple) & (known < eos.T_crit)
        if is_T
        else (known >= eos.P_triple) & (known < eos.P_crit)
    )
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

    # Case 1: P is given
    if ThermoVar.P in vars_set and ThermoVar.T in vars_set:
        # When T and P are specified no biphasic state can exist
        T_guess = jnp.asarray((eos.T_triple + eos.T_crit) / 2.0)
        x_final = jnp.array(0.5)
        is_valid_flag = jnp.array(False)
        return is_valid_flag, T_guess, x_final

    if ThermoVar.P in vars_set:
        P_val = val1 if tvar1 == ThermoVar.P else val2
        other_val = val2 if tvar1 == ThermoVar.P else val1
        other_tvar = tvar2 if tvar1 == ThermoVar.P else tvar1

        P_valid = (P_val >= eos.P_triple) & (P_val < eos.P_crit)
        sat_dict = saturation.state_P(P_val)
        L, V = get_sat_bounds(sat_dict, other_tvar)

        other_norm = jnp.where(
            other_tvar.spec.invert_for_mixing, 1.0 / other_val, other_val
        )
        x_final = (other_norm - L) / (V - L)

        # `sat_dict.is_valid` carries whether the P -> T_sat inversion
        # actually converged (do not return L, V props when they do not exist)
        is_valid_flag = P_valid & is_two_phase(
            x_final, sat_dict.is_valid, slack=TOL.acc.quality_slack
        )
        x_final = clamp_quality(x_final, slack)
        return (
            is_valid_flag,
            sat_dict.T,
            x_final,
        )

    # Case 2: T is given
    if ThermoVar.T in vars_set:
        T_val = val1 if tvar1 == ThermoVar.T else val2
        other_val = val2 if tvar1 == ThermoVar.T else val1
        other_tvar = tvar2 if tvar1 == ThermoVar.T else tvar1

        T_valid = (T_val >= eos.T_triple) & (T_val < eos.T_crit)
        sat_dict = saturation.state_T(T_val)
        L, V = get_sat_bounds(sat_dict, other_tvar)

        other_norm = jnp.where(
            other_tvar.spec.invert_for_mixing, 1.0 / other_val, other_val
        )
        x_final = (other_norm - L) / (V - L)

        # as in the P branch: a saturation state the module itself flags as
        # invalid must not silently become a converged two-phase answer
        is_valid_flag = T_valid & is_two_phase(
            x_final, sat_dict.is_valid, slack=TOL.acc.quality_slack
        )
        x_final = clamp_quality(x_final, slack)
        return (
            is_valid_flag,
            T_val,
            x_final,
        )

    # Case 3: Neither P nor T given - solve for T
    # This assignation can be intriguing, see comment below (about line ~170 of the script)
    if tvar1.spec.linear_in_x:
        linear_meta, other_meta = tvar1, tvar2
        linear_val, other_val = val1, val2
        linear_name = tvar1
    else:
        linear_meta, other_meta = (
            tvar2,
            tvar1,
        )  # hoping for at least one to be linear
        linear_val, other_val = val2, val1
        linear_name = tvar2
        # linear might not be linear in fact, so naming here is rather done to reflect the author's intent
        # never met a case where calling solve pair with a pair of two non linear variables

    linear_norm = jnp.where(
        linear_meta.spec.invert_for_mixing, 1.0 / linear_val, linear_val
    )
    other_norm = jnp.where(
        other_meta.spec.invert_for_mixing, 1.0 / other_val, other_val
    )
    s_linear = jnp.asarray(ThermoVar(linear_name).spec.scale, dtype=other_norm.dtype)

    def residual(T, args):
        linear_norm, other_norm, s_linear = args
        T_clamped = jnp.clip(
            T,
            saturation.T_min,
            saturation.T_crit,
        )
        sat_dict = saturation.state_T(T_clamped)

        L_other, V_other = get_sat_bounds(sat_dict, other_meta)
        denom = jnp.where(
            jnp.abs(V_other - L_other) < TOL.acc.lever_denom_atol,
            1.0,
            V_other - L_other,
        )
        x_induced = (other_norm - L_other) / denom

        # Why so much complexity for passing linear/possibly non linear variables ?
        # Let's say that, in case you pass lin_var = U and other_var = D (D is not linear for the quality, meaning
        # x = (v - v_liq) / (v_gas - v_liq) with v the specific volume, the inverse of density)
        # If we did differently for solving our problem, the jacobian obtained would be poorly conditionned
        # and thus yield NaNs, that is precisely what was done up to commit c796380, we therefore had the
        # subsequent problem when passing U=-2e4 and D=2.0:
        #  - Passing (U, D) resulted in NaNs
        #  - Passing (D, U) perfectly worked
        L_lin, V_lin = get_sat_bounds(
            sat_dict, linear_meta
        )  # linear must be reconstructed, other is used to compute x_induced
        lin_reconstructed = L_lin + x_induced * (V_lin - L_lin)

        res = (lin_reconstructed - linear_norm) / s_linear
        # A non-finite residual anywhere in a vmapped batch aborts the
        # whole batch inside lineax; return a large finite value instead
        # so only this lane fails to converge.
        return jnp.where(jnp.isfinite(res), res, 1e6)

    # Reaching case 3 means neither P nor T pins T_sat, so this Newton is
    # the only thing standing between the caller and a wrong phase
    # Neither P nor T pins T_sat, so this Newton is the only thing standing
    # between the caller and a wrong phase, and it needs a high quality seed !
    # Fortunately superancillaries provide fast and extremely accurate values
    # at saturation, we simply walks it (through the residual) and find a coarse
    # guess for the temperature
    # However : in s = sqrt(1 - T/T_crit), not in T: the dome closes like sqrt(theta), so
    # a scan uniform in T spends its last cell on the whole critical region and
    # steps over the crossing there --> uniform in s it resolves it
    T_lo_scan = jnp.asarray(saturation.T_min)
    T_hi_scan = jnp.asarray(saturation.T_crit)
    T_scan = saturation.T_of_s(
        jnp.linspace(0.0, saturation.s_of_T(T_lo_scan), TOL.caps.n_seed_scan)
    )
    scan_args = (linear_norm, other_norm, s_linear)
    r_scan = jnp.asarray(jax.vmap(lambda t: residual(t, scan_args))(T_scan))
    r_scan = jnp.where(jnp.isfinite(r_scan), r_scan, jnp.inf)

    crosses = (r_scan[:-1] * r_scan[1:]) <= 0.0
    k = jnp.argmax(crosses)
    # no sign change anywhere: the state is not two-phase, and the Newton
    # below will fail to converge, which `is_valid_flag` then reports
    T_guess = jnp.where(
        crosses.any(), 0.5 * (T_scan[k] + T_scan[k + 1]), 0.5 * (T_lo_scan + T_hi_scan)
    )
    T_guess = jnp.where(jnp.isfinite(T_guess), T_guess, 0.5 * (T_lo_scan + T_hi_scan))

    # filler for non-finite lanes; the spline answers, no equilibrium
    # solve is needed to fabricate values that are thrown away
    sat_dummy = saturation.state_T((eos.T_triple + eos.T_crit) / 2.0)
    L_lin_dummy, V_lin_dummy = get_sat_bounds(sat_dummy, linear_meta)
    L_oth_dummy, V_oth_dummy = get_sat_bounds(sat_dummy, other_meta)

    dummy_linear_norm = L_lin_dummy + 0.5 * (V_lin_dummy - L_lin_dummy)
    dummy_other_norm = L_oth_dummy + 0.5 * (V_oth_dummy - L_oth_dummy)

    # Non-finite inputs (eg for 1/0 from invert_for_mixing) are routed to the
    # dummy values so the solver never sees them the lane is then flagged
    # invalid
    inputs_finite = jnp.isfinite(linear_norm) & jnp.isfinite(other_norm)
    route_dummy = is_inactive | jnp.logical_not(inputs_finite)
    safe_linear_norm = jnp.where(route_dummy, dummy_linear_norm, linear_norm)
    safe_other_norm = jnp.where(route_dummy, dummy_other_norm, other_norm)

    args = (safe_linear_norm, safe_other_norm, s_linear)
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
    # lax.cond, not jnp.where: only the taken branch is differentiated, so a
    # non-converged solve's NaN gradient never contaminates the reverse pass
    T_final = jax.lax.cond(
        success, lambda: jnp.asarray(sol.value), lambda: jnp.asarray(T_guess)
    )

    sat_final = saturation.state_T(T_final)

    L_other_f, V_other_f = get_sat_bounds(sat_final, other_meta)

    # check degeneracy (when coming to the critical point)
    well_posed = jnp.abs(V_other_f - L_other_f) >= TOL.acc.lever_denom_atol
    denom_f = jnp.where(well_posed, V_other_f - L_other_f, 1.0)
    x_final = (other_norm - L_other_f) / denom_f

    # `linear` is then what has to come back, which is the residual itself
    # rather than anything the solver's own flag can promise
    L_lin_f, V_lin_f = get_sat_bounds(sat_final, linear_meta)
    lin_rebuilt = L_lin_f + x_final * (V_lin_f - L_lin_f)
    reproduces_linear = (
        jnp.abs(lin_rebuilt - linear_norm) / s_linear <= TOL.acc.root_atol
    )

    in_solver_range = (T_final >= eos.T_triple) & (T_final < saturation.T_crit)
    is_biphasic_flag = (
        success
        & inputs_finite
        & in_solver_range
        & well_posed
        & reproduces_linear
        & is_two_phase(x_final, jnp.asarray(True), slack=slack)
    )

    return (
        is_biphasic_flag,
        T_final,
        clamp_quality(x_final, slack),
    )
