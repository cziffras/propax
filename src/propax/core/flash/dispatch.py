from typing import Dict, FrozenSet, Optional

from ..config import ThermoVar

# For more information, check `docs/flash.md`` :
# Pairs whose flash reduces to a bracketed 1D solve, keyed by the natural
# variable the pair pins down, some inequalities are proven :
#
#   (D, U) : du/dT|_rho = c_v     > 0     thermal stability
#   (D, S) : ds/dT|_rho = c_v / T > 0     thermal stability, T > 0
#   (T, P) : dP/drho|_T          >= 0     mechanical stability, per branch
#
# Every other pair reduces to the sign of the thermal expansion coefficient
# (whose sign is unknown around the critical point):
# dP/dT|_rho = alpha / kappa_T governs (D,P) directly, (D,H) through
# dh/dT|_rho = c_v + v alpha / kappa_T, and (T,S) through the Maxwell relation
# ds/drho|_T = -rho^-2 alpha / kappa_T
#
# /!\ Shared so the runtime dispatch and the table build cannot drift apart

# Pairs solved by a bracket on T laid over a bracket on rho: P fixes the
# isobar, the caloric variable is monotone along it, and the inner density
# solve is bracketed per branch --> see `single_phase.solve_nested`

ROUTES: Dict[FrozenSet[ThermoVar], str] = {
    frozenset({ThermoVar.D, ThermoVar.T}): "natural",
    frozenset({ThermoVar.P, ThermoVar.Q}): "saturated",
    frozenset({ThermoVar.T, ThermoVar.Q}): "saturated",
    frozenset({ThermoVar.D, ThermoVar.U}): "one_dim",
    frozenset({ThermoVar.D, ThermoVar.H}): "one_dim",
    frozenset({ThermoVar.D, ThermoVar.S}): "one_dim",
    frozenset({ThermoVar.T, ThermoVar.P}): "one_dim",
    frozenset({ThermoVar.T, ThermoVar.S}): "one_dim",
    frozenset({ThermoVar.P, ThermoVar.H}): "nested",
    frozenset({ThermoVar.P, ThermoVar.S}): "nested",
    frozenset({ThermoVar.P, ThermoVar.U}): "nested",
}


# Pairs propax declines, we also specify why they are declined (for now)
INVALID_PAIRS = {
    **{
        frozenset({ThermoVar.Q, other}): (
            "Quality if included in [0, 1] helps to describe a state in the"
            "dome and at fixed T or P allows to retrieve by the lever rule any"
            "extensive variable, however, without T or P given (Q, P) -> T has"
            "several solutions and thus is ill-defined"
        )
        for other in (ThermoVar.D, ThermoVar.H, ThermoVar.S, ThermoVar.U)
    },
    frozenset({ThermoVar.T, ThermoVar.H}): (
        "h is not injective in rho at fixed T: outside the dome many segments "
        "are non-monotone, so several densities share one enthalpy"
    ),
    frozenset({ThermoVar.T, ThermoVar.U}): (
        "u is not injective in rho at fixed T: outside the dome many segments "
        "are non-monotone, so several densities share one energy"
    ),
    frozenset({ThermoVar.H, ThermoVar.S}): (
        "two caloric variables: the 2x2 Jacobian is ill-conditioned"
    ),
    frozenset({ThermoVar.H, ThermoVar.U}): (
        "h = u + P/rho, so the two are nearly collinear wherever P/rho is "
        "small; CoolProp declines this pair as well"
    ),
    frozenset({ThermoVar.S, ThermoVar.U}): (
        "two caloric variables, as (H, S); CoolProp declines it too"
    ),
    frozenset({ThermoVar.P, ThermoVar.D}): (
        "no bracket exists in either ordering: drho/dT|_P = -rho alpha and "
        "dP/dT|_rho = alpha / kappa_T both carry the sign of the thermal "
        "expansion coefficient, which is unconstrained"
    ),
}


def check_supported(tvar1: ThermoVar, tvar2: ThermoVar) -> None:
    reason = INVALID_PAIRS.get(frozenset({tvar1, tvar2}))
    if reason is not None:
        raise ValueError(
            f"propax does not support the pair ({tvar1.value}, {tvar2.value}): "
            f"{reason}."
        )


def route_of(tvar1: ThermoVar, tvar2: ThermoVar) -> Optional[str]:
    return ROUTES.get(frozenset({tvar1, tvar2}))


def supported_pairs() -> Dict[FrozenSet[ThermoVar], str]:
    return dict(ROUTES)
