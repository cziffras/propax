from typing import Dict, FrozenSet, Optional

from ..config import ThermoVar

# The EOS' own variables
NATURAL = frozenset({ThermoVar.D, ThermoVar.T})


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
BRACKETABLE: Dict[ThermoVar, FrozenSet[ThermoVar]] = {
    ThermoVar.D: frozenset({ThermoVar.U, ThermoVar.S, ThermoVar.H}),
    ThermoVar.T: frozenset({ThermoVar.P, ThermoVar.S}),
}


def one_dim_known_var(tvar1: ThermoVar, tvar2: ThermoVar) -> Optional[ThermoVar]:
    """Which of (D, T) is the given variable when the pair reduces to a
    bracketed 1D solve.

    The resulting bracket has to contain a single root, which needs the target
    to be strictly monotone in the unknown. `BRACKETABLE` carries the pairs where
    that is proven: (D,U), (D,S), (T,P) and some others that empirically revealed
    themselves as monotone outside the dome.

    NOTE:
    - This orients the single-phase dispatch and not the two-phase one
    """
    pair = {tvar1, tvar2}
    for known, targets in BRACKETABLE.items():
        if known not in pair:
            continue
        other = tvar2 if tvar1 == known else tvar1
        # (D, T) pins both and is degenerate
        if other in targets:
            return known
    return None


def both_natural_pair(tvar1: ThermoVar, tvar2: ThermoVar) -> bool:
    """Is the pair (D, T)? (natural inputs vars of the EOS)"""
    return frozenset({tvar1, tvar2}) == NATURAL


def is_degenerate_pair(tvar1: ThermoVar, tvar2: ThermoVar) -> bool:
    """Is the pair (P, T), which cannot describe a two-phase state?

    Under the dome P and T are not independent  P = P_sat(T)  so a (P, T)
    input either names a single-phase state or is inconsistent. The two-phase
    graph is then never built.
    """
    return frozenset({tvar1, tvar2}) == frozenset({ThermoVar.P, ThermoVar.T})


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
    """Raise on a pair propax cannot answer, saying why (see above)."""
    reason = INVALID_PAIRS.get(frozenset({tvar1, tvar2}))
    if reason is not None:
        raise ValueError(
            f"propax does not support the pair ({tvar1.value}, {tvar2.value}): "
            f"{reason}."
        )


# Pairs solved by a bracket on T laid over a bracket on rho: P fixes the
# isobar, the caloric variable is monotone along it, and the inner density
# solve is bracketed per branch --> see `single_phase.solve_nested`
NESTED = frozenset(
    {
        frozenset({ThermoVar.P, ThermoVar.H}),
        frozenset({ThermoVar.P, ThermoVar.S}),
        frozenset({ThermoVar.P, ThermoVar.U}),
    }
)


# Pairs a quality can be read with
SATURATED = frozenset(
    {
        frozenset({ThermoVar.P, ThermoVar.Q}),
        frozenset({ThermoVar.T, ThermoVar.Q}),
    }
)


def is_saturated_pair(tvar1: ThermoVar, tvar2: ThermoVar) -> bool:
    """Is this a quality read against the pressure or the temperature?"""
    return frozenset({tvar1, tvar2}) in SATURATED


def is_nested_pair(tvar1: ThermoVar, tvar2: ThermoVar) -> bool:
    """Does this pair take the two-stage bracket?"""
    return frozenset({tvar1, tvar2}) in NESTED


def route_of(tvar1: ThermoVar, tvar2: ThermoVar) -> Optional[str]:
    """How `flash` answers this pair, or None if it declines it.

    Read off the same tables the dispatch branches on, so the two cannot
    disagree about what is supported.
    """
    if frozenset({tvar1, tvar2}) in INVALID_PAIRS:
        return None
    if is_saturated_pair(tvar1, tvar2):
        return "saturated"
    if both_natural_pair(tvar1, tvar2):
        return "natural"
    if is_nested_pair(tvar1, tvar2):
        return "nested"
    if is_degenerate_pair(tvar1, tvar2):
        return "one_dim"
    if one_dim_known_var(tvar1, tvar2) is not None:
        return "one_dim"
    return None


def supported_pairs() -> Dict[FrozenSet[ThermoVar], str]:
    """Every pair the flash accepts, mapped to the route that answers it."""
    inputs = [v for v in ThermoVar if v.spec.can_be_input]
    found = {}
    for i, first in enumerate(inputs):
        for second in inputs[i + 1 :]:
            route = route_of(first, second)
            if route is not None:
                found[frozenset({first, second})] = route
    return found
