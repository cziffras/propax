from __future__ import annotations

from typing import NamedTuple, Optional

import mpmath as mp
import numpy as np

from .critical import (
    CriticalPoint,
    ReducedEOS,
    saturated_pair,
    saturated_pairs,
    solve_critical_point,
)
from .precision import PRECISION

_THETA_MIN = 1e-10
"""Where the walk starts, i.e. T_crit - T of about 4e-8 K. Not a limit of the
solve, which converges far below it, but of what the float64 arrays it produces
can carry: the dome is 1e-5 wide in relative terms here, leaving eleven digits."""

_N_POINTS = 400
"""Points on the walk, geometric in theta. It is a seed table, so what matters is
that consecutive points be close enough to extrapolate between, and that holds at
a quarter of this count."""


class SaturationCurve(NamedTuple):
    """The saturation curve, and the range on which it means anything in T.

    Both ends are fluid's properties and are in fact triple and critical point (equal to
    an extremely high precision):
    `T_min` is the triple point, below which there is no equilibrium at all, and
    `T_max` is the critical point, above which there is none either.
    """

    T: np.ndarray
    rho_L: np.ndarray
    rho_V: np.ndarray
    P: np.ndarray
    T_min: float
    T_max: float


def saturation_walk(
    eos,
    crit: Optional[CriticalPoint] = None,
) -> SaturationCurve:
    """The whole curve, from the critical point down to the triple point."""
    crit = solve_critical_point(eos)
    T_c = float(crit.T_crit)
    theta_max = 1.0 - float(eos.T_triple) / T_c
    thetas = np.geomspace(_THETA_MIN, theta_max, _N_POINTS)

    tau, delta_L, delta_V, psi = saturated_pairs(eos, crit, thetas)

    T_red = float(eos.T_red)
    rho_red_mass = float(eos.rho_red_mol) * float(eos.molar_mass)
    scale = float(eos.rho_red_mol) * float(eos.R_u)

    T = np.array([T_red / float(t) for t in tau])
    rho_L = np.array([float(d) * rho_red_mass for d in delta_L])
    rho_V = np.array([float(d) * rho_red_mass for d in delta_V])
    P = np.array([scale * Ti * float(p) for Ti, p in zip(T, psi)])

    order = np.argsort(T)
    T, rho_L, rho_V, P = T[order], rho_L[order], rho_V[order], P[order]
    return SaturationCurve(T, rho_L, rho_V, P, float(T[0]), T_c)


def make_node_solver(
    eos,
    crit: CriticalPoint,
    walk: SaturationCurve,
):
    """T -> (rho_L, rho_V, P_sat), each node solved once in extended precision.

    Every equilibrium solve is seeded from the walk, interpolated in s where
    delta_L and ln delta_V are nearly linear. Closer to the critical point than
    the walk's first point, the critical expansion seeds it instead.
    """
    T_crit = float(crit.T_crit)
    rho_red_mass = float(eos.rho_red_mol) * float(eos.molar_mass)
    rho_crit_mass = float(crit.delta_c) * rho_red_mass
    at_critical_point = (rho_crit_mass, rho_crit_mass, float(crit.P_crit))

    s_walk = np.sqrt(np.clip(1.0 - walk.T / T_crit, 0.0, None))
    order = np.argsort(s_walk)  # np.interp wants its abscissa ascending
    s_walk = s_walk[order]
    delta_L_walk = walk.rho_L[order] / rho_red_mass
    ln_delta_V_walk = np.log(walk.rho_V[order] / rho_red_mass)

    def seed_at(s: float):
        if s < s_walk[0]:
            return None
        delta_L = np.interp(s, s_walk, delta_L_walk)
        delta_V = np.exp(np.interp(s, s_walk, ln_delta_V_walk))
        return float(delta_L), float(delta_V)

    reduced_by_dps: dict = {}

    def node_at(T: float):
        theta = 1.0 - T / T_crit
        if theta <= 0.0:
            return at_critical_point

        seed = seed_at(np.sqrt(theta))
        delta_L, delta_V, tau = saturated_pair(eos, crit, mp.mpf(theta), seed=seed)

        dps = PRECISION.working_dps(theta)
        if dps not in reduced_by_dps:
            reduced_by_dps[dps] = ReducedEOS(eos, dps)
        r = reduced_by_dps[dps]
        mp.mp.dps = dps
        # read on the vapour side, as the walk does
        P_sat = r.rho_red_mol * r.R_u * (r.T_red / tau) * r.psi(delta_V, tau)

        return (
            float(delta_L) * rho_red_mass,
            float(delta_V) * rho_red_mass,
            float(P_sat),
        )

    solved: dict = {}

    def solve(T: np.ndarray):
        temperatures = [float(t) for t in np.atleast_1d(T)]
        for t in temperatures:
            if t not in solved:
                solved[t] = node_at(t)
        rho_L, rho_V, P_sat = np.array([solved[t] for t in temperatures]).T
        return rho_L, rho_V, P_sat

    return solve
