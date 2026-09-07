from __future__ import annotations

from typing import NamedTuple, Optional

import mpmath as mp
import numpy as np

from .critical import (
    CriticalPoint,
    saturated_pair,
    saturated_pairs,
    solve_critical_point,
)

_THETA_MIN = 1e-10
"""Where the walk starts, i.e. T_crit - T of about 4e-8 K. Not a limit of the
solve, which converges far below it, but of what the float64 arrays it produces
can carry: the dome is 1e-5 wide in relative terms here, leaving eleven digits."""

_N_POINTS = 400
"""Points on the walk, geometric in theta. It is a seed table, so what matters is
that consecutive points be close enough to extrapolate between, and that holds at
a quarter of this count (md 12.5)."""


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

    T_c = float(crit.T_crit)
    rho_red_mass = float(eos.rho_red_mol) * float(eos.molar_mass)

    s_w = np.sqrt(np.clip(1.0 - np.asarray(walk.T) / T_c, 0.0, None))
    order = np.argsort(s_w)  # np.interp wants its abscissa ascending
    s_w = s_w[order]
    dL_w = np.asarray(walk.rho_L)[order] / rho_red_mass
    ln_dV_w = np.log(np.asarray(walk.rho_V)[order] / rho_red_mass)
    s_reach = float(s_w[0])

    cache: dict = {}

    def one(T: float):
        if T in cache:
            return cache[T]
        theta = 1.0 - T / T_c
        if theta <= 0.0:
            pair = (float(crit.delta_c), float(crit.delta_c))
        else:
            s = np.sqrt(theta)
            seed = (
                None
                if s < s_reach
                else (
                    float(np.interp(s, s_w, dL_w)),
                    float(np.exp(np.interp(s, s_w, ln_dV_w))),
                )
            )
            dL, dV, _ = saturated_pair(eos, crit, mp.mpf(theta), seed=seed)
            pair = (float(dL), float(dV))
        out = (pair[0] * rho_red_mass, pair[1] * rho_red_mass)
        cache[T] = out
        return out

    def solve(T: np.ndarray):
        pairs = [one(float(t)) for t in np.atleast_1d(T)]
        return (
            np.array([p[0] for p in pairs]),
            np.array([p[1] for p in pairs]),
        )

    return solve
