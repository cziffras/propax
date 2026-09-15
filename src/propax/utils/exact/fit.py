from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from ...fluids.schema import SaturationSuperancillary
from .constants import density_ceiling
from .critical import solve_critical_point
from .curve import make_node_solver, saturation_walk
from .superancillary import ChebyshevChannel

_LIQUID, _VAPOUR = 0, 1
"""Components of the fitted pair. Only saturated densities are stored, every
other saturated quantity is the EOS read at them, that improves consistency and
code simplicity without costing much in terms of speed."""


@dataclass(frozen=True)
class SaturationFit:
    """The fitted saturation channels, both anchored on T."""

    T_crit: float
    T_min: float
    rho_max_mol: float
    pair: ChebyshevChannel
    """T -> (rho_L, ln rho_V), in s = sqrt(1 - T/T_crit)."""
    pressure: ChebyshevChannel
    """T <-> ln P_sat, in theta = 1 - T/T_crit."""

    def __repr__(self) -> str:
        stored = sum(
            len(c.pieces) * (c.degree + 1) * c.n_components
            for c in (self.pair, self.pressure)
        )
        return (
            f"<SaturationFit T {self.T_min:.3f}..{self.T_crit:.3f} K, "
            f"{stored} coefficients>"
        )

    def to_block(self):

        def _layout(channel: ChebyshevChannel, log_components) -> dict:
            pieces = channel.pieces
            return dict(
                edges=[p.xmin for p in pieces] + [pieces[-1].xmax],
                coeffs=np.stack([np.atleast_2d(p.coeffs.T).T for p in pieces]).tolist(),
                log_components=list(log_components),
            )

        return SaturationSuperancillary(
            T_min=self.T_min,
            rho_max_mol=self.rho_max_mol,
            densities=_layout(self.pair, (_VAPOUR,)),  # type: ignore[arg-type]
            pressure=_layout(self.pressure, (0,)),  # type: ignore[arg-type]
        )

    def save(self, path: Path | str) -> Path:
        """Stored as json (to draw a line between tables produced by the `build` which
        are not necessary and superancillaries that are required).s"""
        path = Path(path)
        fluid = json.loads(path.read_text())
        fluid["eos"]["superancillary"] = self.to_block().model_dump()
        path.write_text(json.dumps(fluid, indent=2) + "\n")
        return path


def _require_x64() -> None:
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "fitting a saturation curve needs JAX in 64-bit precision. Add at "
            "the top of your script:\nimport jax\n"
            "jax.config.update('jax_enable_x64', True)"
        )


def from_eos(eos) -> SaturationFit:
    """Sample the equilibrium and fit it, from the triple point to the critical one.

    Every channel reads the same equilibrium solves, keyed by the temperature
    anchoring them. Both densities share one dyadic tree in s for consistency, ln P_sat has its
    own in theta (more precise when evaluated in theta).
    """
    _require_x64()
    crit = solve_critical_point(eos)
    walk = saturation_walk(eos, crit)
    solve = make_node_solver(eos, crit, walk)

    T_crit, T_min = float(crit.T_crit), walk.T_min
    theta_max = 1.0 - T_min / T_crit
    s_max = float(np.sqrt(theta_max))

    def sample(T) -> np.ndarray:
        T = np.clip(np.atleast_1d(np.asarray(T)), T_min, T_crit)
        return np.stack(solve(T), axis=-1)

    def densities(s):
        rho = sample(T_crit * (1.0 - np.asarray(s) ** 2))[..., :2]
        # rho_V spans decades down to the triple point, see `ChebyshevPieces`
        rho[..., _VAPOUR] = np.log(rho[..., _VAPOUR])
        return rho

    def log_P(theta):
        return np.log(sample(T_crit * (1.0 - np.asarray(theta)))[..., 2])

    pair = ChebyshevChannel(densities, 0.0, s_max)
    pressure = ChebyshevChannel(log_P, 0.0, theta_max)

    rho_L_triple = float(pair(s_max)[_LIQUID])
    rho_max_mol = density_ceiling(eos, rho_L_triple)
    return SaturationFit(T_crit, T_min, rho_max_mol, pair, pressure)
