from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ...fluids.schema import SaturationSuperancillary
from .constants import density_ceiling
from .critical import solve_critical_point
from .curve import make_node_solver, saturation_walk
from .superancillary import ChebyshevChannel

FITTED = ("rho_L", "rho_V")
"""Only saturated densities are stored as channels, other saturated
quantities are rederived using the EOS.
"""

_INDEX = {name: i for i, name in enumerate(FITTED)}
"""Which component of the fitted pair each stored channel is."""

DERIVED = ("P_sat", "h_L", "h_V", "s_L", "s_V", "u_L", "u_V")

CHANNELS = FITTED + DERIVED

_BRANCH = {
    "P_sat": "rho_V",  # the vapour side used to retrieve P_sat
    "h_L": "rho_L",
    "h_V": "rho_V",
    "s_L": "rho_L",
    "s_V": "rho_V",
    "u_L": "rho_L",
    "u_V": "rho_V",
}
_QUANTITY = {
    "P_sat": "P",
    "h_L": "h",
    "h_V": "h",
    "s_L": "s",
    "s_V": "s",
    "u_L": "u",
    "u_V": "u",
}


@dataclass(frozen=True)
class SaturationFit:
    """Just a container for saturation fitted chebyshev channels."""

    T_crit: float
    T_min: float
    T_max: float
    rho_max_mol: float
    pair: ChebyshevChannel
    cuts: dict
    log_components: tuple = ()
    """Components of `pair` fitted on their logarithm; see `ChebyshevPieces`."""

    @property
    def s_min(self) -> float:
        return float(np.sqrt(max(1.0 - self.T_max / self.T_crit, 0.0)))

    @property
    def s_max(self) -> float:
        return float(np.sqrt(1.0 - self.T_min / self.T_crit))

    def __repr__(self) -> str:
        p = self.pair
        stored = len(p.pieces) * (p.degree + 1) * p.n_components
        return (
            f"<SaturationFit T {self.T_min:.3f}..{self.T_crit:.3f} K, "
            f"{stored} coefficients, {len(DERIVED)} channels derived>"
        )

    def to_block(self):
        """Layout in storage for a superancillary."""
        pieces = self.pair.pieces
        coeffs = np.stack([np.atleast_2d(np.asarray(p.coeffs).T).T for p in pieces])
        cuts = [_boundaries(g) for g in self.pair.segments]
        width = max(len(c) for c in cuts)
        return SaturationSuperancillary(
            T_min=self.T_min,
            T_max=self.T_max,
            rho_max_mol=self.rho_max_mol,
            edges=[p.xmin for p in pieces] + [pieces[-1].xmax],
            coeffs=coeffs.tolist(),
            # padded such that two components turning over a different
            # number of times still have to share one array
            cuts=[c + [c[-1]] * (width - len(c)) for c in cuts],
            derived_cuts={n: _boundaries(g) for n, g in self.cuts.items()},
            log_components=list(self.log_components),
        )

    def save(self, path: Path | str) -> Path:
        """Stored as json (to draw a line between tables produced by the `build` which
        are not necessary and superancillaries that are required).s"""
        path = Path(path)
        fluid = json.loads(path.read_text())
        fluid["eos"]["superancillary"] = self.to_block().model_dump()
        path.write_text(json.dumps(fluid, indent=2) + "\n")
        return path


def _boundaries(segments) -> list:
    return [segments[0].xmin] + [s.xmax for s in segments]


def fit_saturation(
    eos,
    T_crit: float,
    T_min: float,
    T_max: float,
    sample: Callable[[np.ndarray], np.ndarray],
    props: Callable[[np.ndarray, np.ndarray], dict],
    degree: Optional[int] = None,
    log_components: tuple = (),
) -> SaturationFit:
    """Sample the equilibrium and fit it. The slow half, and the only half here.

    `sample` returns the saturated pair as one (n, 2) block: both densities come
    from one solve per node and share one dyadic tree, so neither is sampled
    twice nor split differently from the other.
    """
    T_crit, T_min, T_max = float(T_crit), float(T_min), float(T_max)
    s_min = float(np.sqrt(max(1.0 - T_max / T_crit, 0.0)))
    s_max = float(np.sqrt(1.0 - T_min / T_crit))

    log_components = tuple(int(c) for c in log_components)

    def fitted(s_nodes):
        y = np.asarray(sample(s_nodes), dtype=float)
        for c in log_components:
            y[..., c] = np.log(y[..., c])
        return y

    pair = ChebyshevChannel(fitted, s_min, s_max, degree)

    def unlog(y):
        y = np.array(y, dtype=float, copy=True)
        for c in log_components:
            y[..., c] = np.exp(y[..., c])
        return y

    def derived_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
        branch, quantity = _BRANCH[name], _QUANTITY[name]

        def f(s):
            s_arr = np.atleast_1d(np.asarray(s, dtype=float))
            T = T_crit * (1.0 - s_arr**2)
            rho = np.atleast_1d(unlog(np.asarray(pair(s_arr)))[..., _INDEX[branch]])
            return np.asarray(props(rho, T)[quantity], dtype=float)

        return f

    cuts = {
        name: ChebyshevChannel(derived_fn(name), s_min, s_max, pair.degree).segments[0]
        for name in DERIVED
    }
    rho_L_triple = float(
        unlog(np.asarray(pair(np.asarray([s_max]))))[0, _INDEX["rho_L"]]
    )
    rho_max_mol = density_ceiling(eos, rho_L_triple)
    return SaturationFit(T_crit, T_min, T_max, rho_max_mol, pair, cuts, log_components)


def _require_x64() -> None:
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "fitting a saturation curve needs JAX in 64-bit precision. Add at "
            "the top of your script:\nimport jax\n"
            "jax.config.update('jax_enable_x64', True)"
        )


def from_eos(
    eos,
    degree: Optional[int] = None,
    T_max: Optional[float] = None,
) -> SaturationFit:
    _require_x64()
    props_eos = jax.jit(jax.vmap(eos.props_rhoT))

    crit = solve_critical_point(eos)
    curve = saturation_walk(eos, crit)
    solve = make_node_solver(eos, crit, curve)

    T_min = curve.T_min
    T_crit = float(crit.T_crit)
    T_top = float(T_max) if T_max is not None else float(crit.T_crit)

    def sample(s_nodes: np.ndarray) -> np.ndarray:
        T = T_crit * (1.0 - np.atleast_1d(np.asarray(s_nodes, dtype=float)) ** 2)
        rho_L, rho_V = solve(np.clip(T, T_min, T_top))
        return np.stack([np.asarray(rho_L), np.asarray(rho_V)], axis=-1)

    def props(rho: np.ndarray, T: np.ndarray) -> dict:
        """The EOS at a density the fit has just returned."""
        out = props_eos(jnp.asarray(rho), jnp.asarray(T))
        return {k: np.asarray(out[k]) for k in ("P", "h", "s", "u")}

    return fit_saturation(
        eos,
        T_crit,
        T_min,
        T_top,
        sample,
        props,
        degree,
        log_components=(_INDEX["rho_V"],),
    )
