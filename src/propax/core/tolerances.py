"""
Gathered here all tolerances and number of solver iterations used
across the runtime Interface, allows float32 and float64 modes.
"""

from dataclasses import dataclass, field, fields, replace
from typing import Dict, Self

import jax
import jax.numpy as jnp

_SCALING_FACTOR = float(jnp.finfo(jnp.float32).eps / jnp.finfo(jnp.float64).eps)


def _loosen(value, factor: float, loosest):
    if isinstance(value, dict):
        return {
            k: _loosen(
                v, factor, loosest.get(k) if isinstance(loosest, dict) else loosest
            )
            for k, v in value.items()
        }
    widened = value * factor
    return widened if loosest is None else min(widened, loosest)


def _above_ceiling(value, loosest) -> bool:
    if loosest is None:
        return False
    if isinstance(value, dict):
        return any(
            _above_ceiling(v, loosest.get(k) if isinstance(loosest, dict) else loosest)
            for k, v in value.items()
        )
    return value > loosest


@dataclass(frozen=True)
class PrecisionScaled:
    def __post_init__(self):
        for f in fields(self):
            if _above_ceiling(getattr(self, f.name), f.metadata.get("loosest")):
                raise ValueError(
                    f"{f.name} sits above its own `loosest`, so widening it for a "
                    "coarser format would tighten it instead"
                )

    def scaled(self, factor: float) -> Self:
        """`scales: False` marks a bound on a quantity the runtime format never
        touches, such as one solved offline in mpmath."""
        return type(self)(
            **{
                f.name: getattr(self, f.name)
                if f.metadata.get("scales") is False
                else _loosen(getattr(self, f.name), factor, f.metadata.get("loosest"))
                for f in fields(self)
            }
        )


@dataclass(frozen=True)
class Accuracies(PrecisionScaled):
    #  bracketed solves
    outer_rtol: float = 1e-13
    """Temperature along an isobar, enclosing the density solve at each step."""

    newton_rtol: float = 1e-15
    """Relative step at which the safeguarded Newton stops. A few ulp."""

    root_atol: float = field(default=1e-8, metadata={"loosest": 1e-5})
    """Acceptance, not convergence: how small a scaled residual counts as a
    root. It rejects an answer rather than stopping a loop."""

    #  the saturation line
    channel_edge: float = 1e-12
    """How far past its ends a fitted channel still answers, as a relative tolerance."""

    sat_edge_nudge: float = 1e-12
    """How far off the saturation line a bracket endpoint is pushed, relative."""

    quality_slack: float = field(default=1e-12, metadata={"scales": False})
    """Slack on [0, 1] when P or T is given. Outward only, and only there.

    Widened for float32 it only reaches further outside [0, 1], where the 
    states are single-phase and answering with the saturated branch is simply wrong,
    whence `scales` is set to false.
    """

    equilibrium: Dict[str, float] = field(
        default_factory=lambda: {"rtol": 1e-9, "atol": 1e-9},
        metadata={"loosest": 1e-4},
    )
    """The optimistix Newtons of `saturation` and `two_phase`."""

    #  degeneracies (when getting to critical point)
    lever_denom_atol: float = field(default=1e-9, metadata={"loosest": 1e-4})
    """Below this the saturated pair is too close for the lever rule to divide."""


@dataclass(frozen=True)
class Caps:
    newton_steps: int = 40
    warm_newton_steps: int = 20
    isobar_steps: int = 100
    equilibrium_steps: int = 10
    n_seed_scan: int = 64


@dataclass(frozen=True)
class Domain:
    rho_lo_frac: float = 1e-6
    """Lowest density considered, as a fraction of rho_crit."""

    rho_ceiling_step: float = 0.05
    """Step used to probe the liquid when solving for rho_max, see utils.exact.constants.py."""


@dataclass(frozen=True)
class Tolerances:
    acc: Accuracies = field(default_factory=Accuracies)
    caps: Caps = field(default_factory=Caps)
    domain: Domain = field(default_factory=Domain)

    def scaled(self, factor: float) -> "Tolerances":
        # only accuracies need to be scaled
        return replace(self, acc=self.acc.scaled(factor))


TOL64 = Tolerances()
TOL32 = TOL64.scaled(_SCALING_FACTOR)


class _PrecisionSelected:
    __slots__ = ()  # _Precision selected has no attributes for itself

    def _table(self) -> Tolerances:
        return TOL64 if jax.config.read("jax_enable_x64") else TOL32

    def __getattr__(self, name: str):
        return getattr(self._table(), name)

    def __repr__(self) -> str:
        return f"<TOL -> {'TOL64' if self._table() is TOL64 else 'TOL32'}>"


TOL = _PrecisionSelected()
