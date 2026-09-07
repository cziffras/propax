from typing import Tuple

import jax.numpy as jnp

from propax.fluids.generic import HelmholtzEOS

from .tolerances import TOL


def rho_bounds(saturation) -> Tuple[float, float]:
    """Every density the correlation describes."""
    return (
        TOL.domain.rho_lo_frac * saturation.eos.rho_crit_mass,
        saturation.rho_max,
    )


def T_bounds(eos: HelmholtzEOS) -> Tuple[float, float]:
    return eos.T_triple, eos.T_max


def node_is_valid(saturation, rho, T):
    T_lo, T_hi = T_bounds(saturation.eos)
    return (rho > 0.0) & (rho <= saturation.rho_max) & (T > T_lo) & (T < T_hi)


def is_two_phase(quality, sat_is_valid, slack: float = 0.0):
    """
    If the sat solve converged and the quality is withing [-slack, 1 + slack],
    this is done to avoid misclassifying biphasic states.
    """
    return sat_is_valid & (quality >= -slack) & (quality <= 1.0 + slack)


def clamp_quality(quality, slack: float = 0.0):
    """The slack lets a quality out of [0, 1]; the lever rule may not see more
    than the caller asked for."""
    return jnp.clip(quality, -slack, 1.0 + slack)
