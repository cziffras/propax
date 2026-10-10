from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from propax.fluids.generic import HelmholtzEOS
from propax.utils.types import jaxBool

from ..tolerances import TOL


def is_a_root(got: Array, wanted: Array, scale) -> jaxBool:
    return jnp.abs(got - wanted) <= TOL.acc.root_atol * jnp.maximum(
        scale, jnp.abs(wanted)
    )


def node_is_valid(saturation, rho, T):
    T_lo, T_hi = T_bounds(saturation.eos)
    return (rho > 0.0) & (rho <= saturation.rho_max) & (T > T_lo) & (T < T_hi)


def state_sensitivity(eos: HelmholtzEOS, rho, T):
    rho = jax.lax.stop_gradient(jnp.asarray(rho))
    T = jax.lax.stop_gradient(jnp.asarray(T))
    _, along_rho = jax.jvp(lambda r: eos.props_rhoT(r, T), (rho,), (rho,))
    _, along_T = jax.jvp(lambda t: eos.props_rhoT(rho, t), (T,), (T,))
    return jax.tree.map(lambda a, b: jnp.abs(a) + jnp.abs(b), along_rho, along_T)


def rho_bounds(saturation) -> Tuple[float, float]:
    return (
        TOL.domain.rho_lo_frac * saturation.eos.rho_crit_mass,
        saturation.rho_max,
    )


def T_bounds(eos: HelmholtzEOS) -> Tuple[float, float]:
    return eos.T_triple, eos.T_max


def is_two_phase(quality, sat_is_valid, slack: float = 0.0):
    """
    If the sat solve converged and the quality is withing [-slack, 1 + slack],
    this is done to avoid misclassifying biphasic states.
    """
    return sat_is_valid & (quality >= -slack) & (quality <= 1.0 + slack)


def clamp_quality(quality, slack: float = 0.0):
    return jnp.clip(quality, -slack, 1.0 + slack)
