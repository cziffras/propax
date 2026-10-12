from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from propax.fluids.generic import HelmholtzEOS
from propax.utils.types import jaxBool

from ..tolerances import TOL


def is_a_root(got: Array, wanted: Array, scale, noise=0.0) -> jaxBool:
    """Whether `got` reproduces `wanted`, this is a key acceptance rule of every flash.

    |got - wanted| <= root_rtol * max(scale, |wanted|) + evaluation_noise * noise

    - The first term is is an accuracy relative to |wanted|, floored
    at `scale` so that a target near zero is not held to an infinite precision.
    Callers pass max(spec.scale, state_sensitivity), which widens it where the
    state is ill-conditioned with respect to (rho, T).
    - The second is the accuracy inherent to the arithmetics of the EOS : `noise`
    is `HelmholtzEOS.noise_rhoT`, the round-off of an EOS evaluation whose terms
    cancel. It is useless in float64 but crucial for some fluids in float32.
    """
    return (
        jnp.abs(got - wanted)
        <= TOL.acc.root_rtol * jnp.maximum(scale, jnp.abs(wanted))
        + TOL.acc.evaluation_noise * noise
    )


def node_is_valid(saturation, rho, T):
    T_lo, T_hi = T_bounds(saturation.eos)
    return (rho > 0.0) & (rho <= saturation.rho_max) & (T > T_lo) & (T < T_hi)


def state_sensitivity(of_rho_T: Callable, rho, T):
    rho = jax.lax.stop_gradient(jnp.asarray(rho))
    T = jax.lax.stop_gradient(jnp.asarray(T))
    _, along_rho = jax.jvp(lambda r: of_rho_T(r, T), (rho,), (rho,))
    _, along_T = jax.jvp(lambda t: of_rho_T(rho, t), (T,), (T,))
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
