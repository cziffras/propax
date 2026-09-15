import logging

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from ...core.config import (  # noqa: E402
    TableSpec,
    ThermoVar,
)
from ...core.flash.results import mixture_state  # noqa: E402
from ...core.flash.two_phase import solve_two_phase  # noqa: E402
from ...core.saturation import Superancillary  # noqa: E402
from .helpers import _mapped  # noqa: E402

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# How far outside the dome a node is still given a mixture value, allows
# ovelapping between monophasic table and dome values
_GHOST_QUALITY = 0.05

# How finely the curve is walked to bound an axis
_N_BOUND_SCAN = 512


def _mixture_value(saturation: Superancillary, var: ThermoVar, T, x):
    """One output of the mixture at (T_sat, quality)."""
    if var == ThermoVar.Q:
        return jnp.asarray(x)
    return mixture_state(saturation, T, x)[var]


def _solve_dome_state(cfg: TableSpec, X1_phys, X2_phys, saturation: Superancillary):
    """Locate grid nodes inside the saturation dome; returns (mask, T_sat, x).

    T_sat and x are also returned just outside the dome, where the lever rule
    still extends smoothly; NaN marks the nodes no mixture describes at all.
    """
    tvar1, tvar2 = cfg.x_axis.variable, cfg.y_axis.variable
    shape = X1_phys.shape

    if {ThermoVar.P, ThermoVar.T} <= {tvar1, tvar2}:
        # specifying (T, P) excludes any biphasic state
        return np.zeros(shape, dtype=bool), np.zeros(shape), np.zeros(shape)

    def one(v1, v2):
        valid, T_sat, x = solve_two_phase(
            saturation.eos,
            saturation,
            tvar1,
            v1,
            tvar2,
            v2,
            jnp.asarray(False),
            slack=_GHOST_QUALITY,
        )
        return jnp.stack([T_sat, x, jnp.asarray(valid, T_sat.dtype)])

    out = np.asarray(
        _mapped(one, jnp.asarray(X1_phys).ravel(), jnp.asarray(X2_phys).ravel())
    )
    T_sat = out[:, 0].reshape(shape)
    x = out[:, 1].reshape(shape)
    usable = out[:, 2].reshape(shape).astype(bool)

    T_sat = np.where(usable, T_sat, np.nan)
    x = np.where(usable, x, np.nan)
    return usable & (x >= 0.0) & (x <= 1.0), T_sat, x


def _dome_values(outputs, saturation: Superancillary, T_sat, x):
    """The mixture's outputs at every node the solve resolved."""

    def node(T, q):
        return jnp.stack([_mixture_value(saturation, var, T, q) for var in outputs])

    resolved = np.isfinite(T_sat).ravel() & np.isfinite(x).ravel()
    vals = np.asarray(
        _mapped(
            node,
            jnp.asarray(np.nan_to_num(T_sat).ravel()),
            jnp.asarray(np.nan_to_num(x).ravel()),
        )
    )
    # a node no mixture describes carries no value, and says so
    vals = np.where(resolved[:, None], vals, np.nan)
    return {var: vals[:, i] for i, var in enumerate(outputs)}


def _dome_derivatives(cfg, outputs, saturation: Superancillary, T_sat, x):
    """d(out)/d(axes) at dome nodes, via (T_sat, x) as internal variables.

    Same shape as the single-phase `J_out @ inv(J_in)`.
    """
    axes = (cfg.x_axis.variable, cfg.y_axis.variable)

    def jacobian(vars_, z):
        return jax.jacfwd(
            lambda w: jnp.stack(
                [_mixture_value(saturation, v, w[0], w[1]) for v in vars_]
            )
        )(z)

    def node(T, q):
        z = jnp.stack([T, q])
        return jacobian(outputs, z) @ jnp.linalg.inv(jacobian(axes, z))  # (C, 2)

    return _mapped(node, jnp.asarray(T_sat), jnp.asarray(x))


def dome_axis_bounds(cfg, saturation: Superancillary):
    """The window of each axis the dome actually occupies.

    Walked in s = sqrt(1 - T/T_crit) rather than in T: the dome closes like
    sqrt(theta), so a walk uniform in T spends its last step on the whole
    critical region.
    """
    s_max = float(saturation.s_of_T(jnp.asarray(saturation.T_min)))
    T = np.asarray(saturation.T_of_s(jnp.linspace(0.0, s_max, _N_BOUND_SCAN)))

    def segment(axis_cfg):
        """Saturated liquid and vapour values of this axis' variable, per row"""
        var = axis_cfg.variable
        edges = _mapped(
            lambda t: jnp.stack(
                [
                    _mixture_value(saturation, var, t, jnp.asarray(0.0)),
                    _mixture_value(saturation, var, t, jnp.asarray(1.0)),
                ]
            ),
            jnp.asarray(T),
        )
        edges = np.asarray(edges)
        return edges[:, 0], edges[:, 1]

    # only the rows whose dome segment meets the configured domain
    keep = np.ones(T.shape, dtype=bool)
    for axis_cfg in (cfg.x_axis, cfg.y_axis):
        L, V = segment(axis_cfg)
        keep &= (np.maximum(L, V) >= axis_cfg.min_val) & (
            np.minimum(L, V) <= axis_cfg.max_val
        )
    if not keep.any():
        raise ValueError(f"{cfg.name}: the dome does not meet the configured domain")

    # The curve is continuous but walked in steps: between the last excluded
    # row and the first kept one there are real two-phase states
    keep[:-1] |= keep[1:]
    keep[1:] |= keep[:-1].copy()

    def bounds(axis_cfg):
        L, V = segment(axis_cfg)
        lo = float(np.minimum(L, V)[keep].min())
        hi = float(np.maximum(L, V)[keep].max())
        return max(lo, axis_cfg.min_val), min(hi, axis_cfg.max_val)

    return bounds(cfg.x_axis), bounds(cfg.y_axis)
