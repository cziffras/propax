import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from ...core.config import TableSpec, ThermoVar  # noqa: E402
from ...core.flash.dispatch import (  # noqa: E402
    check_supported,
    is_nested_pair,
    one_dim_known_var,
)
from ...core.flash.single_phase import solve_1d, solve_nested  # noqa: E402


def make_single_phase_solver(eos, sat_mod, cfg: TableSpec):
    """Vectorised (x1, x2) -> (rho, T, ok) for one table's axes.

    Metastable states are wanted here and are what comes back: these are the
    single-phase solvers, so under the dome they follow the branch's
    continuation rather than the equilibrium. The dome table holds the
    equilibrium values, and the runtime picks between the two.
    """
    v1, v2 = cfg.x_axis.variable, cfg.y_axis.variable
    active = jnp.array(False)

    known = one_dim_known_var(v1, v2)
    if known is not None:
        other = v2 if v1 == known else v1

        def fn(a1, a2):
            k, o = (a1, a2) if v1 == known else (a2, a1)
            return solve_1d(eos, sat_mod, known, k, other, o, active)

    elif is_nested_pair(v1, v2):
        other = v2 if v1 == ThermoVar.P else v1

        def fn(a1, a2):
            p, o = (a1, a2) if v1 == ThermoVar.P else (a2, a1)
            return solve_nested(eos, sat_mod, other, p, o, active)

    else:
        check_supported(v1, v2)
        raise ValueError(
            f"{cfg.name}: ({v1.value}, {v2.value}) has no single-phase solver. "
            "A table pair must either hold a natural variable with the other "
            "proven monotone in it, or be one of the nested (P, x) pairs."
        )

    return jax.jit(jax.vmap(fn))


def solve_grid(solver, X1_phys, X2_phys):
    """(rho_grid, T_grid, ok_grid), each with the shape of `X1_phys`."""
    shape = X1_phys.shape
    ok, rho, T = solver(
        jnp.asarray(np.asarray(X1_phys).ravel()),
        jnp.asarray(np.asarray(X2_phys).ravel()),
    )
    return (
        np.asarray(rho).reshape(shape),
        np.asarray(T).reshape(shape),
        np.asarray(ok).reshape(shape),
    )
