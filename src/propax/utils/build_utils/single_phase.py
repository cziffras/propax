import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from ...core.config import TableSpec  # noqa: E402
from ...core.flash.dispatch import check_supported  # noqa: E402
from ...core.flash.single_phase import solve_single_phase  # noqa: E402


def make_single_phase_solver(eos, sat_mod, cfg: TableSpec):
    """Vectorised (x1, x2) -> (ok, rho, T) for one table's axes.

    Metastable states are wanted here and are what comes back: these are the
    single-phase solvers, so under the dome they follow the branch's
    continuation rather than the equilibrium. The dome table holds the
    equilibrium values, and the runtime picks between the two.
    """
    v1, v2 = cfg.x_axis.variable, cfg.y_axis.variable
    check_supported(v1, v2)
    active = jnp.array(False)
    return jax.jit(
        jax.vmap(
            lambda a1, a2: solve_single_phase(eos, sat_mod, v1, a1, v2, a2, active)
        )
    )


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
