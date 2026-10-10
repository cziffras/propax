import jax
import jax.numpy as jnp
import numpy as np
import pytest

from propax import Interface
from propax.core.config import ThermoVar
from propax.core.flash.flash_utils import state_sensitivity

from .conftest import RESIDUAL

N_STATES = 4000

PAIRS = [
    tuple(ThermoVar(n) for n in pair)
    for pair, route in sorted(Interface.supported_pairs().items())
    if route != "saturated"
]


def _quality(saturation, rho, T):
    rho_L, rho_V = jax.jit(jax.vmap(saturation.densities))(T)
    v, v_L, v_V = 1.0 / rho, 1.0 / rho_L, 1.0 / rho_V
    return (v - v_L) / (v_V - v_L)


def _worst_residual(got, wanted, scale, keep):
    got, wanted = np.asarray(got), np.asarray(wanted)
    denom = np.maximum(
        scale, np.abs(wanted)
    )  # gives either an a tol for small values or a rtol for greater ones
    return float((np.abs(got - wanted) / denom)[keep].max())


def test_the_two_phase_flash_recovers_the_state_it_was_built_from(props, grid):
    dome = grid.where(grid.two_phase).sample(N_STATES)
    T, rho = dome.T, dome.rho
    x = _quality(props.saturation, rho, T)

    sat = jax.jit(jax.vmap(props.saturation.state_T))(T)
    u = (1.0 - x) * sat.L[ThermoVar.U] + x * sat.V[ThermoVar.U]

    out, ok = jax.jit(jax.vmap(lambda r, e: props.flash("D", r, "U", e)))(rho, u)
    solved = np.asarray(ok)
    assert solved.mean() >= RESIDUAL.solved_fraction, (
        f"the flash gave up on {1 - solved.mean():.1%} of {len(dome)} mixtures"
    )

    T_back = jnp.asarray(out["T"])
    for label, got, wanted, scale in (
        ("T", T_back, T, ThermoVar.T.spec.scale),
        (
            "quality",
            _quality(props.saturation, jnp.asarray(out["rho"]), T_back),
            x,
            1.0,
        ),
    ):
        worst = _worst_residual(got, wanted, scale, solved)
        assert worst < RESIDUAL.bound, (
            f"{label} does not come back: worst scaled residual {worst:.2e}"
        )


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: "".join(v.value for v in p))
def test_the_flash_zeroes_its_residual(props, grid, pair):
    off_dome = grid.where(grid.single_phase).sample(N_STATES)
    asked = {v: off_dome.props[v.internal_key] for v in pair}

    out, ok = jax.jit(
        jax.vmap(lambda a, b: props.flash(pair[0].value, a, pair[1].value, b))
    )(*asked.values())
    solved = np.asarray(ok)

    assert solved.mean() >= RESIDUAL.solved_fraction, (
        f"the flash gave up on {1 - solved.mean():.1%} of {len(off_dome)} states"
    )

    rho_out, T_out = jnp.asarray(out["rho"]), jnp.asarray(out["T"])
    back = jax.jit(jax.vmap(props.props_rhoT))(rho_out, T_out)
    sensitivity = jax.jit(jax.vmap(lambda r, t: state_sensitivity(props.eos, r, t)))(
        rho_out, T_out
    )
    for var, wanted in asked.items():
        scale = np.maximum(var.spec.scale, np.asarray(sensitivity[var.internal_key]))
        worst = _worst_residual(back[var.internal_key], wanted, scale, solved)
        assert worst < RESIDUAL.bound, (
            f"{var.value} does not come back: worst scaled residual {worst:.2e}"
        )
