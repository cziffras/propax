import jax
import jax.numpy as jnp
import numpy as np
import pytest

from propax import Interface
from propax.core.config import ThermoVar

from .conftest import RESIDUAL

N_STATES = 4000

# a quality is not a function of (rho, T), so the saturated route is the one
# pair family this re-injection cannot speak about
PAIRS = [
    tuple(ThermoVar(n) for n in pair)
    for pair, route in sorted(Interface.supported_pairs().items())
    if route != "saturated"
]


def _quality(saturation, rho, T):
    """Where a density sits between the two saturated branches at T."""
    rho_L, rho_V = jax.jit(jax.vmap(saturation.densities))(T)
    v, v_L, v_V = 1.0 / rho, 1.0 / rho_L, 1.0 / rho_V
    return (v - v_L) / (v_V - v_L)


def _worst_residual(got, wanted, scale, keep):
    """How far the answer sits from the value it was asked to reproduce."""
    got, wanted = np.asarray(got), np.asarray(wanted)
    denom = np.maximum(scale, np.abs(wanted))
    return float((np.abs(got - wanted) / denom)[keep].max())


def test_the_two_phase_flash_recovers_the_state_it_was_built_from(props, grid):
    """Under the dome the answer is not read back through the EOS.

    (rho, T) there names a mixture, not a homogeneous state, so the reference
    is built from our own saturation curve by the lever rule and the flash has
    to return the temperature and the quality it was made of. No oracle, and
    nothing the flash could satisfy by echoing its inputs: T and the quality
    are precisely what it has to solve for.
    """
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
    """Whether the flash solved the equation it was given, on its own terms.

    No oracle: the answer goes back into the EOS and must reproduce the two
    values it was asked for, each scaled by its own magnitude so one threshold
    means the same for a pressure and for an entropy. Whether the root is the
    physical one is a separate question, and CoolProp's to answer.
    """
    off_dome = grid.where(grid.single_phase).sample(N_STATES)
    asked = {v: off_dome.props[v.internal_key] for v in pair}

    out, ok = jax.jit(
        jax.vmap(lambda a, b: props.flash(pair[0].value, a, pair[1].value, b))
    )(*asked.values())
    solved = np.asarray(ok)

    assert solved.mean() >= RESIDUAL.solved_fraction, (
        f"the flash gave up on {1 - solved.mean():.1%} of {len(off_dome)} states"
    )

    back = jax.jit(jax.vmap(props.props_rhoT))(
        jnp.asarray(out["rho"]), jnp.asarray(out["T"])
    )
    for var, wanted in asked.items():
        worst = _worst_residual(back[var.internal_key], wanted, var.spec.scale, solved)
        assert worst < RESIDUAL.bound, (
            f"{var.value} does not come back: worst scaled residual {worst:.2e}"
        )
