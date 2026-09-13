import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from propax.core.config import ThermoVar

from .conftest import SAMPLED

STATE_KEYS = {"P", "rho", "T", "u", "h", "s", "cv", "cp"}
BRANCH_KEYS = {"rho", "h", "s", "u"}

SWITCHED_AFTER_CREATE = """
import sys
import jax
from propax import Interface

# interface created before setting precision
itf = Interface.create(sys.argv[1], with_transport=False) 
jax.config.update("jax_enable_x64", True)
e = itf.eos
T = float(e.T_crit) + 0.5 * (float(e.T_max) - float(e.T_crit))
P = float(e.P_crit) + 0.5 * (float(e.P_max) - float(e.P_crit))
state, ok = itf.flash("P", P, "T", T)
assert bool(ok), "(P, T) did not converge"
back, ok = itf.flash("P", P, "S", state["s"])
assert bool(ok), "(P, S) did not converge"
assert back["h"].dtype == jax.numpy.float64, back["h"].dtype
"""


def test_props_rhoT_returns_the_expected_keys(props, grid):
    assert set(grid.props) == STATE_KEYS


def test_the_saturated_branches_carry_the_expected_keys(saturation, dome):
    state = saturation.state_T(dome[len(dome) // 2])
    assert set(state.L.keys()) == BRANCH_KEYS
    assert set(state.V.keys()) == BRANCH_KEYS


def test_out_of_domain_is_nan(props, eos, grid):
    """A refusal is a NaN, never a plausible number."""
    T = float(grid.T[len(grid) // 2])
    for rho, t in ((-10.0, T), (0.0, T), (10.0, -5.0)):
        state = props.props_rhoT(jnp.asarray(rho), jnp.asarray(t))
        assert jnp.isnan(state[ThermoVar.P]), (rho, t)
        assert jnp.isnan(state[ThermoVar.H]), (rho, t)


def test_the_flash_vmaps(props, grid):
    probe = grid.where(grid.single_phase).sample(8)
    out, ok = jax.jit(jax.vmap(lambda p, h: props.flash("P", p, "H", h)))(
        probe.props[ThermoVar.P.internal_key], probe.props[ThermoVar.H.internal_key]
    )
    assert np.asarray(ok).shape == (len(probe),)
    for key in STATE_KEYS:
        assert out[key].shape == (len(probe),), key
    assert not jnp.any(jnp.isnan(out[ThermoVar.T.internal_key]))


def test_the_saturation_curve_vmaps(saturation, dome):
    state = jax.jit(jax.vmap(saturation.state_T))(dome)
    assert state.P.shape == dome.shape
    assert state.L[ThermoVar.D].shape == dome.shape
    assert not jnp.any(jnp.isnan(state.P))
    assert not jnp.any(jnp.isnan(state.L[ThermoVar.D]))


def test_a_precision_switched_after_create_is_followed(x64):
    if not x64:
        pytest.skip("the subprocess picks its own precision, the float64 run covers it")
    if not SAMPLED:
        pytest.skip("no shipped fluid")
    run = subprocess.run(
        [sys.executable, "-c", SWITCHED_AFTER_CREATE, SAMPLED[0]],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stderr[-3000:]
