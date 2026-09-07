import jax
import jax.numpy as jnp
import numpy as np

from propax.core.config import ThermoVar

STATE_KEYS = {"P", "rho", "T", "u", "h", "s", "cv", "cp"}
BRANCH_KEYS = {"rho", "h", "s", "u"}


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
