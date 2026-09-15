import jax
import jax.numpy as jnp
import numpy as np
import pytest

CP = pytest.importorskip(
    "CoolProp.CoolProp", reason="CoolProp only serves as a reference oracle"
)

from propax.core.config import ThermoVar  # noqa: E402

from .conftest import RESIDUAL, RIGHT_ROOT  # noqa: E402

LIQUID, VAPOUR = 0, 1


class TestTheCurveIsConsistentWithItself:
    def test_pressure_inverts(self, saturation, dome):
        P = jax.jit(jax.vmap(lambda t: saturation.state_T(t).P))(dome)
        back = jax.jit(jax.vmap(lambda p: saturation.state_P(p).T))(P)
        assert np.asarray(back) == pytest.approx(
            np.asarray(dome), rel=RESIDUAL.saturation_inverts
        )

    def test_density_inverts_onto_its_branch(self, saturation, dome):
        rho_L, rho_V = jax.jit(jax.vmap(saturation.densities))(dome)
        for rho in (rho_L, rho_V):
            T_back, on_dome = jax.jit(jax.vmap(saturation.T_of_rho))(rho)
            assert bool(jnp.all(on_dome))
            assert np.asarray(T_back) == pytest.approx(
                np.asarray(dome), rel=RESIDUAL.saturation_inverts
            )

    def test_the_two_branches_are_in_equilibrium(self, eos, saturation, dome):
        """Equal fugacity is the fit criterion, check it."""

        def ln_fugacity(rho_mass, t):
            rho_mol = rho_mass / eos.molar_mass
            delta = rho_mol / eos.rho_red_mol
            ar, ar_d = jax.value_and_grad(eos.alphar, argnums=0)(delta, eos.T_red / t)
            return ar + delta * ar_d + jnp.log(rho_mol * eos.R_u * t)

        def gap(t):
            rho_L, rho_V = saturation.densities(t)
            return jnp.abs(ln_fugacity(rho_L, t) - ln_fugacity(rho_V, t))

        worst = float(jnp.max(jax.jit(jax.vmap(gap))(dome)))
        assert worst < RESIDUAL.equal_fugacity

    def test_is_valid_covers_exactly_the_dome(self, saturation):
        """Closed at both ends."""
        T_min, T_crit = float(saturation.T_min), float(saturation.T_crit)
        step = 1e-4 * (T_crit - T_min)
        for T in (T_min, T_crit):
            assert bool(saturation.state_T(T).is_valid), T
        for T in (T_min - step, T_crit + step):
            assert not bool(saturation.state_T(T).is_valid), T


class TestTheCurveIsTheSameOneCoolPropFound:
    def test_the_saturated_densities_agree(self, saturation, dome, cp_fluid):
        T = np.asarray(dome)
        rho = dict(zip((LIQUID, VAPOUR), jax.vmap(saturation.densities)(dome)))
        for q, got in rho.items():
            ref = CP.PropsSI("D", "T", T, "Q", np.full_like(T, q), cp_fluid)
            assert np.asarray(got) == pytest.approx(ref, rel=RIGHT_ROOT.density)

    def test_the_branch_energies_agree(self, saturation, dome, cp_fluid):
        T = np.asarray(dome)
        state = jax.jit(jax.vmap(saturation.state_T))(dome)
        for var in (ThermoVar.H, ThermoVar.S):
            for branch, q in ((state.L, LIQUID), (state.V, VAPOUR)):
                ref = CP.PropsSI(var.value, "T", T, "Q", np.full_like(T, q), cp_fluid)
                assert np.asarray(branch[var]) == pytest.approx(
                    ref, rel=RIGHT_ROOT.energy
                )
