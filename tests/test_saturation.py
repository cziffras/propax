import jax
import jax.numpy as jnp
import numpy as np
import pytest

CP = pytest.importorskip(
    "CoolProp.CoolProp", reason="CoolProp only serves as a reference oracle"
)

from propax.core.config import ThermoVar  # noqa: E402
from propax.core.flash.flash_utils import state_sensitivity  # noqa: E402
from propax.core.tolerances import TOL  # noqa: E402

from .conftest import RESIDUAL, RIGHT_ROOT  # noqa: E402

LIQUID, VAPOUR = 0, 1


def _ln_fugacity(eos, rho_mass, t):
    rho_mol = rho_mass / eos.molar_mass
    delta = rho_mol / eos.rho_red_mol
    ar, ar_d = jax.value_and_grad(eos.alphar, argnums=0)(delta, eos.T_red / t)
    return ar + delta * ar_d + jnp.log(rho_mol * eos.R_u * t)


def _equilibrium_gap(eos, rho_L, rho_V, T):
    def ln_fugacity(rho_mass, t):
        return _ln_fugacity(eos, rho_mass, t)

    gap = jnp.abs(ln_fugacity(rho_L, T) - ln_fugacity(rho_V, T))
    sensitivity = state_sensitivity(
        ln_fugacity, rho_L, T, given=ThermoVar.T
    ) + state_sensitivity(ln_fugacity, rho_V, T, given=ThermoVar.T)
    return gap / jnp.maximum(1.0, sensitivity)


def _gap_if_only_one_density_is_wrong(eos, rho_L, rho_V, T):
    def ln_fugacity(rho_mass, t):
        return _ln_fugacity(eos, rho_mass, t)

    def along_density(rho):
        sensitivity = state_sensitivity(ln_fugacity, rho, T, given=ThermoVar.T)
        return jnp.maximum(1.0, sensitivity)

    gap = jnp.abs(ln_fugacity(rho_L, T) - ln_fugacity(rho_V, T))
    return gap / along_density(rho_L), gap / along_density(rho_V)


def _coolprop_dome(eos, dome, cp_fluid):
    T = np.asarray(dome)
    rho = {
        q: CP.PropsSI("D", "T", T, "Q", np.full_like(T, q), cp_fluid)
        for q in (LIQUID, VAPOUR)
    }
    only_L, only_V = jax.jit(
        jax.vmap(lambda a, b, t: _gap_if_only_one_density_is_wrong(eos, a, b, t))
    )(rho[LIQUID], rho[VAPOUR], dome)
    trusted = (np.asarray(only_L) < RESIDUAL.equal_fugacity) & (
        np.asarray(only_V) < RESIDUAL.equal_fugacity
    )
    assert trusted.mean() >= RIGHT_ROOT.oracle_fraction, (
        f"CoolProp's dome is in equilibrium on only {trusted.mean()} of "
        f"{T.size} temperatures"
    )
    return T, rho, trusted


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

        def gap(t):
            rho_L, rho_V = saturation.densities(t)
            return _equilibrium_gap(eos, rho_L, rho_V, t)

        worst = float(jnp.max(jax.jit(jax.vmap(gap))(dome)))
        assert worst < RESIDUAL.equal_fugacity

    def test_is_valid_covers_exactly_the_dome(self, saturation):
        T_min, T_crit = float(saturation.T_min), float(saturation.T_crit)
        step = 1e-4 * (T_crit - T_min)
        for T in (T_min, T_crit):
            assert bool(saturation.state_T(T).is_valid), T
        for T in (T_min - step, T_crit + step):
            assert not bool(saturation.state_T(T).is_valid), T


class TestTheCurveIsTheSameOneCoolPropFound:
    @pytest.fixture(autouse=True)
    def _not_pseudo_pure(self, eos, cp_fluid):
        if eos.pseudo_pure:
            pytest.skip(
                f"{cp_fluid} is pseudo-pure: CoolProp takes its bubble and dew "
                "lines from ancillaries, not from the EOS"
            )

    def test_the_saturated_densities_agree(self, eos, saturation, dome, cp_fluid):
        _, ref, trusted = _coolprop_dome(eos, dome, cp_fluid)
        rho = dict(zip((LIQUID, VAPOUR), jax.vmap(saturation.densities)(dome)))
        for q, got in rho.items():
            assert np.asarray(got)[trusted] == pytest.approx(
                ref[q][trusted], rel=RIGHT_ROOT.density
            )

    def test_the_branch_energies_agree(self, eos, saturation, dome, cp_fluid):
        T, _, trusted = _coolprop_dome(eos, dome, cp_fluid)
        state = jax.jit(jax.vmap(saturation.state_T))(dome)
        for var in (ThermoVar.H, ThermoVar.S):
            for branch, q in ((state.L, LIQUID), (state.V, VAPOUR)):
                ref = CP.PropsSI(var.value, "T", T, "Q", np.full_like(T, q), cp_fluid)
                noise = jax.jit(jax.vmap(eos.noise_rhoT))(branch[ThermoVar.D], dome)
                allowed = np.maximum(
                    RIGHT_ROOT.energy * np.abs(ref),
                    RIGHT_ROOT.energy * var.spec.scale,
                ) + TOL.acc.evaluation_noise * np.asarray(noise[var])
                gap = np.abs(np.asarray(branch[var]) - ref)
                assert (gap <= allowed)[trusted].all(), (
                    f"{var.value}, branch {q}: worst gap over allowance "
                    f"{np.max((gap / allowed)[trusted]):.2f}"
                )
