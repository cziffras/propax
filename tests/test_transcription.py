import jax
import jax.numpy as jnp
import numpy as np
import pytest

CP = pytest.importorskip(
    "CoolProp.CoolProp", reason="the published correlation is read through CoolProp"
)

from propax.core.config import ThermoVar  # noqa: E402
from propax.fluids.schema import DATA_DIR, load_fluid_file  # noqa: E402
from propax.utils.make_utils.eos import verify_eos  # noqa: E402

from .conftest import TRANSCRIPTION  # noqa: E402

N_PROBE = 400

BY_INTERNAL = {v.internal_key: v for v in ThermoVar}
INPUTS = (ThermoVar.D, ThermoVar.T)

# CoolProp spells these `d(P)/d(D)|T`
DERIVATIVES = (
    (ThermoVar.P, ThermoVar.D, ThermoVar.T),
    (ThermoVar.P, ThermoVar.T, ThermoVar.D),
    (ThermoVar.H, ThermoVar.T, ThermoVar.D),
)


def _median_gap(mine, theirs):
    theirs = np.asarray(theirs)
    scale = np.maximum(np.abs(theirs), np.abs(theirs).max())
    return float(np.median(np.abs(np.asarray(mine) - theirs) / scale))


def test_the_shipped_eos_is_the_published_one(fluid_name, cp_fluid, x64):
    if not x64:
        pytest.skip("verify_eos gates make_fluid at 1e-9, below float32 round-off")
    block = load_fluid_file(DATA_DIR / f"{fluid_name}.json").eos.model_dump()
    assert verify_eos(cp_fluid, block) < TRANSCRIPTION.eos_transcription


def test_every_property_matches_the_source(grid, cp_fluid):
    probe = grid.where(grid.single_phase).sample(N_PROBE)
    rho, T = np.asarray(probe.rho), np.asarray(probe.T)

    for name, mine in probe.props.items():
        var = BY_INTERNAL[name]
        if var in INPUTS:
            continue
        ref = CP.PropsSI(var.value, "D", rho, "T", T, cp_fluid)
        gap = _median_gap(mine, ref)
        assert gap < TRANSCRIPTION.state_from_rho_T, f"{var.value}: {gap:.2e}"


def test_the_derivatives_match_the_source(eos, grid, cp_fluid):
    """Autodiff against the same derivatives as CoolProp publishes them."""
    probe = grid.where(grid.single_phase).sample(N_PROBE // 4)
    rho, T = probe.rho, probe.T

    for var, wrt, held in DERIVATIVES:

        def prop(r, t, var=var):
            return eos.props_rhoT(r, t)[var]

        argnum = 0 if wrt is ThermoVar.D else 1
        got = jax.jit(jax.vmap(jax.grad(prop, argnum)))(rho, T)
        key = f"d({var.value})/d({wrt.value})|{held.value}"
        ref = CP.PropsSI(key, "D", np.asarray(rho), "T", np.asarray(T), cp_fluid)
        gap = _median_gap(got, ref)
        assert gap < TRANSCRIPTION.derivative, f"{key}: {gap:.2e}"


class TestTheSolvedCriticalPoint:
    """propax does not read the published critical point, it solves for the one
    the correlation itself has. These read the shipped constants, which is what
    a user gets, rather than re-running the offline solve (which is slow due to
    mpmath precision)."""

    def test_both_conditions_vanish(self, eos):
        T, rho = jnp.asarray(eos.T_crit), jnp.asarray(eos.rho_crit_mass)

        def P(r):
            return eos.props_rhoT(r, T)[ThermoVar.P]

        scale = float(P(rho)) / float(rho)
        assert abs(float(jax.grad(P)(rho))) / scale < TRANSCRIPTION.criticality_first
        assert (
            abs(float(jax.grad(jax.grad(P))(rho))) * float(rho) / scale
            < TRANSCRIPTION.criticality_second
        )

    def test_it_lands_where_the_source_says(self, eos, cp_fluid):
        published = CP.PropsSI("rhomolar_critical", cp_fluid)
        if published == float(eos.rho_red_mol):
            pytest.skip(f"{cp_fluid}: CoolProp reports the reducing point as critical")

        solved = float(eos.rho_crit_mass) / float(eos.molar_mass)
        assert abs(solved / published - 1.0) < TRANSCRIPTION.critical_point_vs_source


def test_the_fluid_is_thermodynamically_stable(grid):
    off_dome = grid.where(grid.single_phase)
    cv = np.asarray(off_dome.props[ThermoVar.CVMASS.internal_key])
    cp = np.asarray(off_dome.props[ThermoVar.CPMASS.internal_key])
    assert (cv > 0.0).all(), "c_v <= 0 is thermally unstable"
    assert (cp > cv).all(), "c_p must exceed c_v away from the dome"
