import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from propax.fluids._registry import EQS_REGISTRY
from propax.fluids.schema import (
    FluidDefinition,
    discover_fluid_files,
    load_fluid_file,
)


# still a draft needs to be completed with all cases of size mismatch
# and recheck validity (as done already when ran the make_fluid command)
class TestSchema:
    def test_every_file_in_the_directory_becomes_a_registry_entry(self):
        found = discover_fluid_files()
        assert set(found) == set(EQS_REGISTRY)

    def test_bundled_files_parse(self):
        for path in discover_fluid_files().values():
            defn = load_fluid_file(path)
            assert defn.eos.T_red > defn.eos.T_triple
            assert defn.eos.P_red > defn.eos.P_triple
            assert defn.eos.T_crit > defn.eos.T_triple
            assert defn.eos.rho_crit_mol > 0.0
            assert defn.eos.P_crit > defn.eos.P_triple

    def test_mismatched_coefficient_lengths_rejected(self):
        raw = load_fluid_file(next(iter(discover_fluid_files().values()))).model_dump()
        raw["eos"]["residual"]["poly_n"] = raw["eos"]["residual"]["poly_n"][:-1]
        with pytest.raises(ValidationError):
            FluidDefinition.model_validate(raw)

    def test_unknown_transport_family_rejected(self):
        for path in discover_fluid_files().values():
            raw = load_fluid_file(path).model_dump()
            if raw.get("viscosity"):
                break
        else:
            pytest.skip("no shipped fluid carries a viscosity correlation")

        raw["viscosity"]["model"] = "not_a_real_correlation"
        with pytest.raises(ValidationError):
            FluidDefinition.model_validate(raw)


class TestGenericModules:
    @pytest.mark.parametrize("fluid", sorted(EQS_REGISTRY))
    def test_modules_evaluate(self, fluid):
        eos_f, anc_f, visc_f, cond_f = EQS_REGISTRY[fluid]
        eos = eos_f()
        visc = visc_f(eos=eos)  # None when the fluid ships EOS-only
        cond = cond_f(eos=eos, viscosity=visc)

        # fluid-appropriate single-phase gas state: low density, supercritical T
        rho, T = jnp.asarray(2.0), jnp.asarray(1.2 * eos.T_crit)
        props = eos.props_rhoT(rho, T)
        assert float(props["P"]) > 0.0
        assert float(props["cp"]) > float(props["cv"]) > 0.0
        # transport is optional (many CoolProp fluids publish none)
        if visc is not None:
            assert 0.0 < float(visc.viscosity_rhoT(rho, T)) < 1e-3
        if cond is not None:
            assert 0.0 < float(cond.conductivity_rhoT(rho, T)) < 10.0
