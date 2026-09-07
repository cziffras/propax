"""The whole loop a user walks to supply a transport term propax has no class for.

`make_fluid --transport-scaffold` leaves a slot like

    "higher_order": {"type": "custom", "name": "<fluid>_higher_order", "params": {...}}

in the fluid file, and the fluid does not load until something is registered under
that name. These tests are the smallest complete example of doing so, and they are
what the docs point at.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from propax import (
    Interface,
    register_conductivity_residual,
    register_viscosity_higher_order,
)
from propax.fluids.generic import ViscositySlots
from propax.fluids.generic.transport_registry import build_conductivity_residual
from propax.fluids.schema import (
    DATA_DIR,
    FluidDefinition,
    discover_fluid_files,
    load_fluid_file,
)
from propax.utils.make_utils import make_fluid


class LinearHigherOrder(eqx.Module):
    """contribution(rho_molar, T, eta0) -> Pa.s, built as cls(**params)"""

    a: float
    b: float

    def contribution(self, rho_molar, T, eta0):
        return self.a * rho_molar + self.b * eta0


def _a_fluid_carrying_viscosity() -> str:
    """A shipped fluid whose definition has a viscosity block to override.

    Only 15 of the 125 shipped fluids carry one, and the first in catalogue
    order is not among them, so this cannot be `next(iter(...))`.
    """
    for name, path in discover_fluid_files().items():
        if load_fluid_file(path).viscosity is not None:
            return name
    pytest.skip("no shipped fluid carries a viscosity correlation")


@pytest.fixture(scope="module")
def viscous_fluid() -> str:
    return _a_fluid_carrying_viscosity()


@pytest.fixture(scope="module")
def propane_definition(viscous_fluid):
    return load_fluid_file(discover_fluid_files()[viscous_fluid]).model_dump()


@pytest.fixture(scope="module")
def propane_eos(viscous_fluid):
    return Interface.create(viscous_fluid).eos


def _with_custom_higher_order(raw: dict, term: str, params: dict) -> dict:
    out = json.loads(json.dumps(raw, default=float))
    out["viscosity"]["higher_order"] = {
        "type": "custom",
        "name": term,
        "params": params,
    }
    return out


def _viscosity_with(raw: dict, term: str, params: dict, eos) -> ViscositySlots:
    defn = FluidDefinition.model_validate(_with_custom_higher_order(raw, term, params))
    return ViscositySlots.from_definition(defn.viscosity, eos=eos)


class TestRegisteringATerm:
    def test_a_builder_receives_the_eos(self, propane_definition, propane_eos):
        seen = {}

        @register_viscosity_higher_order("test_needs_eos")
        def build(params, *, eos):
            seen["T_crit"] = float(eos.T_crit)
            return LinearHigherOrder(**params)

        _viscosity_with(
            propane_definition, "test_needs_eos", {"a": 0.0, "b": 0.0}, propane_eos
        )
        assert seen["T_crit"] == pytest.approx(float(propane_eos.T_crit))

    def test_the_term_is_differentiable_like_the_rest(
        self, propane_definition, propane_eos
    ):
        register_viscosity_higher_order("test_linear", LinearHigherOrder)
        visc = _viscosity_with(
            propane_definition, "test_linear", {"a": 1e-9, "b": 0.5}, propane_eos
        )

        g = jax.grad(lambda r: visc.viscosity_rhoT(r, jnp.asarray(400.0)))(
            jnp.asarray(20.0)
        )
        assert jnp.isfinite(g) and float(g) != 0.0


class TestUnregisteredTerm:
    def test_the_error_names_the_term_and_the_registrar(
        self, propane_definition, propane_eos
    ):
        with pytest.raises(KeyError) as e:
            _viscosity_with(
                propane_definition, "test_never_registered", {}, propane_eos
            )

        message = str(e.value)
        assert "test_never_registered" in message
        assert "register_viscosity_higher_order" in message
        assert "contribution(self, rho_molar, T, eta0)" in message

    def test_a_conductivity_term_points_at_its_own_registrar(self):
        assert callable(register_conductivity_residual)
        with pytest.raises(KeyError) as e:
            build_conductivity_residual("test_absent", {}, eos=None, viscosity=None)
        assert "register_conductivity_residual" in str(e.value)
        assert "contribution(self, rho, T)" in str(e.value)


@pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="make_fluid solves the critical point in mpmath and requires float64",
)
class TestScaffoldWritesIntoTheFluidFile:
    def test_placeholder_lands_in_the_definition(self, tmp_path):
        make_fluid(
            "Hydrogen",
            name="scaffoldtest",
            out_dir=tmp_path,
            superancillary=False,
            scaffold=True,
        )
        assert [p.name for p in tmp_path.glob("*.json")] == ["scaffoldtest.json"]

        block = json.loads((tmp_path / "scaffoldtest.json").read_text())["viscosity"]
        assert block["higher_order"] == {
            "type": "custom",
            "name": "scaffoldtest_higher_order",
            "params": {},
        }
        assert block["dilute"]["type"] != "custom"
        assert "Not verified against CoolProp" in block["reference"]

    def test_without_the_flag_the_property_is_dropped_instead(self, tmp_path):
        make_fluid(
            "Hydrogen",
            name="plaintest",
            out_dir=tmp_path,
            superancillary=False,
            scaffold=False,
        )
        assert (
            json.loads((tmp_path / "plaintest.json").read_text())["viscosity"] is None
        )


def test_shipped_fluids_carry_no_unfilled_slot():
    for path in DATA_DIR.glob("*.json"):
        raw = json.loads(path.read_text())
        for prop in ("viscosity", "conductivity"):
            block = raw.get(prop) or {}
            customs = [
                k
                for k, v in block.items()
                if isinstance(v, dict) and v.get("type") == "custom"
            ]
            assert not customs, f"{path.name}: {prop} has unfilled {customs}"
