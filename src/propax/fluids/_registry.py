"""
Fluid registry built from definition files

Every JSON file in `propax/fluids/data/` (or in the directory pointed to by
the PROPAX_FLUID_DIR environment variable) is parsed and validated at import
time and exposed in EQS_REGISTRY as a triple of factories:

    eos_factory()
    superancillary_factory()
    viscosity_factory()
    conductivity_factory(eos=..., viscosity=...)

which is the calling convention `Interface.create` relies on
"""

import os
from pathlib import Path
from typing import Callable, Dict, Tuple

from .generic import (
    ConductivitySlots,
    DiluteRainwaterFriendViscosity,
    HelmholtzEOS,
    RationalPolynomialConductivity,
    ViscositySlots,
)
from .schema import DATA_DIR, FluidDefinition, discover_fluid_files, load_fluid_file

# one eqx.Module class per supported correlation family (see schema.py)
_VISCOSITY_MODELS = {
    "dilute_rainwater_friend": DiluteRainwaterFriendViscosity,
    "slot_composed": ViscositySlots,
}
_CONDUCTIVITY_MODELS = {
    "rational_polynomial_critical": RationalPolynomialConductivity,
    "slot_composed": ConductivitySlots,
}


def _make_factories(
    defn: FluidDefinition,
) -> Tuple[Callable, Callable, Callable, Callable]:
    # transport blocks are optional (for now); the factories then yield None
    visc_cls = _VISCOSITY_MODELS[defn.viscosity.model] if defn.viscosity else None
    cond_cls = (
        _CONDUCTIVITY_MODELS[defn.conductivity.model] if defn.conductivity else None
    )

    def eos_factory():
        return HelmholtzEOS.from_definition(defn.eos)

    def superancillary_factory():
        return defn.eos.superancillary

    def viscosity_factory(*, eos=None):
        if visc_cls is None or defn.viscosity is None:
            return None
        # slot_composed can carry a custom (user-supplied) higher-order term that
        # needs the EOS : provided by the user /!\
        if defn.viscosity.model == "slot_composed":
            return visc_cls.from_definition(defn.viscosity, eos=eos)
        return visc_cls.from_definition(defn.viscosity)

    def conductivity_factory(*, eos, viscosity):
        # the critical enhancement needs the viscosity, so no viscosity means
        # no conductivity either
        if cond_cls is None or viscosity is None:
            return None
        return cond_cls.from_definition(defn.conductivity, eos=eos, viscosity=viscosity)

    return eos_factory, superancillary_factory, viscosity_factory, conductivity_factory


def _build_registry() -> Dict[str, Tuple[Callable, Callable, Callable, Callable]]:
    data_dir = Path(os.environ.get("PROPAX_FLUID_DIR", DATA_DIR))
    registry = {}
    for name, path in discover_fluid_files(data_dir).items():
        registry[name] = _make_factories(load_fluid_file(path))
    return registry


EQS_REGISTRY: Dict[str, Tuple[Callable, Callable, Callable, Callable]] = (
    _build_registry()
)
