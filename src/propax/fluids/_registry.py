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
    HelmholtzEOS,
    ViscositySlots,
)
from .generic.conductivity.schema_cond import (
    SlotComposedConductivityDefinition,
)
from .generic.viscosity.schema_visc import (
    SlotComposedViscosityDefinition,
)
from .schema import DATA_DIR, FluidDefinition, discover_fluid_files, load_fluid_file


def _make_factories(
    defn: FluidDefinition,
) -> Tuple[Callable, Callable, Callable, Callable]:
    # transport blocks are optional (for now); the factories then yield None
    visc_cls = (
        ViscositySlots
        if isinstance(defn.viscosity, SlotComposedViscosityDefinition)
        else None
    )
    cond_cls = (
        ConductivitySlots
        if isinstance(defn.conductivity, SlotComposedConductivityDefinition)
        else None
    )

    def eos_factory():
        return HelmholtzEOS.from_definition(defn.eos)

    def superancillary_factory():
        return defn.eos.superancillary

    def viscosity_factory(*, eos=None):
        if visc_cls is None or defn.viscosity is None:
            return None
        return visc_cls.from_definition(defn.viscosity, eos=eos)

    def conductivity_factory(*, eos, viscosity):
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


# NOTE : eager build of the registry --> a single invalid json makes it crash
# to go over this either regenerate a valid set of fluid files (through catalogue action)
# or point to a directory of valid json files with PROPAX_FLUID_DIR=/path/to/fluids
EQS_REGISTRY: Dict[str, Tuple[Callable, Callable, Callable, Callable]] = (
    _build_registry()
)
