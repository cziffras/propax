"""
Schema and parser for fluid definition files.

A fluid definition file is a JSON document holding every coefficient needed to
evaluate, for one pure fluid: a multiparameter Helmholtz EOS (ideal + residual)
with saturation ancillaries, and optionally a viscosity and a thermal conductivity
correlation.

Each per-domain schema lives beside its runtime modules in `generic/`, and this
module composes them :

  - `generic.eos.schema_eos`          : Helmholtz EOS blocks + definition
  - `generic.viscosity.schema_visc`   : viscosity families (+ the shared `custom` term)
  - `generic.conductivity.schema_cond`: conductivity families

A new family is a new pydantic block in the relevant domain module (added to its
family-agnostic union), a matching eqx.Module in `propax.fluids.generic`, and a
registry entry in `_registry.py`.

Transport is optional: a fluid with neither block is still fully usable for EOS
properties (density, energies, cp/cv).
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .generic.conductivity.schema_cond import (
    ConductivityDefinition,
    ConductivityDiluteBlock,
    ConductivityResidualBlock,
    Eta0AndPolyDiluteBlock,
    OlchowySengersCriticalEnhancement,
    PolynomialExponentialResidualBlock,
    PolynomialResidualBlock,
    RationalPolynomialConductivityDefinition,
    RatioOfPolynomialsDiluteBlock,
    SimplifiedOlchowySengersBlock,
    SlotComposedConductivityDefinition,
)
from .generic.eos.schema_eos import (
    HelmholtzEOSDefinition,
    IdealHelmholtz,
    ResidualHelmholtz,
    SaturationSuperancillary,
)
from .generic.viscosity.schema_visc import (
    CollisionIntegralDiluteBlock,
    CustomTransportTermBlock,
    DiluteRainwaterFriendViscosityDefinition,
    ModifiedBatschinskiHildebrandBlock,
    PowersOfTDiluteBlock,
    PowersOfTrDiluteBlock,
    RainwaterFriendInitialDensityBlock,
    SlotComposedViscosityDefinition,
    ViscosityDefinition,
    ViscosityDiluteBlock,
    ViscosityHigherOrderBlock,
)

# Default dir for storing fluid json files (they go with the package)
DATA_DIR = Path(__file__).parent / "data"

# NOTE : EOS currently has a single family for now, the alias will have to be changed
EOSDefinition = HelmholtzEOSDefinition


class FluidDefinition(BaseModel):
    name: str
    cas: Optional[str] = None
    eos: EOSDefinition
    # Transport is optional (CoolProp ships no correlations in most cases)
    viscosity: Optional[ViscosityDefinition] = None
    conductivity: Optional[ConductivityDefinition] = None


def load_fluid_file(path: Path | str) -> FluidDefinition:
    path = Path(path)
    with open(path) as fh:
        raw = json.load(fh)
    return FluidDefinition.model_validate(raw)


def discover_fluid_files(data_dir: Path | str = DATA_DIR) -> dict[str, Path]:
    """Map normalized fluid name -> definition file for every JSON file in data_dir"""
    data_dir = Path(data_dir)
    found = {}
    for path in sorted(data_dir.glob("*.json")):
        with open(path) as fh:
            name = json.load(fh).get("name", path.stem)
        found[name.lower().replace(" ", "")] = path
    return found


__all__ = [
    "DATA_DIR",
    "EOSDefinition",
    "FluidDefinition",
    "load_fluid_file",
    "discover_fluid_files",
    # schema_eos
    "SaturationSuperancillary",
    "HelmholtzEOSDefinition",
    "IdealHelmholtz",
    "ResidualHelmholtz",
    # schema_visc
    "CollisionIntegralDiluteBlock",
    "CustomTransportTermBlock",
    "DiluteRainwaterFriendViscosityDefinition",
    "ModifiedBatschinskiHildebrandBlock",
    "PowersOfTDiluteBlock",
    "PowersOfTrDiluteBlock",
    "RainwaterFriendInitialDensityBlock",
    "SlotComposedViscosityDefinition",
    "ViscosityDefinition",
    "ViscosityDiluteBlock",
    "ViscosityHigherOrderBlock",
    # schema_cond
    "ConductivityDefinition",
    "ConductivityDiluteBlock",
    "ConductivityResidualBlock",
    "Eta0AndPolyDiluteBlock",
    "OlchowySengersCriticalEnhancement",
    "PolynomialExponentialResidualBlock",
    "PolynomialResidualBlock",
    "RationalPolynomialConductivityDefinition",
    "RatioOfPolynomialsDiluteBlock",
    "SimplifiedOlchowySengersBlock",
    "SlotComposedConductivityDefinition",
]
