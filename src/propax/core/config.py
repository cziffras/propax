import os
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import List, Literal, Mapping, Optional

from pydantic import BaseModel, Field, model_validator


def get_table_path() -> Path:
    override = os.environ.get("PROPAX_TABLE_DIR")
    if override:
        return Path(override)
    home = Path.home() / ".cache" / "propax"
    home.mkdir(parents=True, exist_ok=True)
    return home


def get_chunk_size() -> Optional[int]:
    """How many nodes a mapped build call may hold at once, None for no cap

    Use with :

    `python -m propax.build_tables --chunk-size`
    """
    raw = os.environ.get("PROPAX_CHUNK_SIZE")
    if not raw:
        return None
    n = int(raw)
    if n < 1:
        raise ValueError(f"PROPAX_CHUNK_SIZE must be at least 1, got {n}")
    return n


@dataclass(frozen=True, slots=True)
class VarSpec:
    """What the flash needs to know about one property.

    The CoolProp key is the enum member's own value, so it is not repeated here.
    """

    internal_key: str
    can_be_input: bool = True
    linear_in_x: bool = False
    invert_for_mixing: bool = False
    sat_key_L: Optional[str] = None
    sat_key_V: Optional[str] = None
    sat_key_unique: Optional[str] = None
    scale: float = 1.0
    unit: str = ""


class ThermoVar(StrEnum):
    """Canonical names for thermodynamic properties, for library internal use.

    The member's value is the CoolProp key; everything else the flash needs
    hangs off `spec`.
    """

    P = "P"
    T = "T"
    H = "H"
    S = "S"
    U = "U"
    D = "D"
    Q = "Q"
    CVMASS = "CVMASS"
    CPMASS = "CPMASS"
    VISCOSITY = "VISCOSITY"
    CONDUCTIVITY = "CONDUCTIVITY"

    @property
    def spec(self) -> VarSpec:
        return _SPECS[self]

    @property
    def internal_key(self) -> str:
        return _SPECS[self].internal_key


_SPECS: Mapping[ThermoVar, VarSpec] = MappingProxyType(
    {
        ThermoVar.P: VarSpec("P", sat_key_unique="P_sat", scale=1e6, unit="Pa"),
        ThermoVar.T: VarSpec("T", sat_key_unique="T_sat", scale=100.0, unit="K"),
        ThermoVar.H: VarSpec(
            "h",
            sat_key_L="h_L",
            sat_key_V="h_V",
            linear_in_x=True,
            scale=1e5,
            unit="H",
        ),
        ThermoVar.S: VarSpec(
            "s",
            sat_key_L="s_L",
            sat_key_V="s_V",
            linear_in_x=True,
            scale=1e3,
            unit="J/K",
        ),
        ThermoVar.U: VarSpec(
            "u",
            sat_key_L="u_L",
            sat_key_V="u_V",
            linear_in_x=True,
            scale=1e5,
            unit="J",
        ),
        ThermoVar.D: VarSpec(
            "rho",
            sat_key_L="rho_L",
            sat_key_V="rho_V",
            invert_for_mixing=True,
            scale=10.0,
            unit="kg/m3",
        ),
        # An input, but only beside P or T `dispatch.SATURATED` says where.
        # Any other partner would need a solve, and a quality is not a state
        # variable of the EOS, so there is no bracket to lay one on.
        ThermoVar.Q: VarSpec("x"),
        ThermoVar.CVMASS: VarSpec("cv", can_be_input=False, unit="J/(K.kg)"),
        ThermoVar.CPMASS: VarSpec("cp", can_be_input=False, unit="J/(K.kg)"),
        ThermoVar.VISCOSITY: VarSpec("viscosity", can_be_input=False, unit="Pa/s"),
        ThermoVar.CONDUCTIVITY: VarSpec("conductivity", can_be_input=False, unit="S/m"),
    }
)


class PhaseID(IntEnum):
    UNKNOWN = -1
    TWO_PHASE = 0
    LIQUID = 1
    GAS = 2
    SUPERCRITICAL_GAS = 3
    SUPERCRITICAL = 4


class Axis(BaseModel):
    variable: ThermoVar
    min_val: float
    max_val: float
    spacing: Literal["linear", "log"] = "linear"
    n_points: int = 1000

    @property
    def key(self) -> str:
        return str(self.variable.value)


class TableSpec(BaseModel):
    """`outputs` is whichever of the natural variables (D, T) the axes do not
    already give; a dome table appends the quality, since below the saturation
    line (D, T) does not name the state on its own. Every other property is
    recovered from the EOS at that (rho, T) when it is asked for, never stored:
    a stored value is one more thing to keep consistent, and interpolating it
    is strictly worse than evaluating the EOS the table just located.
    """

    x_axis: Axis
    y_axis: Axis
    outputs: List[ThermoVar] = Field(default_factory=list)

    @model_validator(mode="after")
    def _outputs_complete_the_flash(self):
        axes = {self.x_axis.variable, self.y_axis.variable}
        expected = {ThermoVar.D, ThermoVar.T} - axes
        if set(self.outputs) != expected:
            raise ValueError(
                f"{self.name}: outputs must be exactly {sorted(v.value for v in expected)}, "
                f"got {sorted(v.value for v in self.outputs)}. A table stores the flash "
                "solution (D, T, and the quality under the dome); everything else comes "
                "from the EOS."
            )
        return self

    @property
    def name(self) -> str:
        return "_".join(("Table", self.x_axis.key, self.y_axis.key))


def _table(x: Axis, y: Axis) -> TableSpec:
    axes = {x.variable, y.variable}
    return TableSpec(
        x_axis=x, y_axis=y, outputs=sorted({ThermoVar.D, ThermoVar.T} - axes)
    )


_DENSITY = Axis(variable=ThermoVar.D, min_val=0.5, max_val=75.0, spacing="log")
_TEMPERATURE = Axis(variable=ThermoVar.T, min_val=15.0, max_val=150.0)
_PRESSURE = Axis(variable=ThermoVar.P, min_val=1e4, max_val=1.5e6, spacing="log")
_ENERGY = Axis(variable=ThermoVar.U, min_val=-5e4, max_val=1e6)
_ENTHALPY = Axis(variable=ThermoVar.H, min_val=-5e4, max_val=1.5e6)
_ENTROPY = Axis(variable=ThermoVar.S, min_val=-1e4, max_val=5e4)

TABLE_REGISTRY: List[TableSpec] = [
    _table(_DENSITY, _ENERGY),
    _table(_TEMPERATURE, _PRESSURE),
    _table(_PRESSURE, _ENTHALPY),
    _table(_PRESSURE, _ENTROPY),
    _table(_PRESSURE, _ENERGY),
    _table(_DENSITY, _ENTHALPY),
    _table(_DENSITY, _ENTROPY),
    _table(_TEMPERATURE, _ENTROPY),
]
