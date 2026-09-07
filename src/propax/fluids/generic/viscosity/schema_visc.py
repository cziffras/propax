"""
Viscosity schema.

Two families, discriminated on the `model` tag:
  - "dilute_rainwater_friend": the fitted Muzny-style whole-correlation;
  - "slot_composed": one form per density regime (dilute / initial-density /
    higher-order), mirroring how CoolProp stores transport.

`CustomTransportTermBlock` (a user-supplied registered JAX term) lives here
because it is the shared `custom` slot form; `schema_cond` imports it too.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "DiluteRainwaterFriendViscosityDefinition",
    "CollisionIntegralDiluteBlock",
    "PowersOfTDiluteBlock",
    "PowersOfTrDiluteBlock",
    "RainwaterFriendInitialDensityBlock",
    "ModifiedBatschinskiHildebrandBlock",
    "CustomTransportTermBlock",
    "ViscosityDiluteBlock",
    "ViscosityHigherOrderBlock",
    "SlotComposedViscosityDefinition",
    "ViscosityDefinition",
]


class DiluteRainwaterFriendViscosityDefinition(BaseModel):
    """
    "dilute_rainwater_friend" (Muzny et al. style):
      eta = eta0(T) + eta1(T) rho + eta_residual(rho, T)   [returned in Pa.s]
      eta0: Chapman-Enskog with ln(S*) = sum a_i ln(T*)^i
      eta1: B*_eta = sum b_i (T*)^-i (Rainwater-Friend)
      residual: c0 rho_r^2 exp(c1 Tr + c2/Tr + c3 rho_r^2/(c4 + Tr) + c5 rho_r^6)
    """

    model: Literal["dilute_rainwater_friend"]
    reference: str = ""
    rho_sc: float  # scaling density [kg/m3]
    T_c: float  # K (correlation's own reducing temperature)
    M: float  # [g/mol]
    sigma: float  # collision diameter [nm]
    eps_kb: float  # K, Lennard-Jones energy / k_B
    a: List[float]  # dilute-gas ln(S*) coefficients
    b: List[float]  # Rainwater-Friend B* coefficients
    c: List[float]  # residual coefficients (6 values)

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.c) != 6:
            raise ValueError("viscosity residual coefficients 'c' must have 6 entries")
        return self


class CollisionIntegralDiluteBlock(BaseModel):
    """eta0 = C sqrt(M T) / (sigma^2 Omega), Omega = exp(sum a_i (ln T*)^t_i)"""

    type: Literal["collision_integral"]
    C: float
    M: float  # g/mol
    sigma: float  # nm
    eps_kb: float  # K
    a: List[float]
    t: List[float]

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.a) != len(self.t):
            raise ValueError("dilute 'a' and 't' must have the same length")
        return self


class PowersOfTDiluteBlock(BaseModel):
    """eta0 = sum a_i T^t_i  [Pa.s]"""

    type: Literal["powers_of_T"]
    a: List[float]
    t: List[float]

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.a) != len(self.t):
            raise ValueError("dilute 'a' and 't' must have the same length")
        return self


class PowersOfTrDiluteBlock(BaseModel):
    """eta0 = sum a_i (T/T_reducing)^t_i  [Pa.s]"""

    type: Literal["powers_of_Tr"]
    T_reducing: float  # K
    a: List[float]
    t: List[float]

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.a) != len(self.t):
            raise ValueError("dilute 'a' and 't' must have the same length")
        return self


class RainwaterFriendInitialDensityBlock(BaseModel):
    """B*(T*) = sum b_i (T*)^t_i, the reduced second viscosity virial coefficient"""

    type: Literal["rainwater_friend"]
    sigma: float  # nm
    eps_kb: float  # K
    b: List[float]
    t: List[float]

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.b) != len(self.t):
            raise ValueError("initial-density 'b' and 't' must have the same length")
        return self


class ModifiedBatschinskiHildebrandBlock(BaseModel):
    """Polynomial/exponential sum plus the optional free-volume pole (f = 0 off)S"""

    type: Literal["modified_batschinski_hildebrand"]
    T_reduce: float  # K
    rho_reduce: float  # mol/m3
    a: List[float]
    d1: List[float]
    t1: List[float]
    gamma: List[float]
    l_exp: List[float]
    f: float = 0.0
    g: List[float] = Field(default_factory=lambda: [1.0])
    h: List[float] = Field(default_factory=lambda: [0.0])
    p: List[float] = Field(default_factory=lambda: [1.0])
    q: List[float] = Field(default_factory=lambda: [0.0])
    d2: float = 0.0
    t2: float = 0.0

    @model_validator(mode="after")
    def _check_lengths(self):
        n = {len(self.a), len(self.d1), len(self.t1), len(self.gamma), len(self.l_exp)}
        if len(n) > 1:
            raise ValueError("higher-order 'a/d1/t1/gamma/l_exp' must be equal length")
        if len(self.g) != len(self.h):
            raise ValueError("higher-order 'g' and 'h' must have the same length")
        if len(self.p) != len(self.q):
            raise ValueError("higher-order 'p' and 'q' must have the same length")
        return self


class CustomTransportTermBlock(BaseModel):
    """A transport slot supplied by the user as a registered JAX term.

    For a literature/hardcoded correlation propax has no built-in class for:
    `name` keys into the transport term registry (see `generic.transport_registry`),
    `params` is passed by the user to the registered builder.
    """

    type: Literal["custom"]
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


ViscosityDiluteBlock = Annotated[
    Union[
        CollisionIntegralDiluteBlock,
        PowersOfTDiluteBlock,
        PowersOfTrDiluteBlock,
    ],
    Field(discriminator="type"),
]

ViscosityHigherOrderBlock = Annotated[
    Union[ModifiedBatschinskiHildebrandBlock, CustomTransportTermBlock],
    Field(discriminator="type"),
]


class SlotComposedViscosityDefinition(BaseModel):
    """
    "slot_composed": eta = eta0(T) + eta1(T) rho + delta_eta(rho, T), each slot
    naming its own form
    """

    model: Literal["slot_composed"]
    reference: str = ""
    molar_mass: float  # kg/mol
    dilute: ViscosityDiluteBlock
    initial_density: Optional[RainwaterFriendInitialDensityBlock] = None
    higher_order: Optional[ViscosityHigherOrderBlock] = None


ViscosityDefinition = Annotated[
    Union[DiluteRainwaterFriendViscosityDefinition, SlotComposedViscosityDefinition],
    Field(discriminator="model"),
]
