from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from ..viscosity.schema_visc import CustomTransportTermBlock

__all__ = [
    "OlchowySengersCriticalEnhancement",
    "RationalPolynomialConductivityDefinition",
    "Eta0AndPolyDiluteBlock",
    "RatioOfPolynomialsDiluteBlock",
    "PolynomialExponentialResidualBlock",
    "PolynomialResidualBlock",
    "SimplifiedOlchowySengersBlock",
    "ConductivityDiluteBlock",
    "ConductivityResidualBlock",
    "SlotComposedConductivityDefinition",
    "ConductivityDefinition",
]


class OlchowySengersCriticalEnhancement(BaseModel):
    nu: float = 0.63
    gamma: float = 1.2415
    Gamma: float = 0.052
    chi_0: float = 1.5e-10  # m
    RD: float = 1.01
    qd_inv: float  # m


class RationalPolynomialConductivityDefinition(BaseModel):
    """
    "rational_polynomial_critical" (https://doi.org/10.1063/1.3606499):

      lambda = lambda0(T) + delta_lambda(rho, T) + delta_lambda_c(rho, T)
      lambda0: sum(A1_i Tr^i) / sum(A2_i Tr^i)
      delta_lambda: sum_{i=1..n} (B1_i + B2_i Tr) rho_r^i
      delta_lambda_c: Olchowy-Sengers (needs EOS + viscosity)
    """

    model: Literal["rational_polynomial_critical"]
    reference: str = ""
    T_c: float  # K
    rho_c: float  # kg/m3
    p_c: float  # MPa (used in the crossover expression)
    A1: List[float]
    A2: List[float]
    B1: List[float]
    B2: List[float]
    critical: OlchowySengersCriticalEnhancement

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.B1) != len(self.B2):
            raise ValueError("B1 and B2 must have the same length")
        return self


class Eta0AndPolyDiluteBlock(BaseModel):
    type: Literal["eta0_and_poly"]
    T_c: float  # K
    A: List[float]
    t: List[float]

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.A) != len(self.t):
            raise ValueError("dilute 'A' and 't' must have the same length")
        return self


class RatioOfPolynomialsDiluteBlock(BaseModel):
    type: Literal["ratio_of_polynomials"]
    T_reducing: float  # K
    A: List[float]  # numerator coefficients (len == len(n))
    n: List[float]  # numerator exponents
    B: List[float]  # denominator coefficients (len == len(m))
    m: List[float]  # denominator exponents

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.A) != len(self.n):
            raise ValueError("dilute 'A' and 'n' must have the same length")
        if len(self.B) != len(self.m):
            raise ValueError("dilute 'B' and 'm' must have the same length")
        return self


class PolynomialExponentialResidualBlock(BaseModel):
    type: Literal["polynomial_and_exponential"]
    T_c: float  # K
    rho_c: float  # kg/m3
    A: List[float]
    d: List[float]
    t: List[float]
    gamma: List[float]
    l_exp: List[float]

    @model_validator(mode="after")
    def _check_lengths(self):
        if (
            len(
                {
                    len(self.A),
                    len(self.d),
                    len(self.t),
                    len(self.gamma),
                    len(self.l_exp),
                }
            )
            > 1
        ):
            raise ValueError("residual 'A/d/t/gamma/l_exp' must be equal length")
        return self


class PolynomialResidualBlock(BaseModel):
    """delta_lambda = sum B_i (rho/rho_reducing)^d_i (T_reducing/T)^t_i.

    The exponential-free residual (CoolProp `residual:polynomial`); the same
    family as polynomial_and_exponential with the damping absent, but reduced by
    the transport correlation's own T/rho rather than the EOS critical point"""

    type: Literal["polynomial"]
    T_reducing: float  # K
    rho_reducing: float  # kg/m3
    B: List[float]
    d: List[float]
    t: List[float]

    @model_validator(mode="after")
    def _check_lengths(self):
        if len({len(self.B), len(self.d), len(self.t)}) > 1:
            raise ValueError("residual 'B/d/t' must be equal length")
        return self


class SimplifiedOlchowySengersBlock(BaseModel):
    """TODO : write a short self-explanatory note on mode-coupling theory"""

    type: Literal["simplified_olchowy_sengers"]
    T_c: float  # K
    rho_c: float  # kg/m3
    p_c: float  # MPa
    T_ref: float  # K
    qD: float  # 1/m
    zeta0: float  # m
    Gamma: float
    R0: float = 1.01
    nu: float = 0.63
    gamma: float = 1.2415


ConductivityDiluteBlock = Annotated[
    Union[Eta0AndPolyDiluteBlock, RatioOfPolynomialsDiluteBlock],
    Field(discriminator="type"),
]
ConductivityResidualBlock = Annotated[
    Union[
        PolynomialExponentialResidualBlock,
        PolynomialResidualBlock,
        CustomTransportTermBlock,
    ],
    Field(discriminator="type"),
]


class SlotComposedConductivityDefinition(BaseModel):
    """
    "slot_composed": lambda = lambda0(T) + delta_lambda + delta_lambda_c, each
    slot naming its own form (mirrors CoolProp's own layout)
    """

    model: Literal["slot_composed"]
    reference: str = ""
    dilute: ConductivityDiluteBlock
    residual: Optional[ConductivityResidualBlock] = None
    critical: Optional[SimplifiedOlchowySengersBlock] = None


# alias
ConductivityDefinition = Annotated[
    Union[RationalPolynomialConductivityDefinition, SlotComposedConductivityDefinition],
    Field(discriminator="model"),
]
