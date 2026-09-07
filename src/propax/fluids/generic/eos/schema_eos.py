"""
Multiparameter Helmholtz EOS schema: the ideal and residual Helmholtz parts, the
saturation superancillary, and the fluid constants.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "IdealHelmholtz",
    "ResidualHelmholtz",
    "SaturationSuperancillary",
    "HelmholtzEOSDefinition",
]


class IdealHelmholtz(BaseModel):
    """alpha0 = ln(delta) + a_log ln(tau) + a1 + a2 tau + sum n_i ln(1 - exp(-v_i tau / T_c))

    a_log is c_v0/R of the low-T limit: mostly specified in related papers
    """

    a1: float
    a2: float
    a_log: float
    pe_n: List[float] = Field(default_factory=list)
    pe_v: List[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.pe_n) != len(self.pe_v):
            raise ValueError("pe_n and pe_v must have the same length")
        return self


class ResidualHelmholtz(BaseModel):
    """
    alphar = sum n d^di t^ti                          (polynomial terms)
           + sum n d^di t^ti exp(-d^pi)               (exponential terms)
           + sum n d^di t^ti exp(-g d^li)             (generalized exponential)
           + sum n d^di t^ti exp(phi (d-D)^2 + beta (t-gamma)^2)   (Gaussian terms)
    """

    poly_n: List[float] = Field(default_factory=list)
    poly_t: List[float] = Field(default_factory=list)
    poly_d: List[float] = Field(default_factory=list)

    exp_n: List[float] = Field(default_factory=list)
    exp_t: List[float] = Field(default_factory=list)
    exp_d: List[float] = Field(default_factory=list)
    exp_p: List[float] = Field(default_factory=list)

    # generalized exponential: like exp_* but with an explicit coefficient g in
    # the exponential (CoolProp ResidualHelmholtzExponential, where g != 1)
    gexp_n: List[float] = Field(default_factory=list)
    gexp_t: List[float] = Field(default_factory=list)
    gexp_d: List[float] = Field(default_factory=list)
    gexp_g: List[float] = Field(default_factory=list)
    gexp_p: List[float] = Field(default_factory=list)

    gauss_n: List[float] = Field(default_factory=list)
    gauss_t: List[float] = Field(default_factory=list)
    gauss_d: List[float] = Field(default_factory=list)
    gauss_phi: List[float] = Field(default_factory=list)
    gauss_beta: List[float] = Field(default_factory=list)
    gauss_gamma: List[float] = Field(default_factory=list)
    gauss_D: List[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_lengths(self):
        groups = {
            "poly": [self.poly_n, self.poly_t, self.poly_d],
            "exp": [self.exp_n, self.exp_t, self.exp_d, self.exp_p],
            "gexp": [
                self.gexp_n,
                self.gexp_t,
                self.gexp_d,
                self.gexp_g,
                self.gexp_p,
            ],
            "gauss": [
                self.gauss_n,
                self.gauss_t,
                self.gauss_d,
                self.gauss_phi,
                self.gauss_beta,
                self.gauss_gamma,
                self.gauss_D,
            ],
        }
        for name, arrays in groups.items():
            if len({len(a) for a in arrays}) > 1:
                raise ValueError(
                    f"All '{name}_*' coefficient arrays must have the same length"
                )
        return self


class SaturationSuperancillary(BaseModel):
    T_min: float  # K, the cold end the fit covers
    T_max: float  # K

    rho_max_mol: float
    """mol/m3 at (T_min, P_max): the densest state the correlation claims.

    It lives here rather than on the EOS because it is only defined against
    the liquid branch -- below that branch the isotherm is the dome's own
    loop, and P = P_max has roots there that mean nothing."""

    edges: List[float]
    """Piece boundaries in s, (n_pieces + 1,) ascending."""

    coeffs: List[List[List[float]]]
    """(n_pieces, degree + 1, n_components), the Chebyshev coefficients."""

    cuts: List[List[float]]
    """(n_components, n_cuts) where each component turns over, so it inverts."""

    derived_cuts: Dict[str, List[float]]
    """The same, for each quantity read from the EOS rather than stored."""

    log_components: List[int] = Field(default_factory=list)
    """Components fitted on their logarithm, exponentiated on the way out."""

    @model_validator(mode="after")
    def _check_shapes(self):
        if len(self.edges) != len(self.coeffs) + 1:
            raise ValueError("edges must have one more entry than coeffs has pieces")
        if any(b <= a for a, b in zip(self.edges, self.edges[1:])):
            raise ValueError("edges must be strictly ascending")
        shapes = {(len(p), len(p[0])) for p in self.coeffs}
        if len(shapes) != 1:
            raise ValueError("every piece must carry the same (degree, component) grid")
        (_, n_components) = shapes.pop()
        if any(len(row) != n_components for p in self.coeffs for row in p):
            raise ValueError("ragged coefficient rows")
        if len(self.cuts) != n_components:
            raise ValueError("cuts must carry one row per component")
        if len({len(row) for row in self.cuts}) != 1:
            raise ValueError("cuts rows must be padded to a common length")
        if any(not 0 <= c < n_components for c in self.log_components):
            raise ValueError("log_components must index a component")
        return self


class HelmholtzEOSDefinition(BaseModel):
    reference: str = ""
    R_u: float = 8.314462618
    molar_mass: float
    T_red: float
    P_red: float
    rho_red_mol: float
    # The correlation's own critical point, where dP/drho and d2P/drho2 zero out,
    # solved by `utils.exact.critical` with mpmath
    T_crit: float
    P_crit: float
    rho_crit_mol: float
    T_triple: float
    P_triple: float
    # Upper ends of the EOS's stated validity range (CoolProp EOS.T_max, p_max)
    T_max: Optional[float] = None  # K
    P_max: float  # Pa
    # The density at (T_triple, P_max) solved during make
    ideal: IdealHelmholtz
    residual: ResidualHelmholtz
    superancillary: Optional[SaturationSuperancillary] = None
