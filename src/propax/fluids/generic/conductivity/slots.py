"""
Slot-composed thermal conductivity :

Same idea as `generic.viscosity.slots`: a correlation is a sum of density
regimes, each picking a form from a menu, and CoolProp mixes them freely.

    lambda(rho, T) = lambda0(T) + delta_lambda(rho, T) + delta_lambda_c(rho, T)
                     dilute       residual               critical enhancement

Unlike viscosity, these slots are coupled to other models: the dilute term of
the Lemmon-Jacobsen family is written in terms of the dilute viscosity, and the
critical enhancement needs cp, cv, dP/drho from the EOS and the full viscosity.
`ConductivitySlots` therefore receives `eos` and `viscosity`, as the existing
family already does.

/!\\ Every form's equation and its CoolProp source are documented in docs/transport.md
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from propax.utils.types import jaxFloat

from ..transport_registry import build_conductivity_residual

KB = 1.380649e-23  # J/K


class DiluteConductivityTerm(eqx.Module):
    # Whether lambda0 rides on the dilute viscosity eta0(T); when False the
    # conductivity loop skips fetching it (so the form works with any viscosity
    # model, or none of the eta0-exposing kind).
    needs_eta0 = False

    def lambda0(self, T: jaxFloat, eta0_uPas: jaxFloat) -> jaxFloat:
        raise NotImplementedError


class ResidualConductivityTerm(eqx.Module):
    def contribution(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        raise NotImplementedError


class Eta0AndPolyDilute(DiluteConductivityTerm):
    """CoolProp `dilute:eta0_and_poly` (Lemmon-Jacobsen) equation and CoolProp
    source in docs/transport.md, "Conductivity - dilute"

    The leading term rides on the dilute viscosity (needs_eta0 = True), so this
    family cannot be evaluated without a viscosity model
    """

    needs_eta0 = True

    T_c: float
    A: np.ndarray
    t: np.ndarray

    def lambda0(self, T: jaxFloat, eta0_uPas: jaxFloat) -> jaxFloat:
        tau = self.T_c / T
        poly = jnp.sum(self.A[1:] * tau ** self.t[1:])
        return self.A[0] * eta0_uPas + poly


class RatioOfPolynomialsDilute(DiluteConductivityTerm):
    """CoolProp `dilute:ratio_of_polynomials` (Assael) equation and CoolProp
    source in docs/transport.md, "Conductivity - dilute"

    A pure temperature ratio, so it needs no viscosity model (needs_eta0 = False)
    """

    needs_eta0 = False

    T_reducing: float
    A: np.ndarray  # numerator coefficients
    n: np.ndarray  # numerator exponents
    B: np.ndarray  # denominator coefficients
    m: np.ndarray  # denominator exponents

    def lambda0(self, T: jaxFloat, eta0_uPas: jaxFloat) -> jaxFloat:
        Tr = T / self.T_reducing
        num = jnp.sum(self.A * Tr**self.n)
        den = jnp.sum(self.B * Tr**self.m)
        return num / den


class PolynomialExponentialResidual(ResidualConductivityTerm):
    """CoolProp `residual:polynomial_and_exponential` equation and CoolProp
    source in docs/transport.md, "Conductivity - residual"
    """

    T_c: float
    rho_c: float  # kg/m3
    A: np.ndarray
    d: np.ndarray
    t: np.ndarray
    gamma: np.ndarray
    l_exp: np.ndarray

    def contribution(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        tau = self.T_c / T
        delta = rho / self.rho_c
        return jnp.sum(
            self.A
            * tau**self.t
            * delta**self.d
            * jnp.exp(-self.gamma * delta**self.l_exp)
        )


class PolynomialResidual(ResidualConductivityTerm):
    """CoolProp `residual:polynomial` equation and CoolProp source in
    docs/transport.md, "Conductivity - residual"

    Exponential-free, reduced by the transport correlation's own T/rho,
    with the reciprocal tau = T_reducing / T (verified against CoolProp)
    """

    T_reducing: float
    rho_reducing: float  # kg/m3
    B: np.ndarray
    d: np.ndarray
    t: np.ndarray

    def contribution(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        tau = self.T_reducing / T
        delta = rho / self.rho_reducing
        return jnp.sum(self.B * delta**self.d * tau**self.t)


class SimplifiedOlchowySengers(eqx.Module):
    """CoolProp `critical:simplified_Olchowy_Sengers` equation and CoolProp source
    in docs/transport.md, "Conductivity - critical enhancement"

    Needs cp, cv and (dP/drho)_T from the EOS plus the viscosity, hence the injected
    modules; T_ref is per-fluid
    """

    T_c: float
    rho_c: float  # kg/m3
    p_c: float  # MPa
    T_ref: float  # K
    nu: float
    gamma: float
    Gamma: float
    zeta0: float  # m
    R0: float
    qD: float  # 1/m

    def _big_chi(self, eos, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        dP_drho_T = jax.grad(lambda r: eos.props_rhoT(r, T)["P"] / 1e6)(rho)
        dP_drho_ref = jax.grad(lambda r: eos.props_rhoT(r, self.T_ref)["P"] / 1e6)(rho)
        delta_grads = 1.0 / dP_drho_T - (self.T_ref / T) / dP_drho_ref
        delta_grads = jnp.maximum(delta_grads, 1e-15)
        K = self.zeta0 * (
            (self.p_c * rho) / (self.Gamma * (T / self.T_c) * self.rho_c**2)
        ) ** (self.nu / self.gamma)
        return K * delta_grads ** (self.nu / self.gamma)

    def contribution(self, eos, viscosity, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        props = eos.props_rhoT(rho, T)
        cp, cv = props["cp"], props["cv"]
        eta = viscosity.viscosity_rhoT(rho, T)
        xi = self._big_chi(eos, rho, T)

        qD_xi = xi * self.qD
        omega = (2.0 / jnp.pi) * (
            (1.0 - cv / cp) * jnp.arctan(qD_xi) + (cv / cp) * qD_xi
        )
        qD_xi_safe = jnp.maximum(qD_xi, 1e-15)
        denom = 1.0 / qD_xi_safe + ((qD_xi * self.rho_c / rho) ** 2) / 3.0
        omega_0 = (2.0 / jnp.pi) * (1.0 - jnp.exp(-1.0 / denom))

        K = (rho * cp * self.R0 * KB * T) / (6.0 * jnp.pi * xi * eta)
        return K * (omega - omega_0)


class ConductivitySlots(eqx.Module):
    eos: eqx.Module
    viscosity: eqx.Module
    dilute: DiluteConductivityTerm
    residual: object = None  # Optional[ResidualConductivityTerm]
    critical: object = None  # Optional[SimplifiedOlchowySengers]
    reference: str = eqx.field(static=True, default="")

    @classmethod
    def from_definition(cls, defn, *, eos, viscosity) -> "ConductivitySlots":
        d = defn.dilute
        if d.type == "ratio_of_polynomials":
            dilute = RatioOfPolynomialsDilute(
                T_reducing=d.T_reducing,
                A=np.asarray(d.A),
                n=np.asarray(d.n),
                B=np.asarray(d.B),
                m=np.asarray(d.m),
            )
        else:  # eta0_and_poly
            dilute = Eta0AndPolyDilute(T_c=d.T_c, A=np.asarray(d.A), t=np.asarray(d.t))
        residual = None
        if defn.residual is not None:
            r = defn.residual
            if r.type == "custom":
                residual = build_conductivity_residual(
                    r.name, r.params, eos=eos, viscosity=viscosity
                )
            elif r.type == "polynomial":
                residual = PolynomialResidual(
                    T_reducing=r.T_reducing,
                    rho_reducing=r.rho_reducing,
                    B=np.asarray(r.B),
                    d=np.asarray(r.d),
                    t=np.asarray(r.t),
                )
            else:  # polynomial_and_exponential
                residual = PolynomialExponentialResidual(
                    T_c=r.T_c,
                    rho_c=r.rho_c,
                    A=np.asarray(r.A),
                    d=np.asarray(r.d),
                    t=np.asarray(r.t),
                    gamma=np.asarray(r.gamma),
                    l_exp=np.asarray(r.l_exp),
                )
        critical = None
        if defn.critical is not None:
            c = defn.critical
            critical = SimplifiedOlchowySengers(
                T_c=c.T_c,
                rho_c=c.rho_c,
                p_c=c.p_c,
                T_ref=c.T_ref,
                nu=c.nu,
                gamma=c.gamma,
                Gamma=c.Gamma,
                zeta0=c.zeta0,
                R0=c.R0,
                qD=c.qD,
            )
        return cls(
            eos=eos,
            viscosity=viscosity,
            dilute=dilute,
            residual=residual,
            critical=critical,
            reference=defn.reference,
        )

    @eqx.filter_jit
    def conductivity_rhoT(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        """Thermal conductivity [W/(m K)] from mass density [kg/m3] and T [K]"""
        # some dilute forms (eta0_and_poly) ride on the dilute viscosity, in
        # uPa.s but ratio_of_polynomials does not, so only fetch it when needed
        eta0_uPas = (
            self.viscosity.dilute.eta0(T) * 1e6  # type: ignore
            if self.dilute.needs_eta0
            else 0.0
        )
        total = self.dilute.lambda0(T, eta0_uPas)
        if self.residual is not None:
            total = total + self.residual.contribution(rho, T)  # type: ignore
        if self.critical is not None:
            total = total + self.critical.contribution(  # type: ignore
                self.eos, self.viscosity, rho, T
            )
        return total
