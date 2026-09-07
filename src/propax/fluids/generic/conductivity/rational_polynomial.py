import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from propax.utils.types import jaxFloat

from .schema_cond import RationalPolynomialConductivityDefinition


class RationalPolynomialConductivity(eqx.Module):
    """
    Thermal conductivity correlation family "rational_polynomial_critical"
    formulated as :

        lambda(rho, T) = lambda0(T) + delta_lambda(rho, T) + delta_lambda_c(rho, T)

    with a rational-polynomial dilute-gas term, a polynomial excess term and an
    Olchowy-Sengers critical enhancement. The critical
    enhancement needs the EOS (for dP/drho and cp/cv) and the viscosity model.
    """

    eos: eqx.Module
    viscosity: eqx.Module

    T_c: float  # K
    rho_c: float  # kg/m3
    p_c: float  # MPa
    kb: float = eqx.field(default=1.380649e-23)  # Boltzmann constant J/K

    # Dilute gas rational polynomial coefficients
    A1: np.ndarray = eqx.field(default=None)
    A2: np.ndarray = eqx.field(default=None)

    # Excess conductivity coefficients (index 0 is padding)
    B1: np.ndarray = eqx.field(default=None)
    B2: np.ndarray = eqx.field(default=None)

    # Critical enhancement (Olchowy-Sengers crossover)
    nu: float = 0.63
    gamma: float = 1.2415
    Gamma: float = 0.052
    chi_0: float = 1.5e-10  # m
    RD: float = 1.01
    qd_inv: float = 5.0e-10  # qD^-1 in m

    # Empirical critical enhancement coefficients (None when the fluid's
    # correlation paper does not publish them; only the theoretical
    # Olchowy-Sengers enhancement is available then)

    reference: str = eqx.field(static=True, default="")

    @classmethod
    def from_definition(
        cls,
        defn: RationalPolynomialConductivityDefinition,
        *,
        eos: eqx.Module,
        viscosity: eqx.Module,
    ) -> "RationalPolynomialConductivity":
        crit = defn.critical
        return cls(
            eos=eos,
            viscosity=viscosity,
            T_c=defn.T_c,
            rho_c=defn.rho_c,
            p_c=defn.p_c,
            A1=np.asarray(defn.A1),
            A2=np.asarray(defn.A2),
            B1=np.asarray(defn.B1),
            B2=np.asarray(defn.B2),
            nu=crit.nu,
            gamma=crit.gamma,
            Gamma=crit.Gamma,
            chi_0=crit.chi_0,
            RD=crit.RD,
            qd_inv=crit.qd_inv,
            reference=defn.reference,
        )

    def _lambda_0(self, T: jaxFloat) -> jaxFloat:
        Tr = T / self.T_c
        num_exponents = jnp.arange(self.A1.shape[0])
        num = jnp.sum(self.A1 * (Tr**num_exponents))
        den_exponents = jnp.arange(self.A2.shape[0])
        den = jnp.sum(self.A2 * (Tr**den_exponents))
        return num / den

    def _delta_lambda(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        Tr = T / self.T_c
        rho_r = rho / self.rho_c
        i_indices = jnp.arange(1, self.B1.shape[0])
        terms = (self.B1[1:] + self.B2[1:] * Tr) * (rho_r**i_indices)
        return jnp.sum(terms)

    def _big_chi(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        T_ref = 1.5 * self.T_c

        dP_drho_T = jax.grad(lambda r: self.eos.props_rhoT(r, T)["P"] / 1e6)(rho)  # type: ignore
        dP_drho_Tref = jax.grad(
            lambda r: self.eos.props_rhoT(r, T_ref)["P"] / 1e6  # type: ignore
        )(rho)

        # Invert to get (d_rho / d_P)_T
        drho_dP_T = 1.0 / dP_drho_T
        drho_dP_Tref = 1.0 / dP_drho_Tref

        Delta_grads = drho_dP_T - (T_ref / T) * drho_dP_Tref
        Delta_grads = jnp.maximum(Delta_grads, 1e-15)

        K = self.chi_0 * (
            (self.p_c * rho) / (self.Gamma * (T / self.T_c) * self.rho_c**2)
        ) ** (self.nu / self.gamma)

        return K * (Delta_grads ** (self.nu / self.gamma))

    def _delta_lambda_c_theoretical(
        self,
        rho: jaxFloat,
        T: jaxFloat,
    ) -> jaxFloat:
        cp = self.eos.props_rhoT(rho, T)["cp"]  # type: ignore
        cv = self.eos.props_rhoT(rho, T)["cv"]  # type: ignore
        eta = self.viscosity.viscosity_rhoT(rho, T)  # type: ignore
        big_chi = self._big_chi(rho, T)

        # q_D * xi
        qD_xi = big_chi / self.qd_inv

        Ω = (2.0 / jnp.pi) * ((1.0 - cv / cp) * jnp.arctan(qD_xi) + (cv / cp) * qD_xi)

        qD_xi_safe = jnp.maximum(qD_xi, 1e-15)

        exp_denom = (1.0 / qD_xi_safe) + ((qD_xi * self.rho_c / rho) ** 2) / 3.0
        Ω_0 = (2.0 / jnp.pi) * (1.0 - jnp.exp(-1.0 / exp_denom))

        K = (rho * cp * self.RD * self.kb * T) / (6.0 * jnp.pi * big_chi * eta)

        return K * (Ω - Ω_0)

    @eqx.filter_jit
    def conductivity_rhoT(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        """
        Total thermal conductivity [W/(m K)].

        lambda = lambda0(T) + delta_lambda(rho, T) + delta_lambda_c(rho, T),
        with the critical enhancement from Olchowy-Sengers theory (parameters must
        thus be provided)
        """
        return (
            self._lambda_0(T)
            + self._delta_lambda(rho, T)
            + self._delta_lambda_c_theoretical(rho, T)
        )
