import equinox as eqx
import jax.numpy as jnp
import numpy as np

from propax.utils.types import jaxFloat

from .schema_visc import DiluteRainwaterFriendViscosityDefinition


class DiluteRainwaterFriendViscosity(eqx.Module):
    """
    Viscosity correlation family "dilute_rainwater_friend" (Muzny et al. style):

        eta(rho, T) = eta0(T) + eta1(T) * rho + eta_residual(rho, T)   [Pa.s]

    with a Chapman-Enskog dilute-gas term, a Rainwater-Friend initial density
    term and an empirical residual term. All coefficients come from the fluid
    definition file.
    """

    rho_sc: float  # scaling density kg/m3
    T_c: float  # K
    M: float  # g/mol
    sigma: float  # nm (collision diameter)
    epskb: float  # K
    Na: float = eqx.field(default=6.02214076e23)  # Avogadro number mol^-1

    pas: np.ndarray = eqx.field(default=None)  # dilute-gas ln(S*) coefficients
    pbs: np.ndarray = eqx.field(default=None)  # Rainwater-Friend B* coefficients
    pcs: np.ndarray = eqx.field(default=None)  # residual coefficients

    reference: str = eqx.field(static=True, default="")

    @classmethod
    def from_definition(
        cls, defn: DiluteRainwaterFriendViscosityDefinition
    ) -> "DiluteRainwaterFriendViscosity":
        return cls(
            rho_sc=defn.rho_sc,
            T_c=defn.T_c,
            M=defn.M,
            sigma=defn.sigma,
            epskb=defn.eps_kb,
            pas=np.asarray(defn.a),
            pbs=np.asarray(defn.b),
            pcs=np.asarray(defn.c),
            reference=defn.reference,
        )

    def _eta0(self, T: jaxFloat) -> jaxFloat:
        T_star = T / self.epskb
        num = 0.021357 * (self.M * T) ** 0.5

        # ln(S*) = sum(a_i * (ln(T*))^i)
        exponents = jnp.arange(self.pas.shape[0])
        ln_S_star = jnp.sum(self.pas * (jnp.log(T_star) ** exponents))

        S_star = jnp.exp(ln_S_star)
        denom = S_star * (self.sigma**2)

        return num / denom

    def _eta1(self, T: jaxFloat) -> jaxFloat:
        T_star = T / self.epskb

        # B_eta* = sum(b_i * (T*)^{-i})
        exponents = -jnp.arange(self.pbs.shape[0])
        B_star = jnp.sum(self.pbs * (T_star**exponents))

        M_kg_mol = self.M * 1e-3  # molar mass kg/mol
        sigma_m = self.sigma * 1e-9  # collision diameter in meters

        conversion_factor = (self.Na * (sigma_m**3)) / M_kg_mol

        return B_star * conversion_factor * self._eta0(T)

    def f(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        rho_r = rho / self.rho_sc
        T_r = T / self.T_c

        result = (
            self.pcs[0]
            * (rho_r**2)
            * jnp.exp(
                self.pcs[1] * T_r
                + self.pcs[2] / T_r
                + (self.pcs[3] * (rho_r**2) / (self.pcs[4] + T_r))
                + self.pcs[5] * (rho_r**6)
            )
        )
        return result

    def viscosity_rhoT(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        """viscosity in Pa.s."""
        return (self._eta0(T) + self._eta1(T) * rho + self.f(rho, T)) * 1e-6
