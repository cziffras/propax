from typing import Any, Dict, Literal, Tuple, Union, overload

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from propax.core._state import PropertyMap
from propax.core.config import ThermoVar
from propax.fluids.generic.eos.pure_terms import (
    ExponentialDensity,
    Gaussian,
    GeneralizedExponential,
    IdealLead,
    IdealPlanckEinstein,
    Polynomial,
    static_exponents,
    static_floats,
)
from propax.utils.numerics import pick

from .schema_eos import HelmholtzEOSDefinition


class HelmholtzEOS(eqx.Module):
    """
    Multiparameter Helmholtz equation of state for a pure fluid, built from a
    parsed fluid definition file (see `propax.fluids.schema`).

    The reduced Helmholtz energy is assembled compositionally: `ideal_terms`
    and `residual_terms` are tuples of `IdealTerm` / `ResidualTerm` modules
    (see above) whose contributions are summed in `alpha0` / `alphar`. This
    class encodes the assembly and the thermodynamics; each functional form
    lives in its own term class.
    """

    # Constants
    R_u: float
    molar_mass: float  # kg/mol
    # /!\ Reducing parameters
    T_red: float  # K
    P_red: float  # Pa
    rho_red_mol: float  # mol/m3
    T_crit: float  # K
    P_crit: float  # Pa
    rho_crit_mol: float  # mol/m3
    T_triple: float  # K
    P_triple: float  # Pa
    T_max: float  # K, upper end of the EOS's stated validity range
    P_max: float  # Pa

    # Helmholtz energy, as sums of term modules
    ideal_terms: tuple  # tuple[IdealTerm, ...] -> alpha0
    residual_terms: tuple  # tuple[ResidualTerm, ...] -> alphar

    reference: str = eqx.field(static=True, default="")

    @classmethod
    def from_definition(cls, defn: HelmholtzEOSDefinition) -> "HelmholtzEOS":
        res = defn.residual
        ideal = defn.ideal

        # ideal-gas part: lead/log/linear always present; Einstein terms if any
        ideal_terms: tuple = (IdealLead(a1=ideal.a1, a2=ideal.a2, a_log=ideal.a_log),)
        pe_n = np.asarray(ideal.pe_n)
        if pe_n.size:
            ideal_terms = ideal_terms + (
                IdealPlanckEinstein(
                    n=static_floats(pe_n),
                    theta=static_floats(np.asarray(ideal.pe_v) / defn.T_red),
                ),
            )

        # residual part: include only the term types the fluid actually uses
        residual_terms: tuple = ()
        if len(res.poly_n):
            residual_terms = residual_terms + (
                Polynomial(
                    n=static_floats(res.poly_n),
                    d=static_exponents(res.poly_d),
                    t=static_exponents(res.poly_t),
                ),
            )
        if len(res.exp_n):
            residual_terms = residual_terms + (
                ExponentialDensity(
                    n=static_floats(res.exp_n),
                    d=static_exponents(res.exp_d),
                    t=static_exponents(res.exp_t),
                    p=static_exponents(res.exp_p),
                ),
            )
        if len(res.gexp_n):
            residual_terms = residual_terms + (
                GeneralizedExponential(
                    n=static_floats(res.gexp_n),
                    d=static_exponents(res.gexp_d),
                    t=static_exponents(res.gexp_t),
                    p=static_exponents(res.gexp_p),
                    g=static_floats(res.gexp_g),
                ),
            )
        if len(res.gauss_n):
            residual_terms = residual_terms + (
                Gaussian(
                    n=static_floats(res.gauss_n),
                    d=static_exponents(res.gauss_d),
                    t=static_exponents(res.gauss_t),
                    phi=static_floats(res.gauss_phi),
                    beta=static_floats(res.gauss_beta),
                    D=static_floats(res.gauss_D),
                    gamma=static_floats(res.gauss_gamma),
                ),
            )

        return cls(
            R_u=defn.R_u,
            molar_mass=defn.molar_mass,
            T_red=defn.T_red,
            P_red=defn.P_red,
            rho_red_mol=defn.rho_red_mol,
            T_crit=defn.T_crit,
            P_crit=defn.P_crit,
            rho_crit_mol=defn.rho_crit_mol,
            T_triple=defn.T_triple,
            P_triple=defn.P_triple,
            # definition files predating T_max fall back convention of 4.5*T_crit
            T_max=defn.T_max if defn.T_max is not None else 4.5 * defn.T_red,
            P_max=defn.P_max,
            ideal_terms=ideal_terms,
            residual_terms=residual_terms,
            reference=defn.reference,
        )

    @property
    def R_spec(self):
        return self.R_u / self.molar_mass

    @property
    def rho_red_mass(self):
        """Reducing density. For the dome and the phase tests, `rho_crit_mass`."""
        return self.rho_red_mol * self.molar_mass

    @property
    def rho_crit_mass(self):
        return self.rho_crit_mol * self.molar_mass

    def alphar(self, delta, tau, namespace=jnp):
        """Reduced residual Helmholtz energy: sum of the residual term modules.

        `namespace` supplies the namespace, so the same sum runs in JAX at
        runtime and in mpmath extended precision for offline computations
        (mostly for fitting superancillaries at saturation).
        """
        total = 0.0 * delta * tau
        for term in self.residual_terms:
            total = total + term.contribution(delta, tau, namespace)
        return total

    def _thermo_state(self, delta, tau):
        """Both potentials and every derivative the thermodynamics needs.

        Summed term by term from the closed forms each family carries, rather
        than obtained by differentiating the sums. The derivatives share the
        powers and the exponential with the value, so a term yields all six for
        little more than its own evaluation, where a Hessian by
        forward-over-reverse costs several passes  reading dP/drho alongside
        P used to triple the price of an evaluation.
        """
        a0 = a0_t = a0_tt = jnp.asarray(0.0)
        for term in self.ideal_terms:
            v, v_t, v_tt = term.tau_derivatives(delta, tau)
            a0, a0_t, a0_tt = a0 + v, a0_t + v_t, a0_tt + v_tt

        ar = ar_d = ar_t = ar_dd = ar_tt = ar_dt = jnp.asarray(0.0)
        for term in self.residual_terms:
            v, v_d, v_t, v_dd, v_tt, v_dt = term.derivatives(delta, tau)
            ar, ar_d, ar_t = ar + v, ar_d + v_d, ar_t + v_t
            ar_dd, ar_tt, ar_dt = ar_dd + v_dd, ar_tt + v_tt, ar_dt + v_dt

        return dict(
            a0=a0,
            ar=ar,
            a0_t=a0_t,
            ar_d=ar_d,
            ar_t=ar_t,
            a0_tt=a0_tt,
            ar_dd=ar_dd,
            ar_tt=ar_tt,
            ar_dt=ar_dt,
        )

    @overload
    def props_rhoT(
        self, rho_mass, T, with_derivatives: Literal[False] = False
    ) -> PropertyMap: ...

    @overload
    def props_rhoT(
        self, rho_mass, T, with_derivatives: Literal[True]
    ) -> Tuple[PropertyMap, Dict[str, Any]]: ...

    def props_rhoT(
        self, rho_mass, T, with_derivatives: bool = False
    ) -> Union[PropertyMap, Tuple[PropertyMap, Dict[str, Any]]]:
        """
            CORE METHOD

            Compute thermodynamic properties from density and temperature.

            Returns NaN for all properties if inputs are physically invalid
            (rho <= 0 or T <= 0). This ensures safe use in vectorized computations
            without exceptions.

            Parameters
        -
            rho_mass : float or jax.Array
                Mass density [kg/m³]. Must be positive.
            T : float or jax.Array
                Temperature [K]. Must be positive.

            Returns
        -
            dict
                Thermodynamic properties containing:
                - P : Pressure [Pa]
                - rho : Mass density [kg/m³]
                - T : Temperature [K]
                - u : Specific internal energy [J/kg]
                - h : Specific enthalpy [J/kg]
                - s : Specific entropy [J/(kg·K)]

                Second order properties
                - cv : Isochoric heat capacity [J/(kg·K)]
                - cp : Isobaric heat capacity [J/(kg·K)]

            `with_derivatives` returns `(state, derivatives)` instead of the
            state alone, the second holding the two first derivatives of the
            pressure:

                - dP_drho : dP/drho at fixed T [Pa.m3/kg]
                - dP_dT   : dP/dT at fixed rho [Pa/K]

            Both are already formed on the way to `cp`.

            Together with `cp` they give every derivative the bracketed solvers
            need:
            - dP/drho for the density solve

            and along an isobar
            - dh/dT = cp, ds/dT = cp/T
            - du/dT = cp - (P/rho^2) dP_dT / dP_drho
        """
        is_valid = (rho_mass > 0.0) & (T > 0.0)

        T_pos, rho_mass_pos = (
            pick(T > 0, T, 1.0),
            pick(rho_mass > 0, rho_mass, 1.0),
        )

        rho_mol = rho_mass_pos / self.molar_mass
        delta = rho_mol / self.rho_red_mol
        tau = self.T_red / T_pos

        derivs = self._thermo_state(delta, tau)  # derivatives are computed only once

        # Pressure
        Z = 1.0 + delta * derivs["ar_d"]
        P = rho_mol * self.R_u * T_pos * Z

        # molar internal energy
        u_mol = self.R_u * T_pos * tau * (derivs["a0_t"] + derivs["ar_t"])
        u_spec = u_mol / self.molar_mass  # J/kg

        # molar enthalpy
        h_mol = (
            self.R_u
            * T_pos
            * (1 + tau * (derivs["a0_t"] + derivs["ar_t"]) + delta * derivs["ar_d"])
        )
        h_spec = h_mol / self.molar_mass  # J/kg

        # molar entropy
        s_mol = self.R_u * (
            tau * (derivs["a0_t"] + derivs["ar_t"]) - derivs["a0"] - derivs["ar"]
        )
        s_spec = s_mol / self.molar_mass  # J/(kg·K)

        # Cv
        cv_mol = -self.R_u * tau**2 * (derivs["a0_tt"] + derivs["ar_tt"])
        cv_spec = cv_mol / self.molar_mass  # J/(kg·K)

        # Cp. Both factors are derivatives of the pressure in disguise:
        # dP/dT|_rho carries `dP_dT_factor` and dP/drho|_T carries `dP_drho_factor`
        dP_dT_factor = 1 + delta * derivs["ar_d"] - delta * tau * derivs["ar_dt"]
        dP_drho_factor = 1 + 2 * delta * derivs["ar_d"] + delta**2 * derivs["ar_dd"]
        cp_mol = cv_mol + self.R_u * (dP_dT_factor**2 / dP_drho_factor)
        cp_spec = cp_mol / self.molar_mass  # J/(kg·K)

        nan_val = jnp.array(jnp.nan)
        result = {
            ThermoVar.P: jnp.where(is_valid, P, nan_val),
            ThermoVar.D: jnp.where(is_valid, rho_mass_pos, nan_val),
            ThermoVar.T: jnp.where(is_valid, T_pos, nan_val),
            ThermoVar.U: jnp.where(is_valid, u_spec, nan_val),
            ThermoVar.H: jnp.where(is_valid, h_spec, nan_val),
            ThermoVar.S: jnp.where(is_valid, s_spec, nan_val),
            ThermoVar.CVMASS: jnp.where(is_valid, cv_spec, nan_val),
            ThermoVar.CPMASS: jnp.where(is_valid, cp_spec, nan_val),
        }  # type: ignore

        state = PropertyMap(result)
        if not with_derivatives:
            return state

        # a derivative is not a state variable, it is no ThermoVar, and it must never be reachable where a
        # flash looks for an input
        derivatives = {
            "dP_drho": jnp.array(
                jnp.where(is_valid, self.R_spec * T_pos * dP_drho_factor, nan_val)
            ),
            "dP_dT": jnp.array(
                jnp.where(is_valid, rho_mass_pos * self.R_spec * dP_dT_factor, nan_val)
            ),
        }
        return state, derivatives
