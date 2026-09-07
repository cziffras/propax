"""
Slot-composed viscosity :

A transport correlation is not one model but a **sum of density regimes**, each
of which picks a form from a menu:

    eta(rho, T) = eta0(T)  +  eta1(T) * rho  +  delta_eta(rho, T)
                  dilute      initial density   higher order

CoolProp mixes those slots freely, for instance : argon is `collision_integral` +
`modified_Batschinski_Hildebrand` with no initial-density term / hydrogen is
`collision_integral` + `Rainwater-Friend` with no higher-order term / methane is
`powers_of_Tr` + `friction_theory`...

Each slot is an eqx.Module with a `contribution(rho_molar, T, eta0)` method and
`ViscositySlots` sums them, so adding a form is adding a class here plus a
schema block.

/!\\ Every form's equation and its CoolProp source are documented in docs/transport.md
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from propax.utils.types import jaxFloat

from ..transport_registry import build_viscosity_higher_order

N_A = 6.02214076e23  # mol^-1 (The Avogadro constant)


class DiluteTerm(eqx.Module):
    """eta0(T), the zero-density limit [Pa.s]"""

    def eta0(self, T: jaxFloat) -> jaxFloat:
        raise NotImplementedError


class InitialDensityTerm(eqx.Module):
    """The term linear in density [Pa.s]"""

    def contribution(
        self, rho_molar: jaxFloat, T: jaxFloat, eta0: jaxFloat
    ) -> jaxFloat:
        raise NotImplementedError


class HigherOrderTerm(eqx.Module):
    """The dense-fluid residual [Pa.s]"""

    def contribution(
        self, rho_molar: jaxFloat, T: jaxFloat, eta0: jaxFloat
    ) -> jaxFloat:
        raise NotImplementedError


class CollisionIntegralDilute(DiluteTerm):
    """Chapman-Enskog dilute gas (CoolProp `dilute:collision_integral`):

        eta0 = C * sqrt(M * T) / (sigma^2 * Omega(T*)),  T* = T / (eps/k)
        Omega(T*) = exp( sum a_i (ln T*)^t_i )

    M is in g/mol and sigma in nm, the convention under which the stored C
    yields Pa.s directly, C is per-fluid (argon 2.66958e-08, propane
    2.1357e-08...), so it is read from the file, not hardcoded.
    """

    C: float
    M: float  # g/mol
    sigma: float  # nm
    eps_kb: float  # K
    a: np.ndarray
    t: np.ndarray

    def eta0(self, T: jaxFloat) -> jaxFloat:
        ln_Tstar = jnp.log(T / self.eps_kb)
        omega = jnp.exp(jnp.sum(self.a * ln_Tstar**self.t))
        return self.C * jnp.sqrt(self.M * T) / (self.sigma**2 * omega)


class PowersOfTDilute(DiluteTerm):
    """CoolProp `dilute:powers_of_T`:  eta0 = sum a_i T^t_i  [Pa.s]"""

    a: np.ndarray
    t: np.ndarray

    def eta0(self, T: jaxFloat) -> jaxFloat:
        return jnp.sum(self.a * T**self.t)


class PowersOfTrDilute(DiluteTerm):
    """CoolProp `dilute:powers_of_Tr`:  eta0 = sum a_i (T/T_reducing)^t_i [Pa.s]"""

    T_reducing: float
    a: np.ndarray
    t: np.ndarray

    def eta0(self, T: jaxFloat) -> jaxFloat:
        return jnp.sum(self.a * (T / self.T_reducing) ** self.t)


class RainwaterFriendInitialDensity(InitialDensityTerm):
    """First density correction (CoolProp `initial_density:Rainwater-Friend`):

        delta_eta = eta0 * B*(T*) * N_A * sigma^3 * rho_molar
        B*(T*) = sum b_i (T*)^t_i

    B* is the reduced second viscosity virial coefficient, which kinetic theory
    makes a function of T* alone.
    """

    sigma: float  # nm
    eps_kb: float  # K
    b: np.ndarray
    t: np.ndarray

    def contribution(
        self, rho_molar: jaxFloat, T: jaxFloat, eta0: jaxFloat
    ) -> jaxFloat:
        B_star = jnp.sum(self.b * (T / self.eps_kb) ** self.t)
        sigma_m = self.sigma * 1e-9
        return eta0 * B_star * N_A * sigma_m**3 * rho_molar


class ModifiedBatschinskiHildebrand(HigherOrderTerm):
    """Dense-fluid residual (CoolProp `higher_order:modified_Batschinski_Hildebrand`):

        delta_eta = sum a_i delta^d1_i tau^t1_i exp(gamma_i delta^l_i)
                  + f * (sum p_j tau^q_j) * delta^d2 * tau^t2
                        * ( 1/(delta0 - delta) - 1/delta0 )
        delta0(tau) = sum g_k tau^h_k

    with delta = rho_molar / rho_reduce and tau = T_reduce / T.
    """

    T_reduce: float
    rho_reduce: float  # mol/m3
    a: np.ndarray
    d1: np.ndarray
    t1: np.ndarray
    gamma: np.ndarray
    l_exp: np.ndarray
    # pole part
    f: float
    g: np.ndarray
    h: np.ndarray
    p: np.ndarray
    q: np.ndarray
    d2: float
    t2: float

    def contribution(
        self, rho_molar: jaxFloat, T: jaxFloat, eta0: jaxFloat
    ) -> jaxFloat:
        delta = rho_molar / self.rho_reduce
        tau = self.T_reduce / T

        poly = jnp.sum(
            self.a
            * delta**self.d1
            * tau**self.t1
            * jnp.exp(self.gamma * delta**self.l_exp)
        )

        delta0 = jnp.sum(self.g * tau**self.h)
        num = jnp.sum(self.p * tau**self.q)
        gap = jnp.where(jnp.abs(delta0 - delta) < 1e-12, 1e-12, delta0 - delta)
        pole = self.f * num * delta**self.d2 * tau**self.t2 * (1.0 / gap - 1.0 / delta0)
        # f = 0 switches the pole off exactly (no rounding error), without evaluating a bad 1/gap
        return poly + jnp.where(self.f == 0.0, 0.0, pole)


class ViscositySlots(eqx.Module):
    dilute: DiluteTerm
    initial_density: object = None  # Optional[InitialDensityTerm]
    higher_order: object = None  # Optional[HigherOrderTerm]
    molar_mass: float = 0.0  # kg/mol, to go from mass to molar density
    reference: str = eqx.field(static=True, default="")

    @classmethod
    def from_definition(cls, defn, *, eos=None) -> "ViscositySlots":
        d = defn.dilute
        if d.type == "powers_of_T":
            dilute = PowersOfTDilute(a=np.asarray(d.a), t=np.asarray(d.t))
        elif d.type == "powers_of_Tr":
            dilute = PowersOfTrDilute(
                T_reducing=d.T_reducing, a=np.asarray(d.a), t=np.asarray(d.t)
            )
        else:  # collision_integral
            dilute = CollisionIntegralDilute(
                C=d.C,
                M=d.M,
                sigma=d.sigma,
                eps_kb=d.eps_kb,
                a=np.asarray(d.a),
                t=np.asarray(d.t),
            )

        initial = None
        if defn.initial_density is not None:
            i = defn.initial_density
            initial = RainwaterFriendInitialDensity(
                sigma=i.sigma,
                eps_kb=i.eps_kb,
                b=np.asarray(i.b),
                t=np.asarray(i.t),
            )

        higher = None
        if defn.higher_order is not None:
            h = defn.higher_order
            if h.type == "custom":
                higher = build_viscosity_higher_order(h.name, h.params, eos=eos)
            else:  # modified_batschinski_hildebrand
                higher = ModifiedBatschinskiHildebrand(
                    T_reduce=h.T_reduce,
                    rho_reduce=h.rho_reduce,
                    a=np.asarray(h.a),
                    d1=np.asarray(h.d1),
                    t1=np.asarray(h.t1),
                    gamma=np.asarray(h.gamma),
                    l_exp=np.asarray(h.l_exp),
                    f=h.f,
                    g=np.asarray(h.g),
                    h=np.asarray(h.h),
                    p=np.asarray(h.p),
                    q=np.asarray(h.q),
                    d2=h.d2,
                    t2=h.t2,
                )

        return cls(
            dilute=dilute,
            initial_density=initial,
            higher_order=higher,
            molar_mass=defn.molar_mass,
            reference=defn.reference,
        )

    def viscosity_rhoT(self, rho: jaxFloat, T: jaxFloat) -> jaxFloat:
        """Dynamic viscosity [Pa.s] from mass density [kg/m3] and T [K]"""
        rho_molar = rho / self.molar_mass
        eta0 = self.dilute.eta0(T)
        total = eta0
        for slot in (self.initial_density, self.higher_order):
            if slot is not None:
                total = total + slot.contribution(rho_molar, T, eta0)  # type: ignore
        return total
