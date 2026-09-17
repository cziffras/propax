from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from propax.fluids._registry import EQS_REGISTRY
from propax.fluids.generic import HelmholtzEOS
from propax.utils.numerics import pick

from ._state import PropertyMap, SaturationResult
from .config import ThermoVar
from .interp import ChebyshevPieces


def _pieces(layout, dtype) -> ChebyshevPieces:
    return ChebyshevPieces.from_arrays(
        layout.edges,
        layout.coeffs,
        tuple(layout.log_components),
        dtype,
    )


class Superancillary(eqx.Module):
    """This module is fundamental, it provides `propax` with all saturation
    properties with high accuracy across the whole dome and high speed computation
    through the Clenshaw iteration, all construction process is made offline in the
    /exact module.

    T anchors the curve and P is tied to it, so every entry reduces to a
    temperature first:

        T  -> (rho_L, rho_V)    `rho_sat`, in s = sqrt(1 - T/T_crit)
        P <-> T                 `P_sat`, in theta = 1 - T/T_crit

    Every caloric quantity, and the pressure a state reports, is the EOS read at
    the saturated densities, so a lever rule closed against an EOS evaluation
    cannot disagree with the endpoints it is closed on. `P_sat` only locates T.

    The densities leave the critical point like rho_c +/- B sqrt(theta), a branch
    point in T that no polynomial resolves and an ordinary analytic function of
    s. ln P_sat is smooth in T: in theta its slope stays finite, so it inverts
    in a few steps up to the critical point.
    """

    eos: HelmholtzEOS
    rho_sat: ChebyshevPieces
    """T -> (rho_L, rho_V), the two components, in s."""

    P_sat: ChebyshevPieces
    """T <-> P_sat, stored as ln P_sat, in theta."""

    T_crit: float = eqx.field(static=True)
    T_min: float = eqx.field(static=True)
    rho_max: float = eqx.field(static=True)

    L, V = 0, 1

    @classmethod
    def from_block(
        cls, eos: HelmholtzEOS, block, dtype=jnp.float64
    ) -> "Superancillary":
        """Assemble from the stored coefficients, a `SaturationSuperancillary`.

        T_crit comes from the EOS, which is where it is solved: the block is
        fitted against that same value and does not carry a second copy.
        """
        return cls(
            eos=eos,
            rho_sat=_pieces(block.densities, dtype),
            P_sat=_pieces(block.pressure, dtype),
            T_crit=float(eos.T_crit),
            T_min=float(block.T_min),
            rho_max=float(block.rho_max_mol) * float(eos.molar_mass),
        )

    def s_of_T(self, T):
        return jnp.sqrt(self.theta_of_T(T))

    def T_of_s(self, s):
        return self.T_of_theta(jnp.asarray(s) ** 2)

    def theta_of_T(self, T):
        return jnp.clip(1.0 - jnp.asarray(T) / self.T_crit, 0.0, None)

    def T_of_theta(self, theta):
        return self.T_crit * (1.0 - jnp.asarray(theta))

    def densities(self, T):
        """T -> (rho_L, rho_V). One Clenshaw, two components."""
        rho = self.rho_sat(self.s_of_T(T))
        return rho[self.L], rho[self.V]

    def T_of_rho(self, rho) -> Tuple[Array, Array]:
        """rho -> T on the dome, and whether the density ever sits on it.

        Above rho_crit that is the liquid branch, below it the vapour one.
        """
        rho = jnp.asarray(rho)
        branch = jnp.where(rho > self.eos.rho_crit_mass, self.L, self.V)
        s = self.rho_sat.invert(rho, branch)
        on_dome = jnp.isfinite(s)
        return self.T_of_s(pick(on_dome, s, 0.0)), on_dome

    def T_of_P(self, P) -> Tuple[Array, Array]:
        """P -> T on the curve, and whether the pressure is ever saturated."""
        theta = self.P_sat.invert(P, 0)
        on_curve = jnp.isfinite(theta)
        return self.T_of_theta(pick(on_curve, theta, 0.0)), on_curve

    def state_T(self, T) -> SaturationResult:
        """T -> both saturated branches, and the pressure they share."""
        T = jnp.asarray(T)
        valid = jax.lax.stop_gradient((T >= self.T_min) & (T <= self.T_crit))
        T_safe = jnp.clip(T, self.T_min, self.T_crit)
        rho_L, rho_V = self.densities(T_safe)

        pL = self.eos.props_rhoT(rho_L, T_safe)
        pV = self.eos.props_rhoT(rho_V, T_safe)

        def branch(rho, p):
            return PropertyMap(
                {
                    ThermoVar.D: rho,
                    ThermoVar.U: p[ThermoVar.U],
                    ThermoVar.H: p[ThermoVar.H],
                    ThermoVar.S: p[ThermoVar.S],
                }
            )

        # The vapour side carries the pressure
        return SaturationResult(
            L=branch(rho_L, pL),
            V=branch(rho_V, pV),
            T=T_safe,
            P=pV[ThermoVar.P],
            is_valid=valid,
        )

    def state_P(self, P) -> SaturationResult:
        """P -> T -> both saturated branches at T."""
        T, on_curve = self.T_of_P(P)
        state = self.state_T(T)
        return SaturationResult(
            L=state.L,
            V=state.V,
            T=state.T,
            P=state.P,
            is_valid=state.is_valid & on_curve,
        )

    def inside_dome(self, rho, T):
        """Is this state two-phase? False wherever the curve does not reach."""
        rho_L, rho_V = self.densities(jnp.clip(T, self.T_min, self.T_crit))
        return (T < self.T_crit) & (T >= self.T_min) & (rho > rho_V) & (rho < rho_L)

    @classmethod
    def create(
        cls, eos: HelmholtzEOS, fluid_name: str, dtype=jnp.float64
    ) -> "Superancillary":
        """Read the coefficients stored in this fluid's definition file."""
        block = EQS_REGISTRY[fluid_name][1]()
        if block is None:
            raise ValueError(
                f"{fluid_name!r} carries no superancillary: fit it with "
                "`python -m propax.make_fluid <name> --refit-superancillary`"
            )
        return cls.from_block(eos, block, dtype)
