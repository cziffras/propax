from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from propax.fluids._registry import EQS_REGISTRY
from propax.fluids.generic import HelmholtzEOS
from propax.utils.solvers import bisect

from ._state import PropertyMap, SaturationResult
from .config import ThermoVar
from .interp import ChebyshevPieces
from .tolerances import TOL


class Superancillary(eqx.Module):
    """This module is fundamental, it provides `propax` with all saturation
    properties with high accuracy across the whole dome and high speed computation
    through the Clenshaw iteration, all construction process is made offline in the
    /exact module.

    Two channels are stored, `rho_L` and `rho_V`, and every caloric quantity is
    the EOS read at one of them so a lever rule closed against an EOS
    evaluation cannot disagree with the endpoints it is closed on.

    The abscissa is s = sqrt(1 - T/T_crit): the saturated densities leave the
    critical point like rho_c +/- B sqrt(theta), a branch point in T that no
    polynomial resolves and an ordinary analytic function of s.
    """

    eos: HelmholtzEOS
    pair: ChebyshevPieces
    """The saturated densities, (rho_L, rho_V) as the two components."""

    cuts: Dict[str, Tuple[float, ...]] = eqx.field(static=True)
    """Where each derived channel turns over, so it can be inverted."""

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
            pair=ChebyshevPieces.from_arrays(
                block.edges,
                block.coeffs,
                block.cuts,
                tuple(block.log_components),
                dtype,
            ),
            cuts={name: tuple(c) for name, c in block.derived_cuts.items()},
            T_crit=float(eos.T_crit),
            T_min=float(block.T_min),
            rho_max=float(block.rho_max_mol) * float(eos.molar_mass),
        )

    def s_of_T(self, T):
        return jnp.sqrt(jnp.clip(1.0 - jnp.asarray(T) / self.T_crit, 0.0, None))

    def T_of_s(self, s):
        return self.T_crit * (1.0 - jnp.asarray(s) ** 2)

    def densities(self, T):
        """(rho_L, rho_V) at T. One Clenshaw, two components."""
        pair = self.pair(self.s_of_T(T))
        return pair[self.L], pair[self.V]

    def T_of_rho(self, rho) -> Tuple[Array, Array]:
        """The temperature at which a density sits on the dome, and whether it
        ever does.

        Above rho_crit that is the liquid branch, below it the vapour one. Both
        are monotone in T, so the whole curve is the bracket, and one Clenshaw
        answers for the branch the density selects.
        """
        rho = jnp.asarray(rho)
        branch = jnp.where(rho > self.eos.rho_crit_mass, self.L, self.V)
        rho_L_cold, rho_V_cold = self.densities(self.T_min)
        on_dome = (rho <= rho_L_cold) & (rho >= rho_V_cold)

        def gap(T, target):
            return self.pair(self.s_of_T(T))[branch] - target

        T_sat, converged = bisect(
            gap,
            jnp.asarray(self.T_min),
            jnp.asarray(self.T_crit),
            rho,
            max_steps=TOL.caps.bisect_steps,
            rtol=TOL.acc.sat_rtol,
            atol=TOL.acc.bisect_atol,
        )
        return jnp.asarray(T_sat), on_dome & converged

    def state_T(self, T) -> SaturationResult:
        """Both saturated branches at T, and the pressure they share."""
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
        """The same, entered by pressure."""
        P = jnp.asarray(P)

        def residual(T, target):
            _, rho_V = self.densities(T)
            return self.eos.props_rhoT(rho_V, T)[ThermoVar.P] - target

        T_sat, converged = bisect(
            residual,
            jnp.asarray(self.T_min),
            jnp.asarray(self.T_crit),
            P,
            max_steps=TOL.caps.bisect_steps,
            rtol=TOL.acc.sat_inversion_rtol,
            atol=TOL.acc.bisect_atol,
        )
        state = self.state_T(T_sat)
        return SaturationResult(
            L=state.L,
            V=state.V,
            T=state.T,
            P=state.P,
            is_valid=state.is_valid & converged,
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
