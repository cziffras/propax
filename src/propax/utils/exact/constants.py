import jax.numpy as jnp

from ...core.config import ThermoVar
from ...core.tolerances import TOL
from ...fluids.generic.eos.helmholtz import HelmholtzEOS
from ...fluids.generic.eos.schema_eos import HelmholtzEOSDefinition
from ..solvers import newton_loop
from .critical import solve_critical_point


def provisional_eos(eos_block: dict):
    """A runnable EOS from a block whose solved constants are still missing
    because its constants were not yet calculated.
    """
    seed = dict(
        eos_block,
        T_crit=eos_block.get("T_crit", eos_block["T_red"]),
        P_crit=eos_block.get("P_crit", eos_block["P_red"]),
        rho_crit_mol=eos_block.get("rho_crit_mol", eos_block["rho_red_mol"]),
    )
    return HelmholtzEOS.from_definition(HelmholtzEOSDefinition(**seed))


def critical_constants(eos_block: dict) -> dict:
    crit = solve_critical_point(provisional_eos(eos_block))
    return {
        "T_crit": float(crit.T_crit),
        "P_crit": float(crit.P_crit),
        "rho_crit_mol": float(crit.rho_crit_mol),
    }


def density_ceiling(eos, rho_L_triple: float) -> float:
    """mol/m3 at (T_triple, P_max), solved along in the liquid."""
    T, P_max = float(eos.T_triple), float(eos.P_max)

    def pressure(rho_mass) -> float:
        return float(eos.props_rhoT(jnp.asarray(rho_mass), jnp.asarray(T))[ThermoVar.P])

    lo, P_lo = float(rho_L_triple), pressure(rho_L_triple)
    if P_lo >= P_max:
        raise ValueError(
            f"the saturated liquid at {T:.2f} K is already past P_max = "
            f"{P_max:.3e} Pa, so the correlation states no compressed liquid"
        )

    hi, P_hi = lo, P_lo
    while P_hi < P_max:
        lo, P_lo = hi, P_hi
        hi = lo * (1.0 + TOL.domain.rho_ceiling_step)
        P_hi = pressure(hi)
        if not jnp.isfinite(P_hi) or P_hi <= P_lo:
            raise ValueError(
                f"the correlation turns over at {hi:.1f} kg/m3 (P = {P_hi:.3e} Pa) "
                f"before reaching P_max = {P_max:.3e} Pa at {T:.2f} K"
            )

    def residual(rho, _):
        state, derivs = eos.props_rhoT(rho, jnp.asarray(T), with_derivatives=True)
        return state[ThermoVar.P] - P_max, derivs["dP_drho"]

    # the walk above leaves P_max bracketed where P rises with rho
    rho_mass, _ = newton_loop(
        residual,
        jnp.asarray(lo),
        jnp.asarray(hi),
        None,
        max_steps=TOL.caps.newton_steps,
        rtol=TOL.acc.newton_rtol,
    )
    return float(rho_mass) / eos.molar_mass
