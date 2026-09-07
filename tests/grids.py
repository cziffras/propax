from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from propax.core.tolerances import TOL

N_RHO = 300
N_T = 300
MARGIN = 1e-3


N_DOME = 200


def _inside(lo: float, hi: float, n: int, margin: float, space):
    """`n` points spanning [lo, hi] without touching either end."""
    edge = margin * (hi - lo)
    return space(lo + edge, hi - edge, n)


def dome_line(saturation, n: int = N_DOME, margin: float = MARGIN):
    """
    The branches merge at the critical point and rho_V has nothing left at the
    triple point, so the curve is sampled strictly between them.
    """
    return _inside(
        float(saturation.T_min), float(saturation.T_crit), n, margin, jnp.linspace
    )


@dataclass(frozen=True)
class Grid:
    """A fluid's own domain, sampled once, with everything the tests read off it.

    The RhoT grid is sampled in fluid dependent bounds.
    """

    rho: Array
    T: Array
    props: dict
    in_dome: Array

    @property
    def single_phase(self) -> np.ndarray:
        return np.asarray(~self.in_dome)

    @property
    def two_phase(self) -> np.ndarray:
        return np.asarray(self.in_dome)

    def where(self, mask) -> "Grid":
        m = np.asarray(mask)
        return Grid(
            rho=self.rho[m],
            T=self.T[m],
            props={k: v[m] for k, v in self.props.items()},
            in_dome=self.in_dome[m],
        )

    def sample(self, n: int, seed: int = 0) -> "Grid":
        """A seeded subset, for the tests that cannot afford every node."""
        if n >= len(self):
            return self
        drawn = np.random.default_rng(seed).choice(len(self), size=n, replace=False)
        return self.where(drawn)

    def __len__(self) -> int:
        return int(self.rho.size)


def build(
    interface, n_rho: int = N_RHO, n_T: int = N_T, margin: float = MARGIN
) -> Grid:
    """
    Density is geometric because it spans decades between the dilute gas and
    the compressed liquid, temperature linear between the triple point and the
    correlation's ceiling.

    `margin` backs off both ends of each axis by that fraction of its width.
    Landing exactly on the triple point puts the whole first row at the foot of
    the dome, where rho_V is very low and the lever rule has nothing to divide by,
    so the solvers rightly refuse and the grid would be testing its own edge.
    """
    eos, sat = interface.eos, interface.saturation

    T = _inside(float(eos.T_triple), float(eos.T_max), n_T, margin, jnp.linspace)
    rho = _inside(
        TOL.domain.rho_lo_frac * float(eos.rho_crit_mass),
        float(sat.rho_max),
        n_rho,
        margin,
        jnp.geomspace,
    )
    RHO, TT = jnp.meshgrid(rho, T, indexing="xy")
    RHO, TT = RHO.ravel(), TT.ravel()

    props = jax.jit(jax.vmap(interface.props_rhoT))(RHO, TT)
    in_dome = jax.jit(jax.vmap(sat.inside_dome))(RHO, TT)

    return Grid(rho=RHO, T=TT, props=dict(props), in_dome=in_dome)
