import jax
import numpy as np
import pytest

from .conftest import RIGHT_ROOT

CP = pytest.importorskip(
    "CoolProp.CoolProp", reason="CoolProp only serves as a reference oracle"
)

PHASE_PROBE = 4000


def _worst(label, err, state, n=6):
    order = np.argsort(-err)[:n]
    rows = "\n".join(
        f"    T={state['T'][i]:9.3f}  rho={state['rho'][i]:10.4f}  "
        f"P={state['P'][i]:.5e}  h={state['h'][i]:+.5e}  err={err[i]:.3e}"
        for i in order
    )
    return f"{label}: worst {len(order)} of {err.size}\n{rows}"


class TestTheFlashLandsOnTheRightRoot:
    """Whether the state propax converges to is the physical one.

    Not how precisely it converges, which `test_residuals` settles without an
    oracle: these bounds are loose on purpose, because a metastable root or the
    wrong branch misses by orders of magnitude while CoolProp's own inversion
    carries a tail of its own that we are in no position to call an error.
    """

    def test_the_dome_mask_agrees_with_coolprop(self, grid, cp_fluid):
        probe = np.random.default_rng(7).choice(
            len(grid), size=min(PHASE_PROBE, len(grid)), replace=False
        )
        rho, T = np.asarray(grid.rho)[probe], np.asarray(grid.T)[probe]
        theirs = np.array(
            [
                CP.PhaseSI("T", float(t), "D", float(r), cp_fluid) == "twophase"
                for t, r in zip(T, rho)
            ]
        )
        disagreed = np.asarray(grid.in_dome)[probe] != theirs
        assert disagreed.mean() <= RIGHT_ROOT.phase_disagreement, (
            f"the dome mask disagrees with CoolProp on {disagreed.mean():.2%} of "
            f"{probe.size} probed states"
        )

    def test_ph_lands_where_coolprop_lands(self, props, grid, cp_fluid):
        off_dome = grid.where(grid.single_phase)
        P, h = off_dome.props["P"], off_dome.props["h"]

        out, ok = jax.jit(jax.vmap(lambda p, x: props.flash("P", p, "H", x)))(P, h)
        solved = np.asarray(ok)

        state = {
            "T": np.asarray(off_dome.T)[solved],
            "rho": np.asarray(off_dome.rho)[solved],
            "P": np.asarray(P)[solved],
            "h": np.asarray(h)[solved],
        }
        T_cp = CP.PropsSI("T", "P", state["P"], "H", state["h"], cp_fluid)
        D_cp = CP.PropsSI("D", "P", state["P"], "H", state["h"], cp_fluid)

        # CoolProp refuses the densest states outright rather than reporting it,
        # so the comparison runs on what both codes actually answered
        oracle = np.isfinite(T_cp) & np.isfinite(D_cp) & (T_cp > 0) & (D_cp > 0)
        assert oracle.mean() >= RIGHT_ROOT.oracle_fraction, (
            f"CoolProp answered on only {oracle.mean():.1%} of the solved states"
        )

        T_px = np.asarray(out["T"])[solved]
        D_px = np.asarray(out["rho"])[solved]
        sub = {k: v[oracle] for k, v in state.items()}

        for label, mine, theirs, bound in (
            ("T", T_px, T_cp, RIGHT_ROOT.temperature),
            ("rho", D_px, D_cp, RIGHT_ROOT.density),
        ):
            err = np.abs(theirs[oracle] - mine[oracle]) / theirs[oracle]
            assert err.max() < bound, _worst(
                f"{label} is not the root CoolProp found", err, sub
            )
