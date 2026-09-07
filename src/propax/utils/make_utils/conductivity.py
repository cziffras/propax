"""
Exact conversion of a CoolProp conductivity correlation into a propax block.

Same rule as `viscosity.py`: transcription of published coefficients, verified
against CoolProp, or nothing at all.
"""

from ...fluids._registry import _make_factories
from ...fluids.schema import FluidDefinition
from .source import require_coolprop

# CoolProp's own built-in defaults for a simplified Olchowy-Sengers block, read
# verbatim from ConductivityCriticalSimplifiedOlchowySengersData in
# CoolPropFluid.h
OS_DEFAULTS = dict(
    nu=0.63, gamma=1.239, R0=1.03, GAMMA=0.0496, zeta0=1.94e-10, qD=2.0e9
)

# used when CoolProp's fluid file publishes no simplified_Olchowy_Sengers block
# at all (propax then adds a generic enhancement CoolProp itself omits)
GENERIC_OS_CRITICAL = dict(
    nu=0.63, gamma=1.2415, Gamma=0.052, chi_0=1.5e-10, RD=1.01, qd_inv=5.0e-10
)


SLOT_COND_DILUTE = {"eta0_and_poly", "ratio_of_polynomials"}
SLOT_COND_RESIDUAL = {"polynomial_and_exponential", "polynomial"}
SLOT_COND_CRITICAL = {"simplified_Olchowy_Sengers"}


def convert_slot_conductivity(
    cp_cond: dict, *, T_c: float, rho_c: float, p_c_MPa: float
) -> dict:
    """CoolProp TRANSPORT.conductivity -> a "slot_composed" definition block.

    Raises ValueError on any slot form without a runtime class, so a fluid can
    never be silently mis-converted. `T_c`, `rho_c` and `p_c` come from the EOS
    reducing state: CoolProp keeps them implicit in the transport block (the
    ratio_of_polynomials / polynomial forms carry their own T_reducing /
    rho_reducing instead, which are copied from there).
    """
    d = cp_cond.get("dilute")
    if not isinstance(d, dict) or d.get("type") not in SLOT_COND_DILUTE:
        raise ValueError(
            f"unsupported dilute form {d.get('type') if isinstance(d, dict) else None}"
        )
    if d["type"] == "ratio_of_polynomials":
        dilute = {
            "type": "ratio_of_polynomials",
            "T_reducing": d["T_reducing"],
            "A": list(map(float, d["A"])),
            "n": list(map(float, d["n"])),
            "B": list(map(float, d["B"])),
            "m": list(map(float, d["m"])),
        }
    else:  # eta0_and_poly
        dilute = {
            "type": "eta0_and_poly",
            "T_c": T_c,
            "A": list(d["A"]),
            "t": list(map(float, d["t"])),
        }
    out = {
        "model": "slot_composed",
        "reference": f"Exact conversion of CoolProp {cp_cond.get('BibTeX', '')}.",
        "dilute": dilute,
    }

    r = cp_cond.get("residual")
    if isinstance(r, dict):
        if r.get("type") not in SLOT_COND_RESIDUAL:
            raise ValueError(f"unsupported residual form '{r.get('type')}'")
        if r["type"] == "polynomial":
            out["residual"] = {
                "type": "polynomial",
                "T_reducing": r["T_reducing"],
                "rho_reducing": r["rhomass_reducing"],
                "B": list(map(float, r["B"])),
                "d": list(map(float, r["d"])),
                "t": list(map(float, r["t"])),
            }
        else:  # polynomial_and_exponential
            out["residual"] = {
                "type": "polynomial_and_exponential",
                "T_c": T_c,
                "rho_c": rho_c,
                "A": list(r["A"]),
                "d": list(map(float, r["d"])),
                "t": list(map(float, r["t"])),
                "gamma": list(map(float, r["gamma"])),
                "l_exp": list(map(float, r["l"])),
            }

    c = cp_cond.get("critical")
    if isinstance(c, dict):
        if c.get("type") not in SLOT_COND_CRITICAL:
            raise ValueError(f"unsupported critical form '{c.get('type')}'")
        d = OS_DEFAULTS
        out["critical"] = {
            "type": "simplified_olchowy_sengers",
            "T_c": T_c,
            "rho_c": rho_c,
            "p_c": p_c_MPa,
            # each field falls back to CoolProp's own default when omitted
            "T_ref": c.get("T_ref", 1.5 * T_c),  # standard O-S reference
            "qD": c.get("qD", d["qD"]),
            "zeta0": c.get("zeta0", d["zeta0"]),
            "Gamma": c.get("GAMMA", d["GAMMA"]),
            "R0": c.get("R0", d["R0"]),
            "gamma": c.get("gamma", d["gamma"]),
        }
    return out


def verify_slot_conductivity(cp_name: str, draft: dict) -> tuple[float, float]:
    CP = require_coolprop()
    fe, _, fv, fc = _make_factories(FluidDefinition.model_validate(draft))
    eos, visc = fe(), fv()
    cond = fc(eos=eos, viscosity=visc)
    Tc = CP.PropsSI("Tcrit", cp_name)
    rho_c = CP.PropsSI("rhomass_critical", cp_name)

    def worst_over(t_fracs, rho_fracs):
        w = 0.0
        for tf in t_fracs:
            for rf in rho_fracs:
                T, rho = tf * Tc, rf * rho_c
                try:
                    ref = CP.PropsSI("L", "T", T, "D", rho, cp_name)
                except Exception:
                    continue
                got = float(cond.conductivity_rhoT(rho, T))
                w = max(w, abs(got / ref - 1.0))
        return w

    transcription = worst_over((2.0, 3.0), (0.02, 0.5, 1.0, 2.0))
    critical = worst_over((0.95, 1.05, 1.2, 1.5), (0.5, 1.0, 1.5))
    return transcription, critical
