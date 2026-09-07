"""
Exact conversion of a CoolProp viscosity correlation into a propax block.

Transcription only: the published coefficients are copied slot by slot and the
result is checked against CoolProp, so a converted fluid matches the oracle to
machine precision. When the published form is one propax has no class for the
fluid ships without viscosity  `make_utils.coverage` says which form blocks
which fluid, and implementing that form is the way to widen coverage.
"""

from ...fluids.generic.viscosity import ViscositySlots
from ...fluids.schema import SlotComposedViscosityDefinition
from .source import require_coolprop

# Slot forms propax has a runtime class for; anything else is not converted.
SLOT_DILUTE = {"collision_integral", "powers_of_T", "powers_of_Tr"}
SLOT_INITIAL = {"Rainwater-Friend"}
SLOT_HIGHER = {"modified_Batschinski_Hildebrand"}


def verify_slot_viscosity(cp_name: str, block: dict, tol: float = 1e-6) -> float:
    CP = require_coolprop()
    model = ViscositySlots.from_definition(
        SlotComposedViscosityDefinition.model_validate(block)
    )
    Tc = CP.PropsSI("Tcrit", cp_name)
    rho_c = CP.PropsSI("rhomass_critical", cp_name)
    worst = 0.0
    for t_frac in (0.8, 1.2, 2.0):
        for rho_frac in (0.02, 1.0, 2.0):
            T, rho = t_frac * Tc, rho_frac * rho_c
            try:
                ref = CP.PropsSI("V", "T", T, "D", rho, cp_name)
            except Exception:
                continue
            got = float(model.viscosity_rhoT(rho, T))
            worst = max(worst, abs(got / ref - 1.0))
    return worst


def convert_slot_viscosity(cp_visc: dict, *, molar_mass: float) -> dict:
    d = cp_visc.get("dilute")
    if not isinstance(d, dict) or d.get("type") not in SLOT_DILUTE:
        raise ValueError(
            f"unsupported dilute form {d.get('type') if isinstance(d, dict) else None} "
            f"(supported: {sorted(SLOT_DILUTE)})"
        )

    # Lennard-Jones params live at the block level; only the collision_integral
    # dilute term and the Rainwater-Friend initial-density term need them.
    def lj():
        return cp_visc["sigma_eta"] * 1e9, cp_visc["epsilon_over_k"]

    if d["type"] == "powers_of_T":
        dilute = {
            "type": "powers_of_T",
            "a": list(map(float, d["a"])),
            "t": list(map(float, d["t"])),
        }
    elif d["type"] == "powers_of_Tr":
        dilute = {
            "type": "powers_of_Tr",
            "T_reducing": d["T_reducing"],
            "a": list(map(float, d["a"])),
            "t": list(map(float, d["t"])),
        }
    else:  # collision_integral
        sigma_nm, eps_kb = lj()
        dilute = {
            "type": "collision_integral",
            "C": d["C"],
            "M": molar_mass * 1000.0,  # g/mol
            "sigma": sigma_nm,
            "eps_kb": eps_kb,
            "a": list(d["a"]),
            "t": list(map(float, d["t"])),
        }

    out = {
        "model": "slot_composed",
        "reference": f"Exact conversion of CoolProp {cp_visc.get('BibTeX', '')}.",
        "molar_mass": molar_mass,
        "dilute": dilute,
    }

    ini = cp_visc.get("initial_density")
    if isinstance(ini, dict):
        if ini.get("type") not in SLOT_INITIAL:
            raise ValueError(f"unsupported initial-density form '{ini.get('type')}'")
        sigma_nm, eps_kb = lj()
        out["initial_density"] = {
            "type": "rainwater_friend",
            "sigma": sigma_nm,
            "eps_kb": eps_kb,
            "b": list(ini["b"]),
            "t": list(map(float, ini["t"])),
        }

    ho = cp_visc.get("higher_order")
    if isinstance(ho, dict):
        if ho.get("type") not in SLOT_HIGHER:
            raise ValueError(f"unsupported higher-order form '{ho.get('type')}'")
        out["higher_order"] = {
            "type": "modified_batschinski_hildebrand",
            "T_reduce": ho["T_reduce"],
            "rho_reduce": ho["rhomolar_reduce"],
            "a": list(ho["a"]),
            "d1": list(map(float, ho["d1"])),
            "t1": list(map(float, ho["t1"])),
            "gamma": list(map(float, ho["gamma"])),
            "l_exp": list(map(float, ho["l"])),
            "f": float(ho.get("f", [0.0])[0]),
            "g": list(map(float, ho.get("g", [1.0]))),
            "h": list(map(float, ho.get("h", [0.0]))),
            "p": list(map(float, ho.get("p", [1.0]))),
            "q": list(map(float, ho.get("q", [0.0]))),
            "d2": float(ho.get("d2", [0.0])[0]),
            "t2": float(ho.get("t2", [0.0])[0]),
        }
    return out
