# Slot forms propax has a runtime class for; anything else is not converted.
SLOT_DILUTE = {"collision_integral", "powers_of_T", "powers_of_Tr"}
SLOT_INITIAL = {"Rainwater-Friend"}
SLOT_HIGHER = {"modified_Batschinski_Hildebrand"}


def convert_slot_viscosity(cp_visc: dict, *, molar_mass: float) -> dict:
    d = cp_visc.get("dilute")
    if not isinstance(d, dict) or d.get("type") not in SLOT_DILUTE:
        raise ValueError(
            f"unsupported dilute form {d.get('type') if isinstance(d, dict) else None} "
            f"(supported: {sorted(SLOT_DILUTE)})"
        )

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
