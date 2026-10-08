from .build import resolve_conductivity, resolve_viscosity
from .eos import convert_eos
from .source import load_coolprop_fluid, require_coolprop


def convertible_fluids() -> dict:
    CP = require_coolprop()
    out = {}
    for fluid in sorted(CP.get_global_param_string("FluidsList").split(",")):
        cp = load_coolprop_fluid(fluid)
        try:
            consts, _, _ = convert_eos(cp, fluid)
        except ValueError as e:
            out[fluid] = {"eos": str(e)}
            continue
        transport = cp.get("TRANSPORT", {})
        visc, visc_why = resolve_viscosity(
            fluid, transport.get("viscosity"), molar_mass=consts["molar_mass"]
        )
        _, cond_why = resolve_conductivity(
            fluid, transport.get("conductivity"), {"eos": consts, "viscosity": visc}
        )
        out[fluid] = {"eos": "ok", "viscosity": visc_why, "conductivity": cond_why}
    return out
