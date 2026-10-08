import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import jax

from ...fluids.schema import DATA_DIR
from ..exact.constants import critical_constants, provisional_eos
from ..exact.fit import from_eos
from .conductivity import convert_slot_conductivity
from .eos import convert_eos
from .source import citation, hardcoded_here, hardcoded_in, load_coolprop_fluid
from .viscosity import convert_slot_viscosity

logger = logging.getLogger(__name__)

Resolved = Tuple[Optional[dict], str]

_CONVERTERS = {
    "viscosity": lambda block, molar_mass: convert_slot_viscosity(
        block, molar_mass=molar_mass
    ),
    "conductivity": lambda block, eos: convert_slot_conductivity(
        block,
        T_c=eos["T_red"],
        rho_c=eos["rho_red_mol"] * eos["molar_mass"],
        p_c_MPa=eos["P_red"] / 1e6,
    ),
}
_REGISTER = {
    "viscosity": "register_viscosity_higher_order",
    "conductivity": "register_conductivity_residual",
}


def _nonslot_reason(block) -> Optional[str]:
    """
    Hardcoded correlations and extended-corresponding-states (ECS) models are
    both real CoolProp physics with no closed slot form to copy: ECS scales
    the property from a reference fluid, hardcoded blocks live in C++ with their
    coefficients absent from the JSON and should be copied by the user via custom
    transport blocks.
    """
    if not isinstance(block, dict):
        return "CoolProp has no transport correlation for this fluid"
    hardcoded = hardcoded_in(block)
    if hardcoded is not None:
        return f"CoolProp uses a hardcoded correlation ({hardcoded})"
    if "reference_fluid" in block:
        return (
            "CoolProp uses extended corresponding states "
            f"(scaled from {block['reference_fluid']})"
        )
    return None


def resolve_viscosity(cp_name: str, cp_visc, *, molar_mass: float) -> Resolved:
    reason = _nonslot_reason(cp_visc)
    if reason is not None:
        return None, f"none [{reason}]"
    try:
        block = convert_slot_viscosity(cp_visc, molar_mass=molar_mass)
    except ValueError as e:
        return None, f"none [unimplemented slot form ({e})]"

    return block, "ok"


def resolve_conductivity(cp_name: str, cp_cond, draft: dict) -> Resolved:
    if draft.get("viscosity") is None:
        return None, "none [no viscosity for the critical enhancement]"
    reason = _nonslot_reason(cp_cond)
    if reason is not None:
        return None, f"none [{reason}]"
    eos = draft["eos"]
    try:
        block = convert_slot_conductivity(
            cp_cond,
            T_c=eos["T_red"],
            rho_c=eos["rho_red_mol"] * eos["molar_mass"],
            p_c_MPa=eos["P_red"] / 1e6,
        )
    except ValueError as e:
        return None, f"none [unimplemented slot form ({e})]"

    return block, "ok"


def _require_x64() -> None:
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "make_fluid needs JAX in 64-bit precision. Add at the top of your "
            "script:\nimport jax\njax.config.update('jax_enable_x64', True)"
        )


def _eos_block(cp: dict, cp_name: str, name: str) -> dict:
    consts, ideal, residual = convert_eos(cp, cp_name)
    block = dict(
        reference=(
            f"Converted from CoolProp {cp_name} (identical Helmholtz coefficients)."
        ),
        **consts,
        ideal=ideal,
        residual=residual,
    )
    # The critical point are computed and replace current critical values (cannot instantiate partially initialized cls)
    block.update(critical_constants(block))  # critical constants returns a dict
    logger.info(
        "%s critical point: T %.7f -> %.7f K (theta %.2e), rho %.6f -> %.6f kg/m3",
        name,
        consts["T_red"],
        block["T_crit"],
        1 - block["T_crit"] / consts["T_red"],
        consts["rho_red_mol"] * consts["molar_mass"],
        block["rho_crit_mol"] * consts["molar_mass"],
    )
    return block


def make_fluid(
    cp_name: str,
    name: Optional[str] = None,
    out_dir: Path | str = DATA_DIR,
    superancillary: bool = True,
    overwrite: bool = False,
    scaffold: bool = False,
) -> Path:
    _require_x64()
    name = (name or cp_name).lower().replace(" ", "")
    out_path = Path(out_dir) / f"{name}.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass overwrite=True (--force) to "
            "replace it, or --refit-superancillary to update only its curve."
        )
    cp = load_coolprop_fluid(cp_name)
    transport = cp.get("TRANSPORT", {})

    eos = _eos_block(cp, cp_name, name)
    definition = dict(
        name=name,
        cas=cp.get("INFO", {}).get("CAS"),
        eos=eos,
        viscosity=None,
        conductivity=None,
    )
    placeholders: list = []

    # viscosity first: the conductivity's critical enhancement reads it
    definition["viscosity"], provenance = resolve_viscosity(
        cp_name, transport.get("viscosity"), molar_mass=eos["molar_mass"]
    )
    if definition["viscosity"] is None and scaffold:
        definition["viscosity"], provenance = _scaffolded(
            "viscosity", transport.get("viscosity"), cp_name, name, eos, placeholders
        )
    logger.info("%s viscosity: %s", name, provenance)

    # a viscosity still carrying placeholders cannot be evaluated, and the
    # conductivity check evaluates it, must skip
    if placeholders:
        definition["conductivity"], provenance = _scaffolded(
            "conductivity",
            transport.get("conductivity"),
            cp_name,
            name,
            eos,
            placeholders,
        )
    else:
        definition["conductivity"], provenance = resolve_conductivity(
            cp_name, transport.get("conductivity"), definition
        )
        if definition["conductivity"] is None and scaffold:
            definition["conductivity"], provenance = _scaffolded(
                "conductivity",
                transport.get("conductivity"),
                cp_name,
                name,
                eos,
                placeholders,
            )
    logger.info("%s conductivity: %s", name, provenance)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(definition, indent=2) + "\n")
    logger.info("wrote %s", out_path)
    if placeholders:
        _report_placeholders(out_path, placeholders)

    return fit_superancillary(out_path) if superancillary else out_path


def fit_superancillary(path: Path | str) -> Path:
    _require_x64()
    path = Path(path)
    eos_block = json.loads(path.read_text())["eos"]
    eos_block.pop("superancillary", None)
    fit = from_eos(provisional_eos(eos_block))
    logger.info("%s superancillary: %s", path.stem, fit)
    return fit.save(path)


def _bespoke_slots(block: dict) -> dict:
    return {
        name: hardcoded_in(slot)
        for name, slot in block.items()
        if isinstance(slot, dict) and hardcoded_in(slot) is not None
    }


def _scaffold_property(
    prop: str, block, name: str, eos: dict
) -> Tuple[Optional[dict], list]:
    if not isinstance(block, dict):
        return None, [f"{prop}: CoolProp has no correlation for this fluid"]
    whole = hardcoded_here(block)
    if whole is not None:
        return None, [f"{prop}: wholly hardcoded ({whole}), no slot to transcribe"]

    bespoke = _bespoke_slots(block)
    if "dilute" in bespoke:
        return None, [
            f"{prop}: its dilute term is hardcoded, nothing stands without it"
        ]

    published = {k: v for k, v in block.items() if k not in bespoke}
    try:
        converted = _CONVERTERS[prop](
            published, eos["molar_mass"] if prop == "viscosity" else eos
        )
    except ValueError as e:
        return None, [f"{prop}: {e}"]

    source = citation(str(block.get("BibTeX", ""))) or block.get("BibTeX")
    notes = []
    for slot, hard in bespoke.items():
        converted[slot] = {"type": "custom", "name": f"{name}_{slot}", "params": {}}
        notes.append(
            f"{prop}.{slot}: hardcoded as {hard} -- coefficients from {source}"
        )
    return converted, notes


def _scaffolded(
    prop: str, cp_block, cp_name: str, name: str, eos: dict, placeholders: list
) -> Resolved:
    block, notes = _scaffold_property(prop, cp_block, name, eos)
    if block is None:
        return None, f"none [{notes[0].split(': ', 1)[-1]}]"

    slots = [
        k for k, v in block.items() if isinstance(v, dict) and v.get("type") == "custom"
    ]

    block["reference"] = (
        f"Scaffold of CoolProp {cp_name} {prop}: published slots transcribed "
        f"exactly, {', '.join(slots)} left to the term named in it."
        if slots
        else f"Exact conversion of CoolProp {cp_name} {prop}."
    )
    for note in notes:
        logger.info("  %s", note)
    placeholders.extend((prop, block[s]["name"]) for s in slots)
    if not slots:
        return block, "transcribed, unverified (its viscosity is still a scaffold)"
    return block, f"scaffold ({len(slots)} slot(s) awaiting a registered term)"


def _report_placeholders(out_path: Path, placeholders: list) -> None:
    for prop, term in placeholders:
        logger.info("  unfilled: %s('%s')", _REGISTER[prop], term)
    logger.info(
        "%s will not load until those %d term(s) are registered; "
        "loading it unregistered names the one it wants and how to write it",
        out_path.name,
        len(placeholders),
    )
