"""
Dev-time audit of how much of the CoolProp catalogue propax can represent.

For every CoolProp fluid this reports, per model (EOS / viscosity /
conductivity), whether propax can evaluate it from *published* coefficients
(no refit, no new runtime physics), and if not, which functional family is the
blocker. It is an analysis tool, not part of the runtime; it needs the
`[coolprop]` extra (`pip install -e ".[coolprop]"`).

Standard CoolProp API used (all public):
  get_global_param_string("fluids_list")   -> canonical fluid names
  get_fluid_param_string(fluid, "JSON")    -> full definition: EOS.alpha0/alphar
        (term "type" tags) and TRANSPORT.{viscosity,conductivity}
  get_BibTeXKey(fluid, key)                -> reference key (not used here)

Caveat: the *shape* of that JSON is CoolProp's internal fluid-file schema, not
a documented-stable contract. It is parsed defensively and the CoolProp
version is printed with every summary.

The "supported" sets are the ground truth from propax itself: the EOS term
handlers registered in `make_utils.eos`, and the transport sub-models that
propax's built-in fluid (hydrogen) uses. EOS support is reported in tiers,
distinguishing a genuine physics gap (an unsupported *residual* term) from a
mere converter-registration gap (an ideal-gas term whose math propax already
evaluates under a different CoolProp type name, e.g. PlanckEinstein).

    python -m propax.make_utils.coverage      # print the summary
    from propax.make_utils.coverage import coverage
    rows = coverage()                          # list of per-fluid dicts
"""

import json
from collections import Counter

import CoolProp
from CoolProp.CoolProp import get_fluid_param_string, get_global_param_string

# Read straight from the converter's registries rather than duplicated here:
# these sets used to be a hand-kept copy and had already drifted once, after
# handlers were added. Now the audit cannot disagree with the code.
from .eos import IDEAL_BLOCK_HANDLERS, RESIDUAL_BLOCK_HANDLERS, convert_eos
from .source import hardcoded_in, load_coolprop_fluid

EOS_IDEAL_NATIVE = set(IDEAL_BLOCK_HANDLERS)
# residual is the real physics gate.
EOS_RESID_OK = set(RESIDUAL_BLOCK_HANDLERS)
# ideal gas given as Cp0(T) (or a generalized Einstein term): needs the
# standard Cp0 -> alpha0 integration, but no new runtime physics.
EOS_IDEAL_CP0 = {
    "IdealGasHelmholtzCP0PolyT",
    "IdealGasHelmholtzCP0Constant",
    "IdealGasHelmholtzCP0AlyLee",
    "IdealGasHelmholtzPlanckEinsteinGeneralized",
} - EOS_IDEAL_NATIVE

# - transport sub-model tags propax can evaluate natively ----------------
# Read from the slot converters, for the same reason as the EOS handlers above:
# the hand-kept copy that used to sit here had drifted, and reported three
# viscosity and two conductivity forms as blocking that `convert_slot_*` in
# fact transcribes  understating viscosity coverage by a factor four.
from .conductivity import (  # noqa: E402
    SLOT_COND_CRITICAL,
    SLOT_COND_DILUTE,
    SLOT_COND_RESIDUAL,
)
from .viscosity import SLOT_DILUTE, SLOT_HIGHER, SLOT_INITIAL  # noqa: E402

TRANSPORT_OK = {
    "viscosity": (
        {f"dilute:{t}" for t in SLOT_DILUTE}
        | {f"initial_density:{t}" for t in SLOT_INITIAL}
        | {f"higher_order:{t}" for t in SLOT_HIGHER}
    ),
    "conductivity": (
        {f"dilute:{t}" for t in SLOT_COND_DILUTE}
        | {f"residual:{t}" for t in SLOT_COND_RESIDUAL}
        | {f"critical:{t}" for t in SLOT_COND_CRITICAL}
    ),
}


def _transport_subtypes(block):
    if not isinstance(block, dict):
        return ("none", set())
    if hardcoded_in(block) is not None:
        return ("hardcoded", set())
    if block.get("type"):
        return ("ok", {f"top:{block['type']}"})
    subs = {
        f"{k}:{v['type']}"
        for k, v in block.items()
        if isinstance(v, dict) and "type" in v
    }
    return ("ok", subs) if subs else ("untyped", set())


def _classify_transport(prop, block):
    kind, subs = _transport_subtypes(block)
    if kind == "none":
        return ("no-data", set())
    if kind in ("hardcoded", "untyped"):
        return (kind, set())
    missing = subs - TRANSPORT_OK[prop]
    return ("covered" if not missing else "partial", missing)


def _classify_eos(eos):
    """-> (tier, blocking_term_types). tier in
    native / cp0-ideal / ideal-other / residual-blocked / no-data."""
    if not isinstance(eos, dict):
        return ("no-data", set())
    a0 = {t.get("type") for t in eos.get("alpha0", []) if isinstance(t, dict)}
    ar = {t.get("type") for t in eos.get("alphar", []) if isinstance(t, dict)}
    if not ar:
        return ("no-data", set())
    resid_bad = ar - EOS_RESID_OK
    if resid_bad:  # a real physics gap dominates
        return ("residual-blocked", resid_bad)
    ideal_extra = a0 - EOS_IDEAL_NATIVE
    if not ideal_extra:
        return ("native", set())
    if ideal_extra <= EOS_IDEAL_CP0:
        return ("cp0-ideal", ideal_extra)
    return ("ideal-other", ideal_extra)


def coverage():
    """Return one dict per CoolProp fluid with the EOS / viscosity /
    conductivity classification and the blocking families for each."""
    rows = []
    for fl in get_global_param_string("fluids_list").split(","):
        try:
            j = json.loads(get_fluid_param_string(fl, "JSON"))[0]
        except Exception:
            continue
        tr = j.get("TRANSPORT", {})
        e_st, e_miss = _classify_eos((j.get("EOS") or [{}])[0])
        v_st, v_miss = _classify_transport("viscosity", tr.get("viscosity"))
        c_st, c_miss = _classify_transport("conductivity", tr.get("conductivity"))
        rows.append(
            dict(
                fluid=fl,
                eos=e_st,
                eos_missing=e_miss,
                visc=v_st,
                visc_missing=v_miss,
                cond=c_st,
                cond_missing=c_miss,
            )
        )
    return rows


_EOS_ORDER = ("native", "cp0-ideal", "ideal-other", "residual-blocked", "no-data")
_TRANSPORT_ORDER = ("covered", "partial", "hardcoded", "untyped", "no-data")


def tally(rows, key: str, order) -> Counter:
    """How many fluids fall in each status, for one of the three axes."""
    return Counter(r[key] for r in rows if r[key] in order)


def blockers(rows, status_key: str, miss_key: str, status: str) -> list:
    """Which families block that status, as (family, fluids, solely-blocked).

    `solely` counts the fluids this family is the *only* thing missing from --
    the ones a single new handler would unlock. Most common first.
    """
    need, solely = Counter(), Counter()
    for r in rows:
        if r[status_key] != status:
            continue
        need.update(r[miss_key])
        if len(r[miss_key]) == 1:
            solely.update(r[miss_key])
    return [(m, n, solely[m]) for m, n in need.most_common()]


def _status_line(label: str, counts: Counter, order) -> str:
    return f"== {label} ==   " + "  ".join(
        f"{k}:{counts[k]}" for k in order if counts.get(k)
    )


def _blocker_lines(entries) -> list:
    return [f"      {m:44s} {n:4d}   ({solo} solely-blocked)" for m, n, solo in entries]


def report(rows=None) -> str:
    """The coverage summary as text, so it can be tested or written to a file."""
    rows = coverage() if rows is None else rows
    out = [f"CoolProp {CoolProp.__version__}  |  {len(rows)} fluids", ""]

    out.append(_status_line("EOS", tally(rows, "eos", _EOS_ORDER), _EOS_ORDER))
    out.append("   residual terms that genuinely block (real physics gap):")
    out += _blocker_lines(blockers(rows, "eos", "eos_missing", "residual-blocked"))
    out.append("")

    for axis, label in (("visc", "VISCOSITY"), ("cond", "CONDUCTIVITY")):
        counts = tally(rows, axis, _TRANSPORT_ORDER)
        out.append(_status_line(label, counts, _TRANSPORT_ORDER))
        out += _blocker_lines(blockers(rows, axis, f"{axis}_missing", "partial"))
        out.append("")

    native = {r["fluid"] for r in rows if r["eos"] == "native"}
    with_cp0 = native | {r["fluid"] for r in rows if r["eos"] == "cp0-ideal"}
    transport = {
        r["fluid"] for r in rows if r["visc"] == "covered" and r["cond"] == "covered"
    }

    out += [
        "EOS reachable (density/energy/cp/cv only, no transport):",
        f"   native (handlers as registered)     : {len(native)}",
        f"   + Cp0->alpha0 integration (moderate): {len(with_cp0)}",
        "",
        "End-to-end usable (EOS AND viscosity AND conductivity):",
        f"   native EOS : {len(native & transport)} -> "
        f"{', '.join(sorted(native & transport))}",
        f"   transport-covered but EOS hard-blocked: "
        f"{', '.join(sorted(transport - with_cp0))}",
    ]
    return "\n".join(out)


def summarize(rows=None) -> None:
    print(report(rows))


def convertible_fluids() -> list:
    names = []
    for name in get_global_param_string("fluids_list").split(","):
        try:
            convert_eos(load_coolprop_fluid(name), name)
        except Exception:
            continue
        names.append(name)
    return names


if __name__ == "__main__":
    summarize()
