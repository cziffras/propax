"""
CoolProp access layer for the fluid converter
"""

import json
import re
from pathlib import Path

_BIB_FIELDS = ("author", "title", "journal", "year", "doi", "comment")


def require_coolprop():
    try:
        import CoolProp.CoolProp as CP
    except ImportError as e:
        raise SystemExit(
            "propax.make_fluid needs CoolProp (dev-time only): "
            'pip install -e ".[coolprop]"'
        ) from e
    return CP


def load_coolprop_fluid(cp_name: str) -> dict:
    """Full CoolProp fluid definition (EOS / TRANSPORT / INFO blocks)."""
    CP = require_coolprop()
    cp = json.loads(CP.get_fluid_param_string(cp_name, "JSON"))
    return cp[0] if isinstance(cp, list) else cp


def hardcoded_in(block) -> str | None:
    """The name of the hardcoded correlation inside a transport block, if any.

    CoolProp marks these with a `hardcoded` key and no coefficients: the model
    lives in its C++ and there is nothing in the JSON to transcribe. The mark
    sits at the top of the block for a wholly hardcoded property, and on a
    single slot when only one term is (hydrogen's higher-order viscosity). A
    slot carrying it has no `type` either, so a scan keyed on `type` walks
    straight past it and the property looks covered by the slots that remain.
    """

    if not isinstance(block, dict):
        return None
    named = hardcoded_here(block)
    if named is not None:
        return named
    for name, slot in block.items():
        named = hardcoded_here(slot)
        if named is not None:
            return f"{named}, in its {name} term"
    return None


def hardcoded_here(block) -> str | None:
    if not isinstance(block, dict) or "hardcoded" not in block:
        return None
    name = str(block["hardcoded"])
    return None if name == "None" else name


def citation(bibtex_key: str) -> str | None:
    """
    CoolProp Bibtex key, flattened.

    Transforms :
    @article{Lemmon2006,
        author = {Lemmon, Eric W. and Jacobsen, Richard T.},
        title = {A New Functional Form for Equations of State},
        journal = {Journal of Physical and Chemical Reference Data},
        year = {2006}
    }

    Into :

    author: Lemmon, Eric W. and Jacobsen, Richard T. | title: A New Functional Form for Equations of State | journal: Journal of Physical and Chemical Reference Data | year: 2006
    """
    CP = require_coolprop()
    lib = Path(CP.__file__).parent / "CoolPropBibTeXLibrary.bib"
    if not bibtex_key or not lib.exists():
        return None
    text = lib.read_text(errors="replace")
    start = text.find("{" + bibtex_key + ",")
    if start < 0:
        return None

    depth, end = 0, start
    for end in range(
        start, len(text)
    ):  # a "{" adds 1 and "}" substracts 1, at the end of the nested accolade depth == 0
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            break
    entry = text[start:end]

    out = []
    for field in _BIB_FIELDS:
        m = re.search(rf"^\s*{field}\s*=\s*[{{\"](.*?)[}}\"],?\s*$", entry, re.M | re.I)
        if m:
            out.append(f"{field}: {m.group(1).strip('{} ')}")
    return " | ".join(out) if out else None
