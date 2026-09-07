"""
Generate a propax fluid definition file from an installed CoolProp fluid.

NOTE : CoolProp is a dev-time dependency: pip install -e ".[coolprop]".
After generating a file, build its tables with `python -m propax.build_tables`.
"""

import argparse
import logging
from pathlib import Path

import jax

from .fluids.schema import DATA_DIR
from .utils.make_utils import fit_superancillary, make_fluid
from .utils.make_utils.coverage import convertible_fluids


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fluid", nargs="?", help="CoolProp fluid name, e.g. Argon")
    parser.add_argument(
        "--list-convertible",
        action="store_true",
        help="print the fluids this converter accepts, one per line, and exit. "
        "Generating the catalogue is then whatever your machine can afford: "
        "`python -m propax.make_fluid --list-convertible | xargs -P 4 -n 1 "
        "python -m propax.make_fluid`",
    )
    parser.add_argument(
        "--name", help="propax fluid name (default: lowercased CoolProp name)"
    )
    parser.add_argument("--out", default=str(DATA_DIR), help="output directory")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        metavar="ERR",
        help="the bar every transport transcription must clear (default 1e-6); "
        "a correlation that misses it is omitted rather than approximated",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing definition file; without it the run refuses, "
        "because a shipped file may carry hand-written content the converter "
        "cannot reproduce",
    )

    parser.add_argument(
        "--transport-scaffold",
        action="store_true",
        help="keep a transport property CoolProp only half publishes: its "
        "transcribable slots go into the fluid file as usual, and every slot "
        "CoolProp hardcodes becomes a `custom` placeholder naming the term you "
        "have to register, with the paper to read it from. Without this the "
        "property is dropped instead",
    )

    curve = parser.add_mutually_exclusive_group()
    curve.add_argument(
        "--no-superancillary",
        action="store_true",
        help="skip the saturation fit, which dominates the run",
    )
    curve.add_argument(
        "--refit-superancillary",
        action="store_true",
        help="fit the saturation curve of an existing file, and nothing else",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    jax.config.update("jax_enable_x64", True)

    if args.list_convertible:
        print("\n".join(convertible_fluids()))
        return 0
    if args.fluid is None:
        _build_parser().error("name a fluid, or pass --list-convertible")

    name = (args.name or args.fluid).lower().replace(" ", "")
    try:
        if args.refit_superancillary:
            fit_superancillary(Path(args.out) / f"{name}.json")
        else:
            make_fluid(
                args.fluid,
                name=args.name,
                out_dir=args.out,
                tol=args.tol,
                superancillary=not args.no_superancillary,
                overwrite=args.force,
                scaffold=args.transport_scaffold,
            )
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        print(f"{args.fluid}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
