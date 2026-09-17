import argparse
import logging
import os
import shutil
import textwrap
from pathlib import Path
from typing import Optional

from .core.config import table_spec
from .utils.build_utils import (
    EQS_REGISTRY,
    build_adaptive_table,
    get_table_path,
    logger,
)


def create_all_tables(
    fluid_name: str,
    pairs: list,
    bounds: dict,
    target: float = 1e-5,
    tables_base_path: Optional[Path] = None,
):
    """
    Build the tables of `pairs` for a fluid, over `bounds` ({variable: (lo, hi)},
    one entry per variable of the pairs), refined until the measured
    interpolation error meets `target`.

    Please note a table does not have to meet high precision requirements, by
    design it is more suited for typical float32 compatible tolerances (~1e-5
    to ~1e-7 rtols).
    """
    if tables_base_path is None:
        tables_base_path = get_table_path()
    # every request is checked before the first, long, build starts
    specs = [table_spec(pair, bounds) for pair in pairs]

    failures: list = []
    for table_cfg in specs:
        try:
            build_adaptive_table(fluid_name, table_cfg, tables_base_path, target=target)
        except Exception as e:
            logger.error(f"  ERROR building {table_cfg.name}: {e}", exc_info=True)
            failures.append((table_cfg.name, e))

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise RuntimeError(
            f"{len(failures)} table(s) of {fluid_name} failed to build: {names}. "
            "The fluid is on disk but incomplete; see the logged tracebacks."
        )


def cached_fluids(tables_base_path: Optional[Path] = None) -> dict:
    base = tables_base_path or get_table_path()
    if not base.exists():
        return {}
    out = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        tables = sorted(t.name for t in d.iterdir() if t.is_dir())
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        out[d.name] = (tables, size)
    return out


def remove_fluid(fluid_name: str, tables_base_path: Optional[Path] = None) -> int:
    base = tables_base_path or get_table_path()
    target = base / fluid_name.lower().replace(" ", "")
    if not target.is_dir():
        raise FileNotFoundError(f"{target} is not in the cache")
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    shutil.rmtree(target)
    return size


def _megabytes(n: int) -> str:
    return f"{n / 1e6:.0f} MB"


def show_cache(base: Path) -> int:
    cache = cached_fluids(base)
    if not cache:
        print(f"cache empty ({base})")
        return 0
    total = sum(size for _, size in cache.values())
    print(f"{base}    {len(cache)} fluid(s), {_megabytes(total)}\n")
    for name, (tables, size) in cache.items():
        print(f"  {name:16s} {size / 1e6:8.1f} MB  {', '.join(tables) or '(empty)'}")
    return 0


def clear_cache(base: Path, assume_yes: bool) -> int:
    cache = cached_fluids(base)
    if not cache:
        print(f"cache already empty ({base})")
        return 0
    total = sum(size for _, size in cache.values())
    question = f"delete {len(cache)} fluid(s), {_megabytes(total)}, from {base}? [y/N] "
    if not assume_yes and not _confirmed(question):
        print("aborted")
        return 1
    shutil.rmtree(base)
    print(f"removed {_megabytes(total)}")
    return 0


def _confirmed(question: str) -> bool:
    try:
        return input(question).strip().lower() == "y"
    except EOFError:  # not answered is a no
        return False


def remove_fluids(names: list, base: Path) -> int:
    status = 0
    for name in names:
        try:
            freed = remove_fluid(name, base)
        except FileNotFoundError as e:
            print(f"{name}: {e}")
            status = 1
            continue
        print(f"removed {name} ({_megabytes(freed)})")
    return status


def build_fluids(
    names: list, base: Path, pairs: list, bounds: dict, target: float
) -> int:
    incomplete = []
    for fluid in names:
        try:
            create_all_tables(fluid, pairs, bounds, target, tables_base_path=base)
        except Exception as e:
            logger.error(f"{fluid}: {e}")
            incomplete.append(fluid)

    if incomplete:
        print(f"\nINCOMPLETE: {len(incomplete)} fluid(s) -> {', '.join(incomplete)}")
        return 1
    print(f"\nbuilt {len(names)} fluid(s) into {base}")
    return 0


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m propax.build_tables",
        description=textwrap.dedent("""
            Build the interpolation tables a fluid needs, into the propax cache
            (PROPAX_TABLE_DIR, else ~/.cache/propax).

            Tables have no default range: name the pairs you use with --pair,
            and the bounds of each of their variables with --bounds. Outside
            them `fast_flash` returns NaN.

            A table is a few tens to a few hundred MB, so build only the fluids
            you actually use  there is no reason to hold all covered fluids.
            Nodes outside the physical domain are extrapolated from their solved
            neighbours.

            example: python -m propax.build_tables n-propane --pair P H \\
                         --bounds P=1e4:5e6 H=-3e5:1.5e6
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fluids", nargs="*", help="fluid names, e.g. hydrogen argon")

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="show what the cache holds")
    action.add_argument(
        "--remove", metavar="FLUID", nargs="+", help="delete these from the cache"
    )
    action.add_argument("--clear", action="store_true", help="delete the whole cache")

    parser.add_argument(
        "--yes", action="store_true", help="answer yes to --clear's confirmation"
    )
    parser.add_argument(
        "--table-dir", metavar="DIR", help="override the cache location"
    )
    parser.add_argument(
        "--chunk-size",
        type=_positive_int,
        metavar="N",
        help="map at most N nodes at a time instead of the whole grid; lowers "
        "peak memory at some cost in speed (try 1024 if a build is swapping)",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="a pair to tabulate, e.g. --pair P H; repeat it for several",
    )
    parser.add_argument(
        "--bounds",
        nargs="+",
        default=[],
        metavar="VAR=LO:HI",
        help="the range of each variable of the pairs, e.g. P=1e4:5e6 H=-3e5:1.5e6",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=1e-5,
        metavar="ERR",
        help="refine the axes until the measured interpolation error meets ERR "
        "(default 1e-5)",
    )
    return parser


def _bounds(args, parser: argparse.ArgumentParser) -> dict:
    """{variable: (lo, hi)} from the VAR=LO:HI items of --bounds."""
    bounds = {}
    for item in args.bounds:
        try:
            var, ends = item.split("=")
            lo, hi = ends.split(":")
            bounds[var] = (float(lo), float(hi))
        except ValueError:
            parser.error(f"--bounds takes VAR=LO:HI items, got {item!r}")
    return bounds


def _apply_settings(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.chunk_size is not None:
        os.environ["PROPAX_CHUNK_SIZE"] = str(args.chunk_size)


def _targets(args, parser: argparse.ArgumentParser) -> list:
    if not args.fluids:
        parser.error("name at least one fluid, or pass --list / --remove / --clear")
    if not args.pair:
        parser.error("name the pairs to tabulate with --pair, e.g. --pair P H")
    names = [f.lower().replace(" ", "") for f in args.fluids]
    unknown = [f for f in names if f not in EQS_REGISTRY]
    if unknown:
        parser.error(
            f"unknown fluid(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(EQS_REGISTRY))}"
        )
    return names


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_settings(args)
    base = Path(args.table_dir) if args.table_dir else get_table_path()

    if args.list:
        return show_cache(base)
    if args.clear:
        return clear_cache(base, args.yes)
    if args.remove:
        return remove_fluids(args.remove, base)
    names = _targets(args, parser)
    return build_fluids(names, base, args.pair, _bounds(args, parser), args.target)


if __name__ == "__main__":
    raise SystemExit(main())
