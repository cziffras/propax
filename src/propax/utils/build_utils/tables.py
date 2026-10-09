import shutil
from typing import Optional

from ...core.config import get_table_path, table_spec
from .assemble import build_adaptive_table
from .helpers import logger


def create_all_tables(
    fluid_name: str,
    pairs: list,
    bounds: dict,
    target: float = 1e-5,
):
    """
    Build the tables of `pairs` for a fluid, over `bounds` ({variable: (lo, hi)},
    one entry per variable of the pairs), refined until the measured
    interpolation error meets `target`.

    Please note a table does not have to meet high precision requirements, by
    design it is more suited for typical float32 compatible tolerances (~1e-5
    to ~1e-7 rtols).
    """
    specs = [table_spec(pair, bounds) for pair in pairs]

    failures: list = []
    for table_cfg in specs:
        try:
            build_adaptive_table(fluid_name, table_cfg, target=target)
        except Exception as e:
            logger.error(f"  ERROR building {table_cfg.name}: {e}", exc_info=True)
            failures.append((table_cfg.name, e))

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise RuntimeError(
            f"{len(failures)} table(s) of {fluid_name} failed to build: {names}. "
            "The fluid is on disk but incomplete; see the logged tracebacks."
        )


def cached_fluids() -> dict:
    base = get_table_path()
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


def clear_cache(names: Optional[list] = None) -> int:
    base = get_table_path()
    cache = cached_fluids()
    status = 0
    if not cache:
        print(f"cache already empty ({base})")
        return 0
    if names is None:
        print("removing all fluids from cache")
        total = sum(size for _, size in cache.values())
        shutil.rmtree(base)
        print(f"removed {total / 1e6:.2f} MB of data")
    else:
        for name in names:
            try:
                target = base / name.lower().replace(" ", "")
                freed = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                shutil.rmtree(target)
            except FileNotFoundError as e:
                print(f"{name}: {e}")
                status = 1
                continue
            print(f"removed {name} ({freed / 1e6:.2f} MB of data)")
    return status
