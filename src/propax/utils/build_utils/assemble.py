import logging
import warnings
from pathlib import Path
from typing import Optional

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

from ...core.config import (  # noqa: E402
    TABLE_REGISTRY,
    TableSpec,
    ThermoVar,
    get_table_path,
)
from ...core.interp import BicubicInterpolation  # noqa: E402
from .dome import (  # noqa: E402
    _dome_derivatives,
    _dome_values,
    _solve_dome_state,
    dome_axis_bounds,
)
from .helpers import (  # noqa: E402
    _get_fluid_modules,
    _make_input_fn,
    _make_outputs_fn,
    _mapped,
    fill_ghost,
)
from .single_phase import (  # noqa: E402
    make_single_phase_solver,
    solve_grid,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _analytic_first_derivatives(f1, f2, outputs_fn, rho_flat, T_flat):
    """
    ``d(out)/d(x1)|x2 and d(out)/d(x2)|x1`` at a node whose (rho, T) is already known

    'out' being a function that returns ``(f1(rho, T), f2(rho, T)) = (x1, x2)`` (input variables
    are obviously output values of the EOS)

    this method computes ``d(out)/d(x1)|x2`` and ``d(out)/d(x2)|x1`` at single-phase nodes
    via the implicit function theorem:

    ``J_out(rho, T) (d(out)/d(rho, T)) @ inv(J_inputs(rho, T)) ((d(x1, x2)/d(rho, T)^-1))``

    The chain rule yields:

    ``d(out)/d(x1, x2) = d(out)/d(rho, T) * d(rho, T)/d(x1, x2) (= d(x1, x2)/d(rho, T)^-1))``

    NOTE :
    - Both Jacobians come from `jacfwd`, so the tangents are exact to machine
    precision. A Hermite spline is extremely sensitive to the precision of its
    first order derivatives nodes.
    - Single-phase only. Inside the dome (rho, T) does not describe the state and
    J_in is meaningless thus `_dome_derivatives` applies the same theorem through
    (T_sat, x) instead.
    """

    def node(rho, T):
        y = jnp.stack([rho, T])
        J_in = jax.jacfwd(lambda z: jnp.stack([f1(z[0], z[1]), f2(z[0], z[1])]))(y)
        J_out = jax.jacfwd(lambda z: outputs_fn(z[0], z[1]))(y)
        return J_out @ jnp.linalg.inv(J_in)  # (C, 2)

    return _mapped(node, rho_flat, T_flat)


def make_field_ctx(table_config, eos, viscosity, conductivity, saturation, fluid_name):
    x1, x2 = table_config.x_axis.variable, table_config.y_axis.variable
    ctx = {
        "f1": _make_input_fn(eos, x1),
        "f2": _make_input_fn(eos, x2),
        "outputs_fn": _make_outputs_fn(
            eos, viscosity, conductivity, table_config.outputs
        ),
    }
    ctx["outputs_vfn"] = jax.jit(jax.vmap(ctx["outputs_fn"]))
    # every pair needs the saturation module now, not only those holding a
    # natural variable: the (P, x) solve reads T_sat(P) to split its isobar
    ctx["sat_mod"] = saturation
    ctx["solver"] = make_single_phase_solver(eos, saturation, table_config)
    return ctx


def _all_finite(*fields) -> np.ndarray:
    """Nodes whose EVERY channel of EVERY field is finite.
    Without it `fill_ghost` extrapolates a neighbour from a potentially infinite
    derivative and corrupts the output of finite fields.
    """
    ok = np.ones(fields[0].shape[:2], dtype=bool)
    for f in fields:
        ok &= np.isfinite(f).all(axis=2)
    return ok


def _grid_axis(lo: float, hi: float, log: bool, n: int) -> np.ndarray:
    """`n` evenly spaced nodes in the axis' internal coordinate"""
    return np.linspace(np.log(lo) if log else lo, np.log(hi) if log else hi, n)


def _save_table(
    save_dir: Path,
    table_config,
    arr_x1,
    arr_x2,
    f,
    dx1,
    dx2,
    dx1dx2,
    unreachable,
    output_names=None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "table_data.npz"
    np.savez(
        save_path,
        arr_x1=arr_x1,
        arr_x2=arr_x2,
        f=f,
        dx1=dx1,
        dx2=dx2,
        dx1dx2=dx1dx2,
        unreachable=unreachable,
        is_log_x1=table_config.x_axis.spacing == "log",
        is_log_x2=table_config.y_axis.spacing == "log",
        output_names=np.array(
            [out.value for out in (output_names or table_config.outputs)]
        ),
        x_name=table_config.x_axis.variable.value,
        y_name=table_config.y_axis.variable.value,
    )
    logger.info(f"  Saved to {save_path}")
    return save_path


def solve_field(
    table_config: TableSpec,
    X1_phys,
    X2_phys,
    eos,
    viscosity,
    conductivity,
    saturation,
    fluid_name: str,
    warn_unconverged: bool = True,
    ctx: Optional[dict] = None,
):
    """Single-phase nodal values on an arbitrary (X1, X2) grid, no derivatives.

    Returns (table_f, rho_grid, T_grid, ok_grid).

    The single-phase branch is kept everywhere it exists, metastable extension
    included. Those dome states sit below the EOS surface and are physically
    spurious, but that branch alone has no kink.

    The equilibrium values live in their own table (`build_dome_table`).
    """
    outputs = table_config.outputs
    n1, n2 = X1_phys.shape
    if ctx is None:
        ctx = make_field_ctx(
            table_config, eos, viscosity, conductivity, saturation, fluid_name
        )

    rho_grid, T_grid, ok_grid = solve_grid(ctx["solver"], X1_phys, X2_phys)

    flat_vals = _mapped(
        ctx["outputs_fn"], rho_grid.ravel(), T_grid.ravel(), vfn=ctx["outputs_vfn"]
    )
    table_f = np.array(flat_vals).reshape(n1, n2, len(outputs))

    bad = ~ok_grid
    if bad.any() and warn_unconverged:
        warnings.warn(
            f"{int(bad.sum())} nodes of {table_config.name} have no converged "
            "solution (out-of-domain corners are expected); values are filled "
            "from the nearest valid node."
        )
    x1_flat, x2_flat = X1_phys.ravel(), X2_phys.ravel()
    for idx in range(len(outputs)):
        channel = table_f[:, :, idx]
        channel[bad] = np.nan
        flat = channel.ravel()
        if (~np.isfinite(flat)).any():
            flat = fill_nan_nearest(flat, x1_flat, x2_flat)
        table_f[:, :, idx] = flat.reshape(n1, n2)

    return table_f, rho_grid, T_grid, ok_grid


def build_vectorized_table(
    fluid_name: str,
    table_config: TableSpec,
    tables_base_path: Optional[Path] = None,
    axes: Optional[tuple] = None,
    ctx: Optional[dict] = None,
) -> Path:
    if tables_base_path is None:
        tables_base_path = get_table_path()

    processed_fluid_name, eos, viscosity, conductivity, saturation = _get_fluid_modules(
        fluid_name
    )

    x1_cfg, x2_cfg = table_config.x_axis, table_config.y_axis
    outputs = table_config.outputs
    n1, n2 = x1_cfg.n_points, x2_cfg.n_points

    apply_log_x1 = x1_cfg.spacing == "log"
    apply_log_x2 = x2_cfg.spacing == "log"

    if axes is None:
        arr_x1_internal = _grid_axis(x1_cfg.min_val, x1_cfg.max_val, apply_log_x1, n1)
        arr_x2_internal = _grid_axis(x2_cfg.min_val, x2_cfg.max_val, apply_log_x2, n2)
    else:
        # explicit (possibly graded) axes, in internal space
        arr_x1_internal, arr_x2_internal = (np.asarray(a) for a in axes)
        n1, n2 = arr_x1_internal.size, arr_x2_internal.size
    arr_x1_phys = np.exp(arr_x1_internal) if apply_log_x1 else arr_x1_internal
    arr_x2_phys = np.exp(arr_x2_internal) if apply_log_x2 else arr_x2_internal
    X1_phys, X2_phys = np.meshgrid(arr_x1_phys, arr_x2_phys, indexing="ij")

    f1 = _make_input_fn(eos, x1_cfg.variable)
    f2 = _make_input_fn(eos, x2_cfg.variable)
    outputs_fn = _make_outputs_fn(eos, viscosity, conductivity, outputs)

    logger.info(f"Building {table_config.name} for {fluid_name} ({n1}x{n2})...")

    table_f, rho_grid, T_grid, ok_grid = solve_field(
        table_config,
        X1_phys,
        X2_phys,
        eos,
        viscosity,
        conductivity,
        saturation,
        processed_fluid_name,
        ctx=ctx,
    )
    logger.info(f"  single-phase converged: {ok_grid.mean():.1%}")

    # derivatives: analytic (implicit function theorem) in the single
    # phase, finite differences across the dome and at repaired nodes
    table_dx1 = np.zeros((n1, n2, len(outputs)))
    table_dx2 = np.zeros((n1, n2, len(outputs)))
    table_dx1dx2 = np.zeros((n1, n2, len(outputs)))

    derivs = np.array(
        _analytic_first_derivatives(
            f1, f2, outputs_fn, rho_grid.ravel(), T_grid.ravel()
        )
    ).reshape(n1, n2, len(outputs), 2)

    for idx in range(len(outputs)):
        dx1_phys = derivs[:, :, idx, 0]
        dx2_phys = derivs[:, :, idx, 1]
        table_dx1[:, :, idx] = dx1_phys * X1_phys if apply_log_x1 else dx1_phys
        table_dx2[:, :, idx] = dx2_phys * X2_phys if apply_log_x2 else dx2_phys
        table_dx1dx2[:, :, idx] = np.gradient(
            table_dx1[:, :, idx], arr_x2_internal, axis=1
        )

    reachable = ok_grid & _all_finite(table_f, table_dx1, table_dx2, table_dx1dx2)
    table_f, table_dx1, table_dx2, table_dx1dx2, reachable = fill_ghost(
        arr_x1_internal,
        arr_x2_internal,
        table_f,
        table_dx1,
        table_dx2,
        table_dx1dx2,
        reachable,
    )

    return _save_table(
        tables_base_path / processed_fluid_name / table_config.name,
        table_config,
        arr_x1_internal,
        arr_x2_internal,
        table_f,
        table_dx1,
        table_dx2,
        table_dx1dx2,
        ~reachable,
    )


def build_dome_table(
    fluid_name: str,
    table_config: TableSpec,
    tables_base_path: Optional[Path] = None,
    axes: Optional[tuple] = None,
) -> Path:
    """Mixture branch of a pair, on axes fitted to the two-phase region.

    The lever rule is evaluated with the quality left unclamped, so the surface
    continues smoothly past the saturation line and no cell interpolates across
    it. Its own axes matter: sharing the single-phase domain would spend the
    resolution on states this table never answers for.
    """
    if tables_base_path is None:
        tables_base_path = get_table_path()

    processed_fluid_name, eos, viscosity, conductivity, saturation = _get_fluid_modules(
        fluid_name
    )
    x1_cfg, x2_cfg = table_config.x_axis, table_config.y_axis
    # quality last: the runtime reads it to tell which branch answers, which
    # keeps `fast_flash` a pure lookup instead of a saturation solve
    outputs = list(table_config.outputs) + [ThermoVar.Q]
    log1, log2 = x1_cfg.spacing == "log", x2_cfg.spacing == "log"

    if axes is None:
        (x1_lo, x1_hi), (x2_lo, x2_hi) = dome_axis_bounds(table_config, saturation)
        arr_x1 = _grid_axis(x1_lo, x1_hi, log1, x1_cfg.n_points)
        arr_x2 = _grid_axis(x2_lo, x2_hi, log2, x2_cfg.n_points)
    else:
        arr_x1, arr_x2 = (np.asarray(a) for a in axes)
    n1, n2 = arr_x1.size, arr_x2.size

    X1 = np.exp(arr_x1) if log1 else arr_x1
    X2 = np.exp(arr_x2) if log2 else arr_x2
    X1_phys, X2_phys = np.meshgrid(X1, X2, indexing="ij")

    name = f"{table_config.name}_dome"
    logger.info(f"Building {name} for {fluid_name} ({n1}x{n2})...")

    _, T_sat, x_q = _solve_dome_state(table_config, X1_phys, X2_phys, saturation)
    solved = np.isfinite(T_sat) & np.isfinite(x_q)
    logger.info(
        f"  mixture solved: {solved.mean():.1%}, "
        f"quality outside [0, 1]: {np.mean((x_q < 0) | (x_q > 1)):.1%}"
    )

    values = _dome_values(outputs, saturation, T_sat, x_q)
    table_f = np.stack(
        [np.asarray(values[var]).reshape(n1, n2) for var in outputs], axis=-1
    )

    derivs = np.asarray(
        _dome_derivatives(
            table_config,
            outputs,
            saturation,
            jnp.asarray(np.nan_to_num(T_sat).ravel()),
            jnp.asarray(np.nan_to_num(x_q).ravel()),
        )
    ).reshape(n1, n2, len(outputs), 2)

    table_dx1 = derivs[..., 0] * X1_phys[..., None] if log1 else derivs[..., 0]
    table_dx2 = derivs[..., 1] * X2_phys[..., None] if log2 else derivs[..., 1]
    table_dx1dx2 = np.gradient(table_dx1, arr_x2, axis=1)

    reachable = solved & _all_finite(table_f, table_dx1, table_dx2, table_dx1dx2)

    table_f, table_dx1, table_dx2, table_dx1dx2 = (
        np.nan_to_num(a, posinf=0.0, neginf=0.0)
        for a in (table_f, table_dx1, table_dx2, table_dx1dx2)
    )
    table_f, table_dx1, table_dx2, table_dx1dx2, reachable = fill_ghost(
        arr_x1,
        arr_x2,
        table_f,
        table_dx1,
        table_dx2,
        table_dx1dx2,
        reachable,
    )

    return _save_table(
        tables_base_path / processed_fluid_name / name,
        table_config,
        arr_x1,
        arr_x2,
        table_f,
        table_dx1,
        table_dx2,
        table_dx1dx2,
        ~reachable,
        output_names=outputs,
    )


def fill_nan_nearest(
    flat_data: np.ndarray, x1_flat: np.ndarray, x2_flat: np.ndarray
) -> np.ndarray:
    mask_valid = np.isfinite(flat_data)
    if mask_valid.all():
        return flat_data
    if not mask_valid.any():
        return flat_data
    points_valid = np.column_stack((x1_flat[mask_valid], x2_flat[mask_valid]))
    values_valid = flat_data[mask_valid]
    points_missing = np.column_stack((x1_flat[~mask_valid], x2_flat[~mask_valid]))
    filled_values = griddata(
        points_valid, values_valid, points_missing, method="nearest"
    )
    flat_data[~mask_valid] = filled_values
    return flat_data


def fluid_table_registry(fluid_name: str) -> list:
    """
    Per-fluid copies of TABLE_REGISTRY with axis ranges derived from the
    fluid's own EOS.
    """
    _, eos, _, _, saturation = _get_fluid_modules(fluid_name)
    Tt, Pc = float(eos.T_triple), float(eos.P_crit)
    rho_c = float(eos.rho_crit_mass)

    T_min, T_max = Tt + 1.0, float(eos.T_max)

    Pt = float(eos.P_triple)
    P_min, P_max = max(1.0e4, 1.1 * Pt), 1.16 * Pc
    rhoL_cold = float(saturation.densities(T_min)[0])
    D_min, D_max = 0.016 * rho_c, 0.98 * rhoL_cold

    lo = eos.props_rhoT(jnp.asarray(D_max), jnp.asarray(T_min))
    hi = eos.props_rhoT(jnp.asarray(D_min), jnp.asarray(T_max))

    def _pad(a: float, b: float, f: float = 0.03):
        span = b - a
        return float(a - f * span), float(b + f * span)

    ranges = {
        ThermoVar.D: (D_min, D_max),
        ThermoVar.T: (T_min, T_max),
        ThermoVar.P: (P_min, P_max),
        **{
            v: _pad(float(lo[v.internal_key]), float(hi[v.internal_key]))
            for v in (ThermoVar.U, ThermoVar.H, ThermoVar.S)
        },
    }

    out = []
    for cfg in TABLE_REGISTRY:
        x_lo, x_hi = ranges[cfg.x_axis.variable]
        y_lo, y_hi = ranges[cfg.y_axis.variable]
        out.append(
            cfg.model_copy(
                update=dict(
                    x_axis=cfg.x_axis.model_copy(
                        update=dict(min_val=x_lo, max_val=x_hi)
                    ),
                    y_axis=cfg.y_axis.model_copy(
                        update=dict(min_val=y_lo, max_val=y_hi)
                    ),
                )
            )
        )
    return out


def _probe_error(table_config, tbl, ax1, ax2, node_bad, truth_at, along_x1: bool):
    """
    A probe is dropped only when a bracketing node is a ghost: the interpolant
    rests there on an extrapolated value, so the error would say nothing.
    """
    log1 = table_config.x_axis.spacing == "log"
    log2 = table_config.y_axis.spacing == "log"
    mid = lambda a: 0.5 * (a[:-1] + a[1:])  # noqa: E731
    g1, g2 = (mid(ax1), ax2) if along_x1 else (ax1, mid(ax2))
    bad_stencil = (
        node_bad[:-1, :] | node_bad[1:, :]
        if along_x1
        else node_bad[:, :-1] | node_bad[:, 1:]
    )

    X1, X2 = np.meshgrid(
        np.exp(g1) if log1 else g1, np.exp(g2) if log2 else g2, indexing="ij"
    )
    truth, ok = truth_at(X1, X2)
    pred = np.asarray(
        jax.jit(jax.vmap(lambda a, b: tbl(a, b)))(
            jnp.asarray(X1.ravel()), jnp.asarray(X2.ravel())
        )
    ).reshape(truth.shape)

    flat = truth.reshape(-1, truth.shape[-1])
    den = np.maximum(np.abs(truth), 1e-3 * np.maximum(np.ptp(flat, axis=0), 1e-30))
    err = np.max(np.abs(pred - truth) / den, axis=-1)
    return np.where(ok & ~bad_stencil, err, np.nan)


def build_adaptive_table(
    fluid_name: str,
    table_config: TableSpec,
    tables_base_path: Optional[Path] = None,
    target: float = 1e-5,
    n_start: int = 96,
    quantile: float = 0.002,
    max_nodes: int = 900,
    max_rounds: int = 5,
) -> Path:
    """Both phase branches of a pair, each graded until it meets `target`.

    Each round measures the error against a fresh solve at the edge midpoints (
    obviously because Bicubics are exact on the nodes themselves), then subdivides
    the axis intervals where more than `quantile` of the cells miss. Returns the
    single-phase table's path; the mixture table sits beside it under the same name
    with a `_dome` suffix.
    """
    if tables_base_path is None:
        tables_base_path = get_table_path()
    processed_fluid_name, eos, viscosity, conductivity, saturation = _get_fluid_modules(
        fluid_name
    )
    ctx = (eos, viscosity, conductivity, saturation, processed_fluid_name)
    field_ctx = make_field_ctx(
        table_config, eos, viscosity, conductivity, saturation, processed_fluid_name
    )

    def single_phase_truth(X1, X2):
        f, _, _, ok = solve_field(
            table_config, X1, X2, *ctx, warn_unconverged=False, ctx=field_ctx
        )
        return f, ok

    dome_outputs = list(table_config.outputs) + [ThermoVar.Q]

    def dome_truth(X1, X2):
        _, T_sat, x_q = _solve_dome_state(table_config, X1, X2, saturation)
        ok = np.isfinite(T_sat) & np.isfinite(x_q)
        values = _dome_values(dome_outputs, saturation, T_sat, x_q)
        f = np.stack(
            [np.asarray(values[var]).reshape(X1.shape) for var in dome_outputs], axis=-1
        )
        return np.nan_to_num(f), ok

    x1_cfg, x2_cfg = table_config.x_axis, table_config.y_axis
    log1, log2 = x1_cfg.spacing == "log", x2_cfg.spacing == "log"

    def start_axes(x1_range, x2_range):
        return (
            _grid_axis(*x1_range, log1, n_start),  # type: ignore
            _grid_axis(*x2_range, log2, n_start),  # type: ignore
        )

    def refine(label, build_one, truth_at, ax1, ax2):
        path = None
        for rnd in range(max_rounds):
            path = build_one(ax1, ax2)
            tbl = BicubicInterpolation.create(str(path.parent), dtype=jnp.float64)
            bad_n = np.load(path)["unreachable"]

            e1 = _probe_error(table_config, tbl, ax1, ax2, bad_n, truth_at, True)
            e2 = _probe_error(table_config, tbl, ax1, ax2, bad_n, truth_at, False)

            def frac(e, ax):
                return np.nan_to_num(np.nanmean((e > target).astype(float), axis=ax))

            m1, m2 = frac(e1, 1) > quantile, frac(e2, 0) > quantile
            logger.info(
                f"  {label} round {rnd}: {ax1.size}x{ax2.size}, "
                f"median {np.nanmedian(np.concatenate([e1.ravel(), e2.ravel()])):.1e}, "
                f"{bad_n.mean():.1%} unreachable, "
                f"{int(m1.sum() + m2.sum())} intervals to refine"
            )
            if not (m1.any() or m2.any()):
                logger.info(
                    f"  {label}: target {target:g} met at {ax1.size}x{ax2.size}"
                )
                break

            def split(a, m):
                return np.unique(np.concatenate([a, 0.5 * (a[:-1] + a[1:])[m]]))

            n1, n2 = split(ax1, m1), split(ax2, m2)
            if n1.size * n2.size > max_nodes**2:
                logger.warning(
                    f"  {label}: node budget reached at {ax1.size}x{ax2.size}, "
                    f"target {target:g} not met everywhere"
                )
                break
            ax1, ax2 = n1, n2
        return path

    path = refine(
        table_config.name,
        lambda a1, a2: build_vectorized_table(
            fluid_name, table_config, tables_base_path, axes=(a1, a2), ctx=field_ctx
        ),
        single_phase_truth,
        *start_axes((x1_cfg.min_val, x1_cfg.max_val), (x2_cfg.min_val, x2_cfg.max_val)),
    )
    # (T, P) pins the saturation state itself, so no two-phase state can be
    # named by that pair: no mixture branch to compute
    pair = {x1_cfg.variable, x2_cfg.variable}
    if {ThermoVar.P, ThermoVar.T} <= pair:
        return path  # type: ignore

    refine(
        f"{table_config.name}_dome",
        lambda a1, a2: build_dome_table(
            fluid_name, table_config, tables_base_path, axes=(a1, a2)
        ),
        dome_truth,
        *start_axes(*dome_axis_bounds(table_config, saturation)),
    )
    return path  # type: ignore
