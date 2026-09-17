import itertools
import logging
from pathlib import Path
from typing import Optional

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from ...core.config import TableSpec, ThermoVar, get_table_path  # noqa: E402
from ...core.flash.results import as_mixed, mixture_state  # noqa: E402
from ...core.flash.single_phase import solve_single_phase  # noqa: E402
from ...core.flash.two_phase import solve_two_phase  # noqa: E402
from ...core.interp import BicubicInterpolation  # noqa: E402
from ...core.tolerances import TOL  # noqa: E402
from .helpers import _get_fluid_modules, _mapped, fill_ghost  # noqa: E402

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# how far outside the dome a node is still given a mixture value, so that the
# two tables of a pair overlap
_GHOST_QUALITY = 0.05


def _stored_value(saturation, var: ThermoVar, T, quality):
    """One output of the mixture, in the form the lever rule is linear in."""
    if var == ThermoVar.Q:
        return quality
    return as_mixed(var, mixture_state(saturation, T, quality)[var])


def _axis(axis, lo, hi, n: int) -> np.ndarray:
    if axis.spacing == "log":
        return np.linspace(np.log(lo), np.log(hi), n)
    return np.linspace(lo, hi, n)


def _physical(axis, u):
    """axis converter to physical units."""
    return jnp.exp(u) if axis.spacing == "log" else u


def _single_phase_node(cfg: TableSpec, eos, saturation):
    """(u1, u2) -> (outputs, ok), in the axes' internal coordinates, so that its
    derivatives are the ones the table stores."""
    x, y = cfg.x_axis, cfg.y_axis

    def node(u1, u2):
        ok, rho, T = solve_single_phase(
            eos,
            saturation,
            x.variable,
            _physical(x, u1),
            y.variable,
            _physical(y, u2),
            jnp.array(False),
        )
        solution = {ThermoVar.D: rho, ThermoVar.T: T}
        return jnp.stack([solution[var] for var in cfg.outputs]), ok

    return node


def _mixture_node(cfg: TableSpec, eos, saturation, outputs):
    x, y = cfg.x_axis, cfg.y_axis

    def node(u1, u2):
        ok, T, quality = solve_two_phase(
            eos,
            saturation,
            x.variable,
            _physical(x, u1),
            y.variable,
            _physical(y, u2),
            jnp.array(False),
            slack=_GHOST_QUALITY,
        )
        values = [_stored_value(saturation, var, T, quality) for var in outputs]
        return jnp.stack(values), ok

    return node


def _dome_window(cfg: TableSpec, saturation):
    """
    This method is essential, windows the dome for the user specified bounds provided
    for each axis :

    Read off the saturated liquid and vapour values along the curve, walked in
    s like the two-phase seed, and not only at its ends since there is no proved
    monoticity : h_V, for one, peaks inside.

    Only the points of the curve within the bounds returned by this method of both
    axes count.
    """
    s = jnp.linspace(0.0, saturation.s_of_T(saturation.T_min), TOL.caps.n_seed_scan)
    T = saturation.T_of_s(s)
    axes = (cfg.x_axis, cfg.y_axis)  # the two inputs

    def segment(axis):
        """(lower, upper) of the liquid and vapour values, at each point."""

        def saturated(t, quality):
            return mixture_state(saturation, t, jnp.asarray(quality))[axis.variable]

        ends = [jax.vmap(saturated, (0, None))(T, q) for q in (0.0, 1.0)]
        return np.sort(np.stack(ends), axis=0)

    segments = [segment(axis) for axis in axes]
    within = np.all(
        [(hi >= a.min_val) & (lo <= a.max_val) for (lo, hi), a in zip(segments, axes)],
        axis=0,
    )
    if not within.any():
        raise ValueError(f"{cfg.name}: the dome lies outside the bounds")
    # the curve is walked in steps: a bound crosses it between a kept point and
    # its neighbour, which must count too or the window stops short of the bound
    # same convolves centering the filter on the second array :
    # np.conv([1, 1, 0], ones(3)) > 0 --> ([2, 2, 1]) > 0 --> [1, 1, 1]
    # the neighbour is counted
    within = np.convolve(within, np.ones(3), mode="same") > 0
    return tuple(
        (
            max(float(lo[within].min()), a.min_val),
            min(float(hi[within].max()), a.max_val),
        )
        for (lo, hi), a in zip(segments, axes)
    )


def _branches(cfg: TableSpec, eos, saturation):
    """A branch can be either single phase or in the dome.

    (table name, node, outputs, (x range, y range)) for each table of a pair.

    The mixture table gets its own window, fitted to the dome see the preceding method:
    on the whole bounds most of its nodes would lie outside the dome, unsolved.
    """
    x, y = cfg.x_axis, cfg.y_axis
    outputs = list(cfg.outputs) + [ThermoVar.Q]  # the quality last, it picks the branch
    return [
        (
            cfg.name,
            _single_phase_node(cfg, eos, saturation),  # solved
            cfg.outputs,  # what was solved
            ((x.min_val, x.max_val), (y.min_val, y.max_val)),  # bounds
        ),
        (
            f"{cfg.name}_dome",
            _mixture_node(cfg, eos, saturation, outputs),  # solved
            outputs,  # what was solved
            _dome_window(cfg, saturation),  # bounds
        ),
    ]


def _build_table(path: Path, cfg: TableSpec, node, outputs, ax1, ax2) -> None:
    """
    Solve and differentiate every node, extrapolate the failed ones with
    taylor extrapolation : these points will never be evaluated, they are likely to
    be out of physical bounds but still extrapolated to fail silently when one call
    was made out of the table. Finally save in .npz format.
    """
    logger.info(f"Building {path.parent.name} ({ax1.size}x{ax2.size})...")

    def with_values(u1, u2):
        values, ok = node(u1, u2)
        return values, (values, ok)

    U1, U2 = np.meshgrid(ax1, ax2, indexing="ij")
    derivatives = jax.jacfwd(with_values, argnums=(0, 1), has_aux=True)
    (dx1, dx2), (f, ok) = _mapped(derivatives, U1.ravel(), U2.ravel())

    shape = (ax1.size, ax2.size, len(outputs))
    f, dx1, dx2 = f.reshape(shape), dx1.reshape(shape), dx2.reshape(shape)
    ok = ok.reshape(U1.shape)

    # a failed node's derivatives are finite but meaningless NaN keeps them out
    # of its neighbours' cross derivative, yields greater errors at the edges of
    # the domain
    dx1 = np.where(ok[..., None], dx1, np.nan)
    dx1dx2 = np.gradient(dx1, ax2, axis=1)

    everything = np.concatenate([f, dx1, dx2, dx1dx2], axis=-1)
    solved = ok & np.isfinite(everything).all(axis=-1)
    if not solved.any():
        raise RuntimeError(f"{path.parent.name}: no node could be solved")
    logger.info(f"  solved: {solved.mean():.1%}")

    known = solved
    while not known.all():
        f, dx1, dx2, dx1dx2, known = fill_ghost(ax1, ax2, f, dx1, dx2, dx1dx2, known)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        arr_x1=ax1,
        arr_x2=ax2,
        f=f,
        dx1=dx1,
        dx2=dx2,
        dx1dx2=dx1dx2,
        extrapolated=~solved,  # flags for probes : do not take them into account
        is_log_x1=cfg.x_axis.spacing == "log",
        is_log_x2=cfg.y_axis.spacing == "log",
        output_names=np.array([var.value for var in outputs]),
        x_name=cfg.x_axis.variable.value,
        y_name=cfg.y_axis.variable.value,
    )


def _probe_errors(table, cfg: TableSpec, node, ax1, ax2, along_x1: bool):
    """Relative error of the table at the midpoints of the cell edges since the
    interpolation is exact at the sampled points the errors there are meaningless.

    NaN where nothing is measured: the solve failed there, or the edge rests on
    an extrapolated node, which makes no claim of precision.
    """
    extrapolated = table.extrapolated.array
    if along_x1:  # replaced by its middle points
        ax1 = 0.5 * (ax1[:-1] + ax1[1:])
        # one of the two nodes [a, (a+b)/2, b] a or b is extrapolated
        # brush it aside
        unmeasured = extrapolated[:-1, :] | extrapolated[1:, :]
    else:
        ax2 = 0.5 * (ax2[:-1] + ax2[1:])
        unmeasured = extrapolated[:, :-1] | extrapolated[:, 1:]

    U1, U2 = np.meshgrid(ax1, ax2, indexing="ij")
    # first compute ground truth
    truth, ok = _mapped(node, U1.ravel(), U2.ravel())
    # the table is looked up in physical units
    X1 = np.exp(U1) if table.log_x else U1
    X2 = np.exp(U2) if table.log_y else U2
    lookup = _mapped(lambda a, b: table(a, b), X1.ravel(), X2.ravel())

    # the quality a dome table carries last only picks the branch, and turns
    # singular at the critical point: the target is held on the outputs alone
    n = len(cfg.outputs)
    truth = np.where(ok[:, None], truth[:, :n], np.nan)
    # the outputs are D, T or v, never close to zero: a plain relative error
    error = np.max(np.abs(lookup[:, :n] - truth) / np.abs(truth), axis=1)
    return np.where(unmeasured, np.nan, error.reshape(U1.shape))


def _missed(error, target, axis):
    measured = np.isfinite(error).sum(axis=axis)
    return (error > target).sum(axis=axis) / np.maximum(measured, 1)


def _split(ax, where):
    """`ax` with the midpoint of every flagged interval inserted, an invalid axis
    splits at its middle, this is what we call refining."""
    return np.unique(np.concatenate([ax, 0.5 * (ax[:-1] + ax[1:])[where]]))


def _refine(path, cfg, node, outputs, ax1, ax2, target, quantile, max_nodes):
    """Build, measure at the edge midpoints (a bicubic is exact on the nodes),
    and split the intervals where more than `quantile` of the probes miss,
    until none does or the node budget is spent: every round splits at least
    one interval, so the budget bounds the rounds."""
    name = path.parent.name
    for rnd in itertools.count():
        _build_table(path, cfg, node, outputs, ax1, ax2)
        table = BicubicInterpolation.create(str(path), dtype=jnp.float64)
        error1 = _probe_errors(table, cfg, node, ax1, ax2, along_x1=True)
        error2 = _probe_errors(table, cfg, node, ax1, ax2, along_x1=False)
        if np.isnan(error1).all() and np.isnan(error2).all():
            raise RuntimeError(f"{name}: no probe could be measured")

        split1 = _missed(error1, target, axis=1) > quantile
        split2 = _missed(error2, target, axis=0) > quantile
        median = np.nanmedian(np.concatenate([error1.ravel(), error2.ravel()]))
        logger.info(
            f"  {name} round {rnd}: {ax1.size}x{ax2.size}, median {median:.1e}, "
            f"{table.extrapolated.array.mean():.1%} extrapolated, "
            f"{split1.sum() + split2.sum()} intervals to refine"
        )
        if not (split1.any() or split2.any()):
            logger.info(f"  {name}: target {target:g} met")
            return

        new1, new2 = _split(ax1, split1), _split(ax2, split2)
        if new1.size * new2.size > max_nodes**2:
            logger.warning(f"  {name}: node budget reached, target {target:g} not met")
            return
        ax1, ax2 = new1, new2


def build_adaptive_table(
    fluid_name: str,
    table_config: TableSpec,
    tables_base_path: Optional[Path] = None,
    target: float = 1e-5,
    n_start: int = 96,
    quantile: float = 0.002,
    max_nodes: int = 900,
) -> Path:
    """Both tables of a pair, each refined until it meets `target`. Returns the
    single-phase table's path, the mixture one sits beside it with `_dome`."""
    fluid, eos, saturation = _get_fluid_modules(fluid_name)
    base = (tables_base_path or get_table_path()) / fluid
    x, y = table_config.x_axis, table_config.y_axis
    for name, node, outputs, (x_range, y_range) in _branches(
        table_config, eos, saturation
    ):
        _refine(
            base / name / "table_data.npz",
            table_config,
            node,
            outputs,
            _axis(x, *x_range, n_start),  # type: ignore
            _axis(y, *y_range, n_start),  # type: ignore
            target,
            quantile,
            max_nodes,
        )
    return base / table_config.name / "table_data.npz"
