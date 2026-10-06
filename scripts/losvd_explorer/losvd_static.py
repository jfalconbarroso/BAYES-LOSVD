"""Static (matplotlib) version of the explorer panels for PNG/PDF export.

Used by the explorer's *Save* buttons and by ``inspect_fits.py``. matplotlib is
used instead of plotly/kaleido so that export works headless without a browser
engine and produces publication-style vector PDFs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from losvd_data import (
    MAP_QUANTITY_BY_KEY,
    Q_16,
    Q_84,
    Q_HI,
    Q_LO,
    Q_MED,
    LosvdDataset,
    color_limits,
)

CMAPS = {"diverging": "RdBu_r", "sequential": "viridis"}
FIGSIZE = (11, 9)


def _segments(xs: list[float | None], ys: list[float | None]) -> list[np.ndarray]:
    segments, current = [], []
    for x, y in zip(xs, ys):
        if x is None:
            if len(current) > 1:
                segments.append(np.array(current))
            current = []
        else:
            current.append((x, y))
    if len(current) > 1:
        segments.append(np.array(current))
    return segments


def _draw_map(ax, ds: LosvdDataset, idx: int, key: str, stat: str, subtract_median: bool) -> None:
    quantity = MAP_QUANTITY_BY_KEY[key]
    values = ds.quantity_values(key, stat, subtract_median and key == "vel")
    vmin, vmax = color_limits(values, quantity.kind)

    if ds.layout_kind == "map":
        grid = ds.grid
        x_edges = np.append(grid.x_centers - 0.5 * grid.dx, grid.x_centers[-1] + 0.5 * grid.dx)
        y_edges = np.append(grid.y_centers - 0.5 * grid.dy, grid.y_centers[-1] + 0.5 * grid.dy)
        image = np.ma.masked_invalid(grid.image_of(values))
        mesh = ax.pcolormesh(x_edges, y_edges, image, cmap=CMAPS[quantity.kind], vmin=vmin, vmax=vmax, shading="flat")
        ax.add_collection(LineCollection(_segments(*ds.bin_outline), colors="0.25", linewidths=0.35, alpha=0.6))
        ax.add_collection(LineCollection(_segments(*grid.outline(only_bin=idx)), colors="k", linewidths=1.8))
        empty = ~ds.has_result
        if empty.any():
            ax.plot(ds.xbin[empty], ds.ybin[empty], "x", color="0.4", ms=4, mew=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("x [arcsec]")
        ax.set_ylabel("y [arcsec]")
        ax.figure.colorbar(mesh, cax=ax.inset_axes([1.03, 0.0, 0.045, 1.0]), label=quantity.axis_title)
    else:  # profile
        coord = ds.profile_coordinate
        ax.plot(coord, values, "o-", color="#1f4e79", ms=5, lw=0.8)
        ax.plot(coord[idx], values[idx], "o", mfc="none", mec="k", ms=12, mew=2)
        ax.set_xlabel("position along slit [arcsec]")
        ax.set_ylabel(quantity.axis_title)
    title = quantity.label + (f" ({stat})" if quantity.uses_stat else "")
    ax.set_title(title, fontsize=11)


def _draw_losvd(ax, view) -> None:
    if not view.has_result:
        ax.text(0.5, 0.5, "no fit result", transform=ax.transAxes, ha="center", va="center", color="0.4")
        return
    v, losvd = view.xvel, view.losvd
    ax.fill_between(v, losvd[Q_LO], losvd[Q_HI], step="mid", color="#1f4e79", alpha=0.15, lw=0, label="99.8 %")
    ax.fill_between(v, losvd[Q_16], losvd[Q_84], step="mid", color="#1f4e79", alpha=0.40, lw=0, label="68 %")
    ax.step(v, losvd[Q_MED], where="mid", color="#1f4e79", lw=1.8, label="median")
    if view.gh_model is not None:
        ax.plot(v, view.gh_model, "--", color="#ea580c", lw=1.6, label="Gauss-Hermite")
    ax.axhline(0.0, color="0.6", ls="--", lw=0.8)
    ax.axvline(0.0, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("velocity [km/s]")
    ax.set_ylabel("LOSVD")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8, frameon=False)

    lines = [f"r = {view.radius:.2f}″"]
    for key, label, fmt in (("vel_star", "v", "{:.1f}"), ("sigma_star", "σ", "{:.1f}"), ("h3_star", "h3", "{:.3f}"), ("h4_star", "h4", "{:.3f}")):
        summary = view.params.get(key)
        if summary is not None:
            err = 0.5 * (summary[Q_84] - summary[Q_16])
            lines.append(f"{label} = {fmt.format(summary[Q_MED])} ± {fmt.format(err)}")
    snr = view.params.get("snr_real")
    if snr is not None:
        lines.append(f"S/N = {snr[Q_MED]:.1f}")
    if np.isfinite(view.chi2_red):
        lines.append(f"χ²_red = {view.chi2_red:.2f}")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=8,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    ax.set_title("LOSVD", fontsize=11)


def _draw_spectrum(ax, view) -> None:
    """Spectral fit in the style of the original ``bayes_losvd_inspect_fits.py``."""
    if not view.has_spectrum:
        ax.text(0.5, 0.5, "no spectral-fit diagnostics", transform=ax.transAxes, ha="center", va="center", color="0.4")
        if view.wave is not None:
            ax.plot(view.wave, view.spec, "k", lw=0.8)
        return
    wave, model, cont, spec = view.wave, view.model, view.continuum, view.spec
    top = 1.1 * np.nanmax(spec)
    floor = 0.7 * np.nanmin(spec)

    ax.fill_between(wave, cont[Q_16], cont[Q_84], facecolor="yellow", alpha=0.5, zorder=0, label="Leg. polynomial")
    ax.plot(wave, cont[Q_16], color="gray", ls="--", lw=1, zorder=0)
    ax.plot(wave, cont[Q_84], color="gray", ls="--", lw=1, zorder=0)
    ax.plot(wave, spec, "k", zorder=1, label="Obs. data")
    ax.fill_between(wave, model[Q_16], model[Q_84], facecolor="orange", alpha=0.75, zorder=2)
    ax.plot(wave, model[Q_MED], color="red", zorder=3, label="Bestfit")
    ax.plot(wave, spec - model[Q_MED] + floor + 0.1, color="green", label="Residuals")
    ax.axhline(floor + 0.1, color="k", ls="--")

    fitted = np.flatnonzero(np.isfinite(view.residual))
    if fitted.size:
        ax.axvline(wave[fitted[0]], color="k", ls=":")
        ax.axvline(wave[fitted[-1]], color="k", ls=":")
        for gap in np.flatnonzero(np.diff(fitted) > 1):
            ax.axvspan(wave[fitted[gap]], wave[fitted[gap + 1]], alpha=0.25, color="gray")

    ax.set_ylim(floor, top)
    ax.set_xlim(wave.min(), wave.max())
    ax.set_ylabel("Norm. flux")
    ax.set_xlabel("Wavelength ($\\mathrm{\\AA}$)")


def render_bin_figure(
    ds: LosvdDataset,
    idx: int,
    quantity: str = "vel",
    stat: str = "median",
    subtract_median: bool = False,
    fig: Figure | None = None,
) -> Figure:
    """Draw one bin: map and LOSVD on top, spectral fit below. Pass a pyplot ``fig`` to show it."""
    view = ds.bin_view(idx)
    if fig is None:
        fig = Figure(figsize=FIGSIZE, layout="constrained")
    grid = GridSpec(2, 2, figure=fig)

    if ds.layout_kind in {"map", "profile"}:
        _draw_map(fig.add_subplot(grid[0, 0]), ds, idx, quantity, stat, subtract_median)
        _draw_losvd(fig.add_subplot(grid[0, 1]), view)
    else:
        _draw_losvd(fig.add_subplot(grid[0, :]), view)
    _draw_spectrum(fig.add_subplot(grid[1, :]), view)

    fig.suptitle(f"BinID: {idx}", fontsize=14, fontweight="bold")
    return fig


def output_path(ds: LosvdDataset, idx: int, outdir: str | Path | None = None, fmt: str = "png") -> Path:
    """Same naming as the original ``bayes_losvd_inspect_fits.py``: ``<file stem>_bin<idx>.<fmt>``."""
    directory = Path(outdir) if outdir is not None else ds.path.parent
    return directory / f"{ds.path.stem}_bin{idx}.{fmt}"


def save_bin_figure(ds: LosvdDataset, idx: int, outdir: str | Path | None = None, fmt: str = "png", dpi: int = 150, **kwargs) -> Path:
    path = output_path(ds, idx, outdir, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_bin_figure(ds, idx, **kwargs)
    fig.savefig(path, dpi=dpi)
    return path
