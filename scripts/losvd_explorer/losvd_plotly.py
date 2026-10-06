"""Plotly figures for the interactive explorer.

Every panel has a ``build_*`` function that creates the traces once and an
``update_*`` function that only swaps the per-bin ``y`` data. The explorer calls
``build_*`` when a file is loaded and ``update_*`` whenever the selected bin changes,
so the (constant) wavelength and velocity grids are sent to the browser only once.
Both ``go.Figure`` and ``go.FigureWidget`` can be passed as ``figure_cls``.
"""
from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from losvd_data import (
    MAP_QUANTITY_BY_KEY,
    Q_16,
    Q_84,
    Q_HI,
    Q_LO,
    Q_MEAN,
    Q_MED,
    Q_STD,
    BinView,
    LosvdDataset,
    color_limits,
    excluded_spans,
)

COLORS = {
    "observed": "#1f2937",
    "model": "#dc2626",
    "model_band_inner": "rgba(220,38,38,0.28)",
    "model_band_outer": "rgba(220,38,38,0.12)",
    "continuum": "#a16207",
    "continuum_band": "rgba(234,179,8,0.30)",
    "residual": "#15803d",
    "noise_band": "rgba(21,128,61,0.15)",
    "losvd": "#1f4e79",
    "losvd_band_inner": "rgba(31,78,121,0.35)",
    "losvd_band_outer": "rgba(31,78,121,0.13)",
    "gh": "#ea580c",
    "mean": "#64748b",
    "excluded": "rgba(100,116,139,0.18)",
    "selected": "#000000",
    "outline": "rgba(15,23,42,0.35)",
}
COLORSCALES = {"diverging": "RdBu_r", "sequential": "Viridis"}
GH_NAME = "Gauss-Hermite (v, σ, h3, h4)"
ERROR_KEY = {"vel": "dvel", "sigma": "dsigma", "h3": "dh3", "h4": "dh4"}

_BASE_LAYOUT = dict(
    autosize=True,
    margin=dict(l=60, r=20, t=50, b=50),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(size=12),
    hoverlabel=dict(font_size=12),
)


def _trace(fig, name: str):
    for trace in fig.data:
        if trace.name == name:
            return trace
    raise KeyError(name)


def _band(lo_name: str, hi_name: str, color: str, shape: str = "linear", **kw) -> list[go.Scatter]:
    """Two traces forming a filled band; only the upper one (``hi_name``) appears in the legend."""
    return [
        go.Scatter(name=lo_name, mode="lines", line=dict(width=0, shape=shape), hoverinfo="skip", showlegend=False, **kw),
        go.Scatter(name=hi_name, mode="lines", line=dict(width=0, shape=shape), fill="tonexty", fillcolor=color, hoverinfo="skip", **kw),
    ]


def _grid_z(image: np.ndarray) -> list[list[float | None]]:
    return [[float(v) if np.isfinite(v) else None for v in row] for row in image]


# =============================================================================
# Map / profile panel
# =============================================================================
def map_title(ds: LosvdDataset, key: str, stat: str, subtract_median: bool) -> str:
    quantity = MAP_QUANTITY_BY_KEY[key]
    parts = [quantity.label]
    if quantity.uses_stat:
        parts.append(f"({'posterior mean' if stat == 'mean' else 'posterior median'})")
    if subtract_median and key == "vel":
        raw = ds.quantity_values(key, stat)
        parts.append(f"− v_sys ({np.nanmedian(raw):.1f} km/s)")
    return " ".join(parts)


def build_map_figure(ds: LosvdDataset, figure_cls=go.Figure, height: int = 470):
    fig = figure_cls()
    fig.update_layout(**_BASE_LAYOUT, height=height, showlegend=False, dragmode="pan")

    if ds.layout_kind == "map":
        grid = ds.grid
        fig.add_trace(
            go.Heatmap(
                name="map",
                x=grid.x_centers,
                y=grid.y_centers,
                z=_grid_z(np.full(grid.bin_image.shape, np.nan)),
                customdata=grid.bin_image,
                hoverongaps=False,
                colorbar=dict(thickness=14, len=0.9),
            )
        )
        xs, ys = ds.bin_outline
        fig.add_trace(go.Scatter(name="outline", x=xs, y=ys, mode="lines", line=dict(color=COLORS["outline"], width=0.7), hoverinfo="skip"))
        empty = np.flatnonzero(~ds.has_result)
        fig.add_trace(
            go.Scatter(
                name="empty", x=ds.xbin[empty], y=ds.ybin[empty], mode="markers", customdata=empty,
                marker=dict(symbol="x-thin", size=8, line=dict(width=1.5, color="#64748b")),
                hovertemplate="Bin %{customdata}: no result<extra></extra>",
            )
        )
        fig.add_trace(go.Scatter(name="selected", mode="lines", line=dict(color=COLORS["selected"], width=2.5), hoverinfo="skip"))
        fig.update_xaxes(title="x [arcsec]", zeroline=False, showgrid=False, constrain="domain")
        fig.update_yaxes(title="y [arcsec]", zeroline=False, showgrid=False, scaleanchor="x", scaleratio=1, constrain="domain")
    elif ds.layout_kind == "profile":
        coord = ds.profile_coordinate
        fig.add_trace(
            go.Scatter(
                name="profile", x=coord, mode="markers+lines", customdata=np.arange(ds.nbins),
                line=dict(color="#94a3b8", width=1), marker=dict(size=9, color=COLORS["losvd"]),
            )
        )
        fig.add_trace(go.Scatter(name="selected", mode="markers", marker=dict(size=16, symbol="circle-open", color=COLORS["selected"], line=dict(width=2.5)), hoverinfo="skip"))
        fig.update_xaxes(title="position along slit [arcsec]", zeroline=False)
    else:
        fig.update_layout(
            annotations=[dict(text="Single spectrum – no spatial map", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))]
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    return fig


def update_map_quantity(fig, ds: LosvdDataset, key: str, stat: str = "median", subtract_median: bool = False) -> None:
    quantity = MAP_QUANTITY_BY_KEY[key]
    values = ds.quantity_values(key, stat, subtract_median and key == "vel")
    title = map_title(ds, key, stat, subtract_median)
    unit = f" {quantity.unit}" if quantity.unit else ""
    fmt = ".1f" if quantity.unit == "km/s" else ".3g"

    with fig.batch_update():
        fig.layout.title = dict(text=f"{html.escape(ds.run_name)} · {title}", font=dict(size=14))
        if ds.layout_kind == "map":
            zmin, zmax = color_limits(values, quantity.kind)
            _trace(fig, "map").update(
                z=_grid_z(ds.grid.image_of(values)),
                zmin=zmin,
                zmax=zmax,
                colorscale=COLORSCALES[quantity.kind],
                colorbar_title_text=quantity.axis_title,
                hovertemplate=(
                    "Bin %{customdata}<br>x = %{x:.2f}″, y = %{y:.2f}″<br>"
                    f"{quantity.label} = %{{z:{fmt}}}{unit}<extra></extra>"
                ),
            )
        elif ds.layout_kind == "profile":
            err_key = ERROR_KEY.get(key)
            errors = ds.quantity_values(err_key, stat) if err_key else None
            _trace(fig, "profile").update(
                y=values,
                error_y=dict(type="data", array=errors, visible=errors is not None, color="#94a3b8", thickness=1),
                hovertemplate=f"Bin %{{customdata}}<br>position = %{{x:.2f}}″<br>{quantity.label} = %{{y:{fmt}}}{unit}<extra></extra>",
            )
            fig.update_yaxes(title=quantity.axis_title)


def update_map_selection(fig, ds: LosvdDataset, idx: int) -> None:
    if ds.layout_kind == "map":
        xs, ys = ds.grid.outline(only_bin=idx)
        if not xs:  # bin has no pixels on the grid – mark its centre instead
            xs, ys = [float(ds.xbin[idx])], [float(ds.ybin[idx])]
        _trace(fig, "selected").update(x=xs, y=ys)
    elif ds.layout_kind == "profile":
        profile = _trace(fig, "profile")
        _trace(fig, "selected").update(x=[ds.profile_coordinate[idx]], y=[profile.y[idx] if profile.y is not None else None])


# =============================================================================
# LOSVD panel
# =============================================================================
def build_losvd_figure(figure_cls=go.Figure, height: int = 470):
    fig = figure_cls(
        data=[
            *_band("losvd_outer_lo", "99.8 % interval", COLORS["losvd_band_outer"], shape="hvh"),
            *_band("losvd_inner_lo", "68 % interval", COLORS["losvd_band_inner"], shape="hvh"),
            go.Scatter(name="LOSVD median", mode="lines", line=dict(color=COLORS["losvd"], width=2.2, shape="hvh"),
                       hovertemplate="v = %{x:.0f} km/s<br>LOSVD = %{y:.4f}<extra>median</extra>"),
            go.Scatter(name="LOSVD mean", mode="lines", line=dict(color=COLORS["mean"], width=1.2, dash="dot", shape="hvh"),
                       visible="legendonly", hoverinfo="skip"),
            go.Scatter(name=GH_NAME, mode="lines", line=dict(color=COLORS["gh"], width=2, dash="dash"), hoverinfo="skip"),
        ]
    )
    fig.update_layout(
        **{**_BASE_LAYOUT, "margin": dict(l=60, r=20, t=50, b=100)},
        height=height,
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, font=dict(size=11)),
        shapes=[
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=0, line=dict(color="#94a3b8", width=1, dash="dash")),
            dict(type="line", xref="x", x0=0, x1=0, yref="paper", y0=0, y1=1, line=dict(color="#94a3b8", width=1, dash="dot")),
        ],
    )
    fig.update_xaxes(title="velocity [km/s]", zeroline=False)
    fig.update_yaxes(title="LOSVD", zeroline=False)
    return fig


def update_losvd(fig, view: BinView) -> None:
    names = ["losvd_outer_lo", "99.8 % interval", "losvd_inner_lo", "68 % interval", "LOSVD median", "LOSVD mean", GH_NAME]
    traces = {name: _trace(fig, name) for name in names}
    with fig.batch_update():
        if not view.has_result:
            for trace in traces.values():
                trace.update(x=[], y=[])
            fig.layout.title = dict(text=f"Bin {view.idx} · no result", font=dict(size=14))
            fig.layout.annotations = [dict(text="This bin has no fit result (NaN).", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))]
            return
        losvd = view.losvd
        rows = {
            "losvd_outer_lo": losvd[Q_LO], "99.8 % interval": losvd[Q_HI],
            "losvd_inner_lo": losvd[Q_16], "68 % interval": losvd[Q_84],
            "LOSVD median": losvd[Q_MED], "LOSVD mean": losvd[Q_MEAN],
            GH_NAME: view.gh_model if view.gh_model is not None else [],
        }
        for name, y in rows.items():
            traces[name].update(x=view.xvel if len(y) else [], y=y)
        fig.layout.title = dict(text=f"Bin {view.idx} · LOSVD", font=dict(size=14))
        fig.layout.annotations = []


# =============================================================================
# Spectrum + residual panel
# =============================================================================
def build_spectrum_figure(ds: LosvdDataset, figure_cls=go.Figure, height: int = 560):
    base = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig = figure_cls(base)
    fig.update_layout(
        **{**_BASE_LAYOUT, "margin": dict(l=60, r=20, t=50, b=95)},
        height=height,
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(title="wavelength [Å]", row=2, col=1)
    fig.update_yaxes(title="normalised flux", row=1, col=1)
    fig.update_yaxes(title="residual", row=2, col=1, zeroline=True, zerolinecolor="#94a3b8")

    if not ds.has_spectra:
        fig.update_layout(annotations=[dict(text="This HDF5 file does not contain spectral-fit diagnostics.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#64748b"))])
        return fig

    wave = ds.wave_obs
    row1 = [
        *_band("continuum_lo", "Continuum 68 %", COLORS["continuum_band"], x=wave),
        go.Scatter(name="Continuum (median)", x=wave, mode="lines", line=dict(color=COLORS["continuum"], width=1, dash="dash"), hoverinfo="skip"),
        *_band("model_outer_lo", "Model 99.8 %", COLORS["model_band_outer"], x=wave),
        *_band("model_inner_lo", "Model 68 %", COLORS["model_band_inner"], x=wave),
        go.Scatter(name="Observed", x=wave, mode="lines", line=dict(color=COLORS["observed"], width=1.2),
                   hovertemplate="λ = %{x:.2f} Å<br>flux = %{y:.4f}<extra>observed</extra>"),
        go.Scatter(name="Best fit (median)", x=wave, mode="lines", line=dict(color=COLORS["model"], width=1.6),
                   hovertemplate="λ = %{x:.2f} Å<br>model = %{y:.4f}<extra>best fit</extra>"),
    ]
    row2 = [
        *_band("noise_lo", "±1σ model noise", COLORS["noise_band"], x=wave),
        go.Scatter(name="Residual", x=wave, mode="lines", line=dict(color=COLORS["residual"], width=1.1),
                   hovertemplate="λ = %{x:.2f} Å<br>residual = %{y:.4f}<extra></extra>", showlegend=False),
    ]
    for trace in row1:
        fig.add_trace(trace, row=1, col=1)
    for trace in row2:
        fig.add_trace(trace, row=2, col=1)
    fig.update_layout(
        shapes=[
            dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, fillcolor=COLORS["excluded"], line=dict(width=0), layer="below")
            for x0, x1 in excluded_spans(ds.wave_obs, ds.good_pixels)
        ]
    )
    return fig


def update_spectrum(fig, view: BinView) -> None:
    if view.wave is None:
        return
    names = ["continuum_lo", "Continuum 68 %", "Continuum (median)", "model_outer_lo", "Model 99.8 %",
             "model_inner_lo", "Model 68 %", "Observed", "Best fit (median)", "noise_lo", "±1σ model noise", "Residual"]
    traces = {name: _trace(fig, name) for name in names}

    with fig.batch_update():
        if not view.has_result:
            for name, trace in traces.items():
                trace.y = view.spec if name == "Observed" else np.full(view.wave.size, np.nan)
            fig.layout.title = dict(text=f"Bin {view.idx} · no fit result – observed spectrum only", font=dict(size=14))
            return

        model, cont, noise = view.model, view.continuum, view.noise
        good = np.isfinite(view.residual)
        rows = {
            "continuum_lo": cont[Q_16], "Continuum 68 %": cont[Q_84], "Continuum (median)": cont[Q_MED],
            "model_outer_lo": model[Q_LO], "Model 99.8 %": model[Q_HI],
            "model_inner_lo": model[Q_16], "Model 68 %": model[Q_84],
            "Observed": view.spec, "Best fit (median)": model[Q_MED],
            "noise_lo": np.where(good, -noise, np.nan) if noise is not None else np.full(view.wave.size, np.nan),
            "±1σ model noise": np.where(good, noise, np.nan) if noise is not None else np.full(view.wave.size, np.nan),
            "Residual": view.residual,
        }
        for name, y in rows.items():
            traces[name].y = y

        fitted = np.concatenate([view.spec[good], model[Q_MED][good]])
        lo, hi = float(np.nanmin(fitted)), float(np.nanmax(fitted))
        pad = 0.06 * (hi - lo if hi > lo else 1.0)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
        spread = float(np.nanpercentile(np.abs(view.residual[good]), 99.5)) if good.any() else 1.0
        if noise is not None and good.any():
            spread = max(spread, 3.0 * float(np.nanmedian(noise[good])))
        fig.update_yaxes(range=[-1.15 * spread, 1.15 * spread], row=2, col=1)
        fig.layout.title = dict(
            text=f"Bin {view.idx} · spectral fit · χ²_red = {view.chi2_red:.2f} · rms = {view.rms:.2e}",
            font=dict(size=14),
        )


# =============================================================================
# Template weights
# =============================================================================
def build_weights_figure(figure_cls=go.Figure, height: int = 300):
    fig = figure_cls(data=[go.Bar(name="weights", marker_color="#475569", hovertemplate="%{x}: %{y:.3f}<extra></extra>")])
    fig.update_layout(**{**_BASE_LAYOUT, "margin": dict(l=60, r=20, t=45, b=40)}, height=height, showlegend=False, bargap=0.25)
    fig.update_yaxes(title="weight", rangemode="tozero")
    return fig


def update_weights(fig, view: BinView) -> None:
    bar = _trace(fig, "weights")
    with fig.batch_update():
        if view.weights is None or not view.has_result:
            bar.update(x=[], y=[])
            fig.layout.title = dict(text="Template weights · n/a", font=dict(size=14))
            return
        w = view.weights
        bar.update(
            x=[f"T{i}" for i in range(w.shape[1])],
            y=w[Q_MED],
            error_y=dict(type="data", symmetric=False, array=w[Q_84] - w[Q_MED], arrayminus=w[Q_MED] - w[Q_16], color="#94a3b8", thickness=1.2),
        )
        fig.layout.title = dict(text="Template weights (median, 68 %)", font=dict(size=14))


# =============================================================================
# HTML panels
# =============================================================================
_PARAM_ROWS = (
    ("vel_star", "v", "km/s", "{:.1f}"),
    ("sigma_star", "σ", "km/s", "{:.1f}"),
    ("h3_star", "h3", "", "{:.3f}"),
    ("h4_star", "h4", "", "{:.3f}"),
    ("snr_real", "S/N (fit)", "", "{:.1f}"),
    ("ell_gp", "GP ℓ", "", "{:.3g}"),
    ("sigma_gp", "GP σ", "", "{:.3g}"),
)

TABLE_CSS = """
<style>
.lx-table{border-collapse:collapse;font-size:13px;width:100%;font-variant-numeric:tabular-nums}
.lx-table th{text-align:left;color:#64748b;font-weight:600;padding:3px 8px;border-bottom:1px solid #e2e8f0}
.lx-table td{padding:3px 8px;border-bottom:1px solid #f1f5f9;color:#0f172a}
.lx-table td.num{text-align:right;font-family:Menlo,Consolas,monospace;font-size:12px}
.lx-card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px 12px}
.lx-card h4{margin:0 0 6px 0;font-size:14px;color:#0f172a}
.lx-muted{color:#64748b}
</style>
"""


def _pm(fmt: str, value: float, error: float) -> str:
    if not np.isfinite(value):
        return "–"
    return f"{fmt.format(value)} ± {fmt.format(error)}" if np.isfinite(error) else fmt.format(value)


def bin_info_html(view: BinView) -> str:
    rows = []
    for key, label, unit, fmt in _PARAM_ROWS:
        summary = view.params.get(key)
        if summary is None:
            continue
        median = _pm(fmt, summary[Q_MED], 0.5 * (summary[Q_84] - summary[Q_16]))
        mean = _pm(fmt, summary[Q_MEAN], summary[Q_STD])
        unit_html = f" <span class='lx-muted'>[{unit}]</span>" if unit else ""
        rows.append(f"<tr><td>{label}{unit_html}</td><td class='num'>{median}</td><td class='num'>{mean}</td></tr>")

    extra = [
        ("Position (x, y, r)", f"{view.x:.2f}″, {view.y:.2f}″, {view.radius:.2f}″"),
        ("χ²_red / rms", f"{view.chi2_red:.2f} / {view.rms:.2e}" if np.isfinite(view.chi2_red) else "–"),
        ("S/N (input bin)", f"{view.bin_snr:.1f}" if np.isfinite(view.bin_snr) else "–"),
        ("Bin flux", f"{view.bin_flux:.3g}" if np.isfinite(view.bin_flux) else "–"),
    ]
    extra_rows = "".join(f"<tr><td>{k}</td><td class='num' colspan='2'>{v}</td></tr>" for k, v in extra)
    status = "" if view.has_result else "<div style='color:#b45309;margin-bottom:6px'>No fit result for this bin.</div>"
    return (
        TABLE_CSS
        + f"<div class='lx-card'><h4>Bin {view.idx}</h4>{status}"
        + "<table class='lx-table'><tr><th>Parameter</th><th>median ± ½(p84−p16)</th><th>mean ± std</th></tr>"
        + "".join(rows)
        + extra_rows
        + "</table></div>"
    )


def run_info_html(ds: LosvdDataset) -> str:
    cells = "".join(
        f"<div><span class='lx-muted'>{html.escape(k)}</span><br><span>{html.escape(v)}</span></div>"
        for k, v in ds.summary_lines()
    )
    return (
        TABLE_CSS
        + "<div class='lx-card'>"
        + f"<h4>{html.escape(ds.run_name)}</h4>"
        + f"<div style='font-family:Menlo,Consolas,monospace;font-size:11px;color:#64748b;margin-bottom:8px;word-break:break-all'>{html.escape(str(ds.path))}</div>"
        + f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;font-size:13px'>{cells}</div>"
        + "</div>"
    )
