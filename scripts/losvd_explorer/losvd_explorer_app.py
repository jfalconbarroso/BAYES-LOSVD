"""Voila/Jupyter front end of the LOSVD explorer.

Usage (in a notebook or via ``voila interactive_losvd_explorer.ipynb``)::

    from losvd_explorer_app import launch_app
    launch_app()                       # scans ./results or $LOSVD_EXPLORER_PATH
    launch_app("path/to/run_results.hdf5")  # opens a file directly
"""
from __future__ import annotations

import functools
import html
import os
import traceback
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

import losvd_plotly as lp
import losvd_static
from losvd_data import (
    MAP_QUANTITY_BY_KEY,
    LosvdDataset,
    detect_default_search_root,
    infer_run_name,
    load_dataset,
    resolve_hdf5_candidates,
)

try:
    import plotly.graph_objects as go

    go.FigureWidget()
    FIGURE_WIDGET_ERROR = None
except Exception as exc:  # plotly >= 6 needs anywidget for FigureWidget
    go = None
    FIGURE_WIDGET_ERROR = exc

ENV_PATH = "LOSVD_EXPLORER_PATH"

STATUS_STYLES = {
    "info": ("#eff6ff", "#bfdbfe", "#1d4ed8"),
    "busy": ("#f8fafc", "#cbd5e1", "#334155"),
    "success": ("#ecfdf5", "#bbf7d0", "#047857"),
    "warning": ("#fffbeb", "#fde68a", "#b45309"),
    "error": ("#fef2f2", "#fecaca", "#b91c1c"),
}

HEADER_HTML = """
<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;padding:4px 2px 10px 2px;border-bottom:1px solid #e2e8f0;margin-bottom:8px">
  <span style="font-size:22px;font-weight:700;color:#0f172a">LOSVD Explorer</span>
  <span style="font-size:13px;color:#64748b">Bayes-LOSVD results: kinematic maps, LOSVDs and spectral fits per bin</span>
</div>
"""

HELP_HTML = """
<ol style="margin:0;padding-left:1.2rem;line-height:1.6;font-size:13px">
  <li><b>Path</b>: results directory (searched recursively) or a single <code>*_results.hdf5</code>; press Enter or <b>Scan</b>.</li>
  <li><b>File</b>: pick the run. Unpacked runs (only <code>*_results_bin&lt;N&gt;.hdf5</code> files) are assembled automatically.</li>
  <li><b>Map</b>: choose the quantity (v, σ, h3, h4, their errors, S/N, χ², GP hyper-parameters), posterior median or mean,
      and optionally subtract the median velocity (v<sub>sys</sub>).</li>
  <li><b>Select a bin</b> by clicking it on the map, with ◀ / ▶ (steps through bins that have results), or via the dropdown.
      Hovering only shows a tooltip, unless <b>Select on hover</b> is ticked.</li>
  <li><b>Save bin</b> writes a PNG/PDF of the current bin next to the HDF5 file (same name as the old
      <code>bayes_losvd_inspect_fits.py -s</code>); <b>Save all</b> does this for every bin with results.
      For batch jobs on the command line use <code>python inspect_fits.py -f FILE -l all -s</code>.</li>
</ol>
<p style="font-size:12px;color:#64748b;margin:6px 0 0 0">
  LOSVD panel: shaded bands = 68 % / 99.8 % posterior intervals, orange dashed = Gauss-Hermite profile from the median
  (v, σ, h3, h4). Spectrum: grey regions are excluded from the fit; lower panel shows residuals with the ±1σ noise the likelihood used (1/min(S/N, 100)), which is also the basis of χ²_red.
</p>
"""


def _reports_errors(method):
    """Widget callbacks swallow exceptions silently; show them in the status bar instead."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self.show_error(f"{type(exc).__name__}: {exc}")
            return None

    return wrapper


def _flex(min_width: str) -> widgets.Layout:
    return widgets.Layout(flex=f"1 1 {min_width}", min_width="0", width="auto")


def _panel_grid(children: list[widgets.Widget], min_width: str) -> widgets.GridBox:
    """Responsive row: panels side by side, wrapping below each other on narrow screens."""
    return widgets.GridBox(
        children,
        layout=widgets.Layout(
            grid_template_columns=f"repeat(auto-fit, minmax({min_width}, 1fr))",
            grid_gap="10px",
            width="100%",
        ),
    )


class ExplorerApp:
    def __init__(self, initial_path: str | Path | None = None):
        self.dataset: LosvdDataset | None = None
        self.current_bin: int | None = None
        self._syncing = False
        self.figures: dict[str, object] = {}

        initial_path = initial_path or os.environ.get(ENV_PATH) or detect_default_search_root()

        # --- file controls -------------------------------------------------
        self.path_text = widgets.Text(
            value=str(initial_path),
            placeholder="Results directory or *_results.hdf5 file",
            description="Path",
            layout=_flex("400px"),
        )
        self.scan_button = widgets.Button(description="Scan", icon="search", layout=widgets.Layout(width="90px"))
        self.file_dropdown = widgets.Dropdown(options=[("—", None)], description="File", layout=_flex("400px"))

        # --- view controls ---------------------------------------------------
        self.quantity_dropdown = widgets.Dropdown(description="Map", layout=widgets.Layout(width="260px"))
        self.stat_toggle = widgets.ToggleButtons(
            options=[("Median", "median"), ("Mean", "mean")], value="median",
            style={"button_width": "70px"}, layout=widgets.Layout(width="auto"),
        )
        self.vsys_checkbox = widgets.Checkbox(value=False, description="Subtract v_sys", indent=False, layout=widgets.Layout(width="130px"))
        self.prev_button = widgets.Button(icon="chevron-left", tooltip="Previous bin with result", layout=widgets.Layout(width="40px"))
        self.next_button = widgets.Button(icon="chevron-right", tooltip="Next bin with result", layout=widgets.Layout(width="40px"))
        self.bin_dropdown = widgets.Dropdown(description="Bin", layout=widgets.Layout(width="230px"))
        self.hover_checkbox = widgets.Checkbox(value=False, description="Select on hover", indent=False, layout=widgets.Layout(width="130px"))
        self.format_dropdown = widgets.Dropdown(options=["png", "pdf"], value="png", layout=widgets.Layout(width="70px"))
        self.save_button = widgets.Button(description="Save bin", icon="download", layout=widgets.Layout(width="100px"))
        self.save_all_button = widgets.Button(description="Save all", icon="copy", layout=widgets.Layout(width="100px"))
        self.progress = widgets.IntProgress(layout=widgets.Layout(width="160px", visibility="hidden"))

        self.status_html = widgets.HTML()
        self.run_html = widgets.HTML()
        self.bin_html = widgets.HTML(layout=widgets.Layout(min_width="0"))
        self.body = widgets.VBox()

        view_row = widgets.HBox(
            [
                self.quantity_dropdown, self.stat_toggle, self.vsys_checkbox,
                widgets.HTML("<span style='border-left:1px solid #cbd5e1;height:24px;margin:0 6px'></span>"),
                self.prev_button, self.bin_dropdown, self.next_button, self.hover_checkbox,
                widgets.HTML("<span style='border-left:1px solid #cbd5e1;height:24px;margin:0 6px'></span>"),
                self.format_dropdown, self.save_button, self.save_all_button, self.progress,
            ],
            layout=widgets.Layout(flex_flow="row wrap", align_items="center"),
        )
        self.view_controls = widgets.VBox([view_row], layout=widgets.Layout(display="none"))

        help_box = widgets.Accordion(children=[widgets.HTML(HELP_HTML)], titles=("How to use",), selected_index=None)

        self.root = widgets.VBox(
            [
                widgets.HTML(HEADER_HTML),
                help_box,
                widgets.HBox([self.path_text, self.scan_button], layout=widgets.Layout(width="100%")),
                widgets.HBox([self.file_dropdown], layout=widgets.Layout(width="100%")),
                self.status_html,
                self.run_html,
                self.view_controls,
                self.body,
            ],
            layout=widgets.Layout(width="100%", max_width="1600px"),
        )

        # --- wiring --------------------------------------------------------
        self.path_text.on_submit(lambda _w: self.scan())
        self.scan_button.on_click(lambda _b: self.scan())
        self.file_dropdown.observe(self._on_file_change, names="value")
        self.quantity_dropdown.observe(lambda _c: self.refresh_map(), names="value")
        self.stat_toggle.observe(lambda _c: self.refresh_map(), names="value")
        self.vsys_checkbox.observe(lambda _c: self.refresh_map(), names="value")
        self.bin_dropdown.observe(self._on_bin_dropdown, names="value")
        self.prev_button.on_click(lambda _b: self.step(-1))
        self.next_button.on_click(lambda _b: self.step(+1))
        self.save_button.on_click(lambda _b: self.save_current())
        self.save_all_button.on_click(lambda _b: self.save_all())

    # ------------------------------------------------------------------ status
    def set_status(self, message: str, kind: str = "info") -> None:
        bg, border, color = STATUS_STYLES.get(kind, STATUS_STYLES["info"])
        spinner = "<i class='fa fa-spinner fa-spin' style='margin-right:6px'></i>" if kind == "busy" else ""
        self.status_html.value = (
            f"<div style='background:{bg};border:1px solid {border};color:{color};border-radius:8px;"
            f"padding:7px 11px;font-size:13px'>{spinner}{message}</div>"
        )

    def show_error(self, message: str) -> None:
        self.set_status(
            f"{html.escape(message)}<details style='margin-top:4px'><summary>Traceback</summary>"
            f"<pre style='font-size:11px;white-space:pre-wrap'>{html.escape(traceback.format_exc())}</pre></details>",
            "error",
        )

    # ------------------------------------------------------------ file choice
    @_reports_errors
    def scan(self) -> None:
        self.set_status("Scanning for result files…", "busy")
        try:
            matches = resolve_hdf5_candidates(self.path_text.value)
        except Exception as exc:
            self._set_file_options([])
            self.set_status(html.escape(f"{type(exc).__name__}: {exc}"), "error")
            return

        base = Path(self.path_text.value).expanduser()
        base = base if base.is_dir() else base.parent
        options = []
        for path in matches:
            try:
                rel = path.relative_to(base.resolve())
            except ValueError:
                rel = path
            options.append((f"{infer_run_name(path)}  ·  {rel.parent}", str(path)))
        self._set_file_options(options)

        if len(matches) == 1:
            self.file_dropdown.value = str(matches[0])  # triggers load
        elif matches:
            self.set_status(f"Found {len(matches)} result files. Select one under <b>File</b>.", "success")
        else:
            self.set_status("No <code>*_results.hdf5</code> files found under this path.", "warning")

    def _set_file_options(self, options: list[tuple[str, str]]) -> None:
        self._syncing = True
        try:
            self.file_dropdown.options = [("— select a run —", None), *options]
            self.file_dropdown.value = None
        finally:
            self._syncing = False

    def _on_file_change(self, change) -> None:
        if not self._syncing and change["new"]:
            self.load(change["new"])

    # ----------------------------------------------------------------- loading
    @_reports_errors
    def load(self, path: str | Path) -> None:
        self.set_status(f"Loading <code>{html.escape(Path(path).name)}</code>…", "busy")
        try:
            ds = load_dataset(path)
        except Exception as exc:
            self.show_error(f"Could not load {Path(path).name}: {exc}")
            return

        self.dataset = ds
        self.run_html.value = lp.run_info_html(ds)
        try:
            self._build_figures(ds)
        except Exception as exc:
            self.show_error(f"Failed to build the figures: {exc}")
            return

        self._syncing = True
        try:
            quantities = ds.available_quantities()
            self.quantity_dropdown.options = [(q.label, q.key) for q in quantities]
            self.quantity_dropdown.value = "vel" if "vel" in {q.key for q in quantities} else quantities[0].key
            self.quantity_dropdown.disabled = ds.layout_kind == "single"
            self.bin_dropdown.options = [
                (f"bin {i}" + ("" if ds.has_result[i] else "  (no result)"), i) for i in range(ds.nbins)
            ]
        finally:
            self._syncing = False

        self.view_controls.layout.display = "flex"
        self.refresh_map()
        self.select_bin(ds.center_bin)

        n_missing = int((~ds.has_result).sum())
        message = f"Loaded <b>{html.escape(ds.run_name)}</b> ({html.escape(ds.source)})."
        if ds.layout_kind == "map":
            message += " Click a bin on the map to inspect it."
        if n_missing:
            self.set_status(message + f" <b>{n_missing}</b> of {ds.nbins} bins have no result (marked ×).", "warning")
        else:
            self.set_status(message, "success")

    def _build_figures(self, ds: LosvdDataset) -> None:
        fw = go.FigureWidget
        map_fig = lp.build_map_figure(ds, fw)
        losvd_fig = lp.build_losvd_figure(fw)
        spectrum_fig = lp.build_spectrum_figure(ds, fw)
        weights_fig = lp.build_weights_figure(fw)
        self.figures = {"map": map_fig, "losvd": losvd_fig, "spectrum": spectrum_fig, "weights": weights_fig}
        for fig in self.figures.values():
            # FigureWidget's own resize hook never fires under Voila; let plotly.js follow the window instead.
            fig._config = {**fig._config, "responsive": True, "displaylogo": False}

        for trace in map_fig.data:
            if trace.name in {"map", "profile", "empty"}:
                trace.on_click(self._on_map_click)
                trace.on_hover(self._on_map_hover)

        # Column flow stretches each figure to its grid cell; in a row it keeps plotly's 700 px default.
        panel = widgets.Layout(min_width="0", width="auto", flex_flow="column")
        top = [widgets.Box([losvd_fig], layout=panel)]
        if ds.layout_kind != "single":
            top.insert(0, widgets.Box([map_fig], layout=panel))
        self.body.children = [
            _panel_grid(top, "440px"),
            _panel_grid([self.bin_html, widgets.Box([weights_fig], layout=panel)], "360px"),
            _panel_grid([widgets.Box([spectrum_fig], layout=panel)], "300px"),
        ]

    # ----------------------------------------------------------- interaction
    @_reports_errors
    def refresh_map(self) -> None:
        if self.dataset is None or self._syncing or self.quantity_dropdown.value is None:
            return
        key = self.quantity_dropdown.value
        self.vsys_checkbox.disabled = key != "vel"
        self.stat_toggle.disabled = not MAP_QUANTITY_BY_KEY[key].uses_stat
        map_fig = self.figures["map"]
        lp.update_map_quantity(map_fig, self.dataset, key, self.stat_toggle.value, self.vsys_checkbox.value)
        if self.current_bin is not None:
            lp.update_map_selection(map_fig, self.dataset, self.current_bin)

    @_reports_errors
    def select_bin(self, idx: int) -> None:
        ds = self.dataset
        if ds is None:
            return
        idx = int(idx)
        self.current_bin = idx
        view = ds.bin_view(idx)
        lp.update_map_selection(self.figures["map"], ds, idx)
        lp.update_losvd(self.figures["losvd"], view)
        lp.update_spectrum(self.figures["spectrum"], view)
        lp.update_weights(self.figures["weights"], view)
        self.bin_html.value = lp.bin_info_html(view)

        results = ds.result_bins
        self.prev_button.disabled = bool(results.size == 0 or idx <= results.min())
        self.next_button.disabled = bool(results.size == 0 or idx >= results.max())
        if self.bin_dropdown.value != idx:
            self._syncing = True
            try:
                self.bin_dropdown.value = idx
            finally:
                self._syncing = False

    @_reports_errors
    def step(self, direction: int) -> None:
        if self.dataset is None or self.current_bin is None:
            return
        results = self.dataset.result_bins
        candidates = results[results > self.current_bin] if direction > 0 else results[results < self.current_bin][::-1]
        if candidates.size:
            self.select_bin(int(candidates[0]))

    def _bin_from_points(self, trace, points) -> int | None:
        if not points.point_inds:
            return None
        if trace.name == "map":
            return self.dataset.grid.bin_at(points.xs[0], points.ys[0])
        return int(trace.customdata[points.point_inds[0]])

    @_reports_errors
    def _on_map_click(self, trace, points, _state) -> None:
        idx = self._bin_from_points(trace, points)
        if idx is not None and idx != self.current_bin:
            self.select_bin(idx)

    def _on_map_hover(self, trace, points, state) -> None:
        if self.hover_checkbox.value:
            self._on_map_click(trace, points, state)

    def _on_bin_dropdown(self, change) -> None:
        if not self._syncing and change["new"] is not None and change["new"] != self.current_bin:
            self.select_bin(change["new"])

    # ----------------------------------------------------------------- export
    def _export_kwargs(self) -> dict[str, object]:
        return dict(
            quantity=self.quantity_dropdown.value or "vel",
            stat=self.stat_toggle.value,
            subtract_median=self.vsys_checkbox.value,
            fmt=self.format_dropdown.value,
        )

    @_reports_errors
    def save_current(self) -> None:
        if self.dataset is None or self.current_bin is None:
            return
        self.set_status(f"Saving bin {self.current_bin}…", "busy")
        try:
            path = losvd_static.save_bin_figure(self.dataset, self.current_bin, **self._export_kwargs())
        except Exception as exc:
            self.show_error(f"Export failed: {exc}")
            return
        self.set_status(f"Saved <code>{html.escape(str(path))}</code>", "success")

    @_reports_errors
    def save_all(self) -> None:
        ds = self.dataset
        if ds is None:
            return
        bins = ds.result_bins
        self.progress.max = max(int(bins.size), 1)
        self.progress.value = 0
        self.progress.layout.visibility = "visible"
        self.save_all_button.disabled = True
        self.set_status(f"Saving {bins.size} bins to <code>{html.escape(str(ds.path.parent))}</code>… (the explorer is busy until this finishes)", "busy")
        try:
            for count, idx in enumerate(bins, start=1):
                losvd_static.save_bin_figure(ds, int(idx), **self._export_kwargs())
                self.progress.value = count
        except Exception as exc:
            self.show_error(f"Export stopped at bin {idx}: {exc}")
            return
        finally:
            self.save_all_button.disabled = False
            self.progress.layout.visibility = "hidden"
        skipped = ds.nbins - bins.size
        note = f" Skipped {skipped} bins without results." if skipped else ""
        self.set_status(f"Saved {bins.size} figures to <code>{html.escape(str(ds.path.parent))}</code>.{note}", "success")


def launch_app(path: str | Path | None = None) -> ExplorerApp | None:
    """Display the explorer. ``path`` may be a results directory or a single HDF5 file."""
    if FIGURE_WIDGET_ERROR is not None:
        display(widgets.HTML(
            "<div style='background:#fef2f2;border:1px solid #fecaca;color:#b91c1c;border-radius:8px;padding:10px'>"
            "plotly <code>FigureWidget</code> is not available "
            f"(<code>{html.escape(repr(FIGURE_WIDGET_ERROR))}</code>). Install the missing packages into the "
            "environment, e.g. <code>conda install -c conda-forge plotly anywidget</code>, and restart.</div>"
        ))
        return None
    app = ExplorerApp(path)
    display(app.root)
    app.scan()
    return app
