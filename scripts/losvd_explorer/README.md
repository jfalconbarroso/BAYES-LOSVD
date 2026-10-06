# LOSVD Explorer

Interactive viewer and batch plotter for Bayes-LOSVD `*_results.hdf5` products.
Replaces `../bayes_losvd_inspect_fits.py`.

Per spatial bin it shows

- a clickable map of v, σ, h3, h4, their errors, S/N, χ²_red, residual rms or the GP hyper-parameters
  (posterior median or mean, optional v_sys subtraction; a v/σ profile for long-slit data),
- the LOSVD with 68 % and 99.8 % posterior bands and the Gauss-Hermite profile of the median (v, σ, h3, h4),
- the spectral fit with model/continuum bands, excluded regions and a separate residual panel with the ±1σ noise band,
- the template weights and a table of all parameters (median ± ½(p84−p16) and mean ± std).

It reads packed result files as well as unpacked runs (`<run>_results.hdf5` without `out/` plus
`<run>_results_bin<N>.hdf5`), and marks bins without results instead of failing on them.

A walkthrough of the whole pipeline on the NGC0000 example (config → preprocessing → fit → convergence → this explorer →
using the results) is in [`../bayes_losvd_workflow.ipynb`](../bayes_losvd_workflow.ipynb).

## Setup

On top of the BAYES-LOSVD environment (numpy, h5py, matplotlib) the explorer needs plotly ≥ 6,
ipywidgets ≥ 8 and anywidget; the browser app also needs voila:

```bash
conda install -c conda-forge plotly ipywidgets anywidget voila
```

## Interactive explorer

```bash
./start_explorer.sh                                   # scans the nearest results/ directory
./start_explorer.sh ../../results/NGC0000_GP           # scans one directory
./start_explorer.sh path/to/RUN_results.hdf5          # opens one run directly
```

This starts Voila and opens the app in the browser (stop with `Ctrl+C`). Equivalent without the script:
`voila interactive_losvd_explorer.ipynb` (optionally with `LOSVD_EXPLORER_PATH=<dir or file>`).

Inside Jupyter/VS Code you can also run

```python
from losvd_explorer_app import launch_app
app = launch_app("../../results/NGC0000_GP")
```

Controls: **Path** + Enter/**Scan** → **File** → the run loads with the central bin selected.
Click a bin on the map (or use ◀ ▶ / the **Bin** dropdown) to inspect it. Hover only shows a tooltip
unless **Select on hover** is ticked. **Save bin** / **Save all** write PNG or PDF figures next to the
HDF5 file.

## Command line (batch / replacement for `bayes_losvd_inspect_fits.py`)

```bash
python inspect_fits.py -f RUN_results.hdf5              # show the central bin
python inspect_fits.py -f RUN_results.hdf5 -l 12        # show bin 12
python inspect_fits.py -f RUN_results.hdf5 -l all -s    # PNG for every bin with results
python inspect_fits.py -f RUN_results.hdf5 -l 0,5,10-20 -s --format pdf --map sigma --outdir figs/
```

Output names match the old script: `<file stem>_bin<N>.<fmt>` next to the HDF5 file.
`python inspect_fits.py -h` lists all options.

## Files

| File | Purpose |
| --- | --- |
| `losvd_data.py` | Loading (packed + per-bin files), validation, per-bin views, map quantities, fit statistics. No plotting. |
| `losvd_plotly.py` | Interactive plotly panels: `build_*` once per file, `update_*` per selected bin. |
| `losvd_static.py` | PNG/PDF export per bin (map, LOSVD and spectral fit, in the style of the old `inspect_fits` plots). |
| `losvd_explorer_app.py` | ipywidgets/Voila front end (`launch_app`). |
| `inspect_fits.py` | Command-line interface. |
| `interactive_losvd_explorer.ipynb` | Voila entry point. |
| `start_explorer.sh` | Launcher. |

