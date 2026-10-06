"""Loading and per-bin preparation of Bayes-LOSVD ``*_results.hdf5`` products.

This module has no plotting dependencies. Both the interactive explorer
(``losvd_plotly.py``) and the static export (``losvd_static.py``) consume the
objects defined here, so every number shown on screen and in exported figures
comes from the same code path.

Layout of the ``out/`` arrays (see ``lib.misc_functions.save_percentiles_to_hdf5``):
axis 1 holds the quantiles ``QUANTILE_LEVELS`` followed by the posterior mean and
standard deviation, i.e. ``(nbins, 7, ...)``.
"""
from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

QUANTILE_LEVELS = (0.1, 15.9, 50.0, 84.1, 99.9)
Q_LO, Q_16, Q_MED, Q_84, Q_HI, Q_MEAN, Q_STD = range(7)
N_SUMMARY = 7
MODEL_SNR_CAP = 100.0  # models/bayes_losvd_model_*.py: sigma = 1 / min(bin_snr, 100)

REQUIRED_IN = ("xbin", "ybin", "xvel")
REQUIRED_OUT = ("losvd",)
SPECTRAL_OUT = ("model_spec", "continuum")

_BIN_FILE_RE = re.compile(r"_bin(\d+)$")
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 compatibility


class DatasetError(RuntimeError):
    """Raised when an HDF5 file cannot be interpreted as a Bayes-LOSVD result."""


# =============================================================================
# File discovery
# =============================================================================
def detect_default_search_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        results_dir = candidate / "results"
        if results_dir.is_dir():
            return results_dir.resolve()
    return Path.cwd().resolve()


def infer_run_name(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_results")] if stem.endswith("_results") else stem


def resolve_hdf5_candidates(path_text: str | Path, pattern: str = "*_results.hdf5") -> list[Path]:
    """Return the result files for a directory (searched recursively) or a single file."""
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in {".hdf5", ".h5"}:
            raise ValueError(f"Path is not an HDF5 file: {path}")
        return [path]
    return sorted(p for p in path.rglob(pattern) if p.is_file())


def per_bin_files(path: Path) -> dict[int, Path]:
    """Unpacked per-bin result files ``<stem>_bin<idx><suffix>`` next to ``path``."""
    files: dict[int, Path] = {}
    for candidate in path.parent.glob(f"{path.stem}_bin*{path.suffix}"):
        match = _BIN_FILE_RE.search(candidate.stem)
        if match:
            files[int(match.group(1))] = candidate
    return dict(sorted(files.items()))


# =============================================================================
# Physics helpers
# =============================================================================
@lru_cache(maxsize=1)
def pipeline_gauss_hermite() -> Callable | None:
    """``gauss_hermite`` from the pipeline's ``scripts/lib/gauss_hermite_fit.py``.

    The explorer draws the Gauss-Hermite profile with exactly the function that produced
    ``vel_star``/``sigma_star``/``h3_star``/``h4_star``, so it follows any change made there.
    Returns ``None`` (no GH overlay) if the pipeline library cannot be imported.
    """
    scripts_dir = str(Path(__file__).resolve().parent.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from lib.gauss_hermite_fit import gauss_hermite
    except Exception as exc:  # e.g. explorer used outside the repository
        warnings.warn(f"Gauss-Hermite overlay disabled: cannot import lib.gauss_hermite_fit ({exc})")
        return None
    return gauss_hermite


def excluded_spans(wave: np.ndarray, good: np.ndarray) -> list[tuple[float, float]]:
    """Wavelength intervals of consecutive pixels that were excluded from the fit."""
    if wave.size == 0:
        return []
    half_step = 0.5 * np.diff(wave)
    left = wave - np.concatenate([[half_step[0] if half_step.size else 0.5], half_step])
    right = wave + np.concatenate([half_step, [half_step[-1] if half_step.size else 0.5]])
    bad = ~good
    edges = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(int), [0]])))
    return [(float(left[start]), float(right[stop - 1])) for start, stop in zip(edges[::2], edges[1::2])]


# =============================================================================
# Pixel grid of the bins
# =============================================================================
@dataclass
class BinGrid:
    x_centers: np.ndarray
    y_centers: np.ndarray
    bin_image: np.ndarray  # (ny, nx), -1 where no bin
    dx: float
    dy: float

    @classmethod
    def from_pixels(cls, x: np.ndarray, y: np.ndarray, bin_num: np.ndarray, n_bins: int, psize: float | None) -> "BinGrid":
        def step(values: np.ndarray) -> float:
            if psize is not None and np.isfinite(psize) and psize > 0:
                return float(psize)
            diffs = np.diff(np.unique(values))
            diffs = diffs[diffs > 1e-9]
            return float(np.median(diffs)) if diffs.size else 1.0

        dx, dy = step(x), step(y)
        ix = np.rint((x - x.min()) / dx).astype(int)
        iy = np.rint((y - y.min()) / dy).astype(int)
        nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
        image = np.full((ny, nx), -1, dtype=int)
        valid = (bin_num >= 0) & (bin_num < n_bins)
        image[iy[valid], ix[valid]] = bin_num[valid]
        return cls(
            x_centers=x.min() + dx * np.arange(nx),
            y_centers=y.min() + dy * np.arange(ny),
            bin_image=image,
            dx=dx,
            dy=dy,
        )

    def image_of(self, values: np.ndarray) -> np.ndarray:
        image = np.full(self.bin_image.shape, np.nan)
        valid = self.bin_image >= 0
        image[valid] = np.asarray(values, dtype=float)[self.bin_image[valid]]
        return image

    def bin_at(self, x: float, y: float) -> int | None:
        ix = int(np.rint((x - self.x_centers[0]) / self.dx))
        iy = int(np.rint((y - self.y_centers[0]) / self.dy))
        if 0 <= iy < self.bin_image.shape[0] and 0 <= ix < self.bin_image.shape[1]:
            value = int(self.bin_image[iy, ix])
            return value if value >= 0 else None
        return None

    def outline(self, only_bin: int | None = None) -> tuple[list[float | None], list[float | None]]:
        """Line segments separating different bins (or outlining ``only_bin``).

        Returned as x/y lists with ``None`` separators, ready for a single line trace.
        """
        padded = np.pad(self.bin_image, 1, constant_values=-1)
        if only_bin is not None:
            padded = np.where(padded == only_bin, 1, 0)
        xs: list[float | None] = []
        ys: list[float | None] = []
        x0 = self.x_centers[0] - self.dx  # centre of padded column 0
        y0 = self.y_centers[0] - self.dy

        # Vertical edges: between columns j and j+1 of the padded image.
        diff_v = padded[:, :-1] != padded[:, 1:]
        if only_bin is None:
            diff_v &= (padded[:, :-1] >= 0) | (padded[:, 1:] >= 0)
        for i, j in zip(*np.nonzero(diff_v)):
            xe = x0 + (j + 0.5) * self.dx
            yc = y0 + i * self.dy
            xs += [xe, xe, None]
            ys += [yc - 0.5 * self.dy, yc + 0.5 * self.dy, None]

        # Horizontal edges: between rows i and i+1.
        diff_h = padded[:-1, :] != padded[1:, :]
        if only_bin is None:
            diff_h &= (padded[:-1, :] >= 0) | (padded[1:, :] >= 0)
        for i, j in zip(*np.nonzero(diff_h)):
            ye = y0 + (i + 0.5) * self.dy
            xc = x0 + j * self.dx
            xs += [xc - 0.5 * self.dx, xc + 0.5 * self.dx, None]
            ys += [ye, ye, None]
        return xs, ys


# =============================================================================
# Map quantities
# =============================================================================
@dataclass(frozen=True)
class MapQuantity:
    key: str
    label: str
    unit: str
    kind: str  # "diverging" (symmetric around 0) or "sequential"
    getter: Callable[["LosvdDataset", str], np.ndarray | None]
    uses_stat: bool = True

    @property
    def axis_title(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label


def _param_value(name: str) -> Callable[["LosvdDataset", str], np.ndarray | None]:
    def getter(ds: "LosvdDataset", stat: str) -> np.ndarray | None:
        summary = ds.param(name)
        if summary is None:
            return None
        return summary[:, Q_MEAN] if stat == "mean" else summary[:, Q_MED]
    return getter


def _param_error(name: str) -> Callable[["LosvdDataset", str], np.ndarray | None]:
    def getter(ds: "LosvdDataset", stat: str) -> np.ndarray | None:
        summary = ds.param(name)
        if summary is None:
            return None
        return summary[:, Q_STD] if stat == "mean" else 0.5 * (summary[:, Q_84] - summary[:, Q_16])
    return getter


MAP_QUANTITIES: tuple[MapQuantity, ...] = (
    MapQuantity("vel", "Velocity", "km/s", "diverging", _param_value("vel_star")),
    MapQuantity("sigma", "σ", "km/s", "sequential", _param_value("sigma_star")),
    MapQuantity("h3", "h3", "", "diverging", _param_value("h3_star")),
    MapQuantity("h4", "h4", "", "diverging", _param_value("h4_star")),
    MapQuantity("dvel", "Δ velocity", "km/s", "sequential", _param_error("vel_star")),
    MapQuantity("dsigma", "Δ σ", "km/s", "sequential", _param_error("sigma_star")),
    MapQuantity("dh3", "Δ h3", "", "sequential", _param_error("h3_star")),
    MapQuantity("dh4", "Δ h4", "", "sequential", _param_error("h4_star")),
    MapQuantity("snr_fit", "S/N (fit)", "", "sequential", _param_value("snr_real")),
    MapQuantity("snr_in", "S/N (input bins)", "", "sequential", lambda ds, _s: ds.bin_snr, uses_stat=False),
    MapQuantity("flux", "Bin flux", "", "sequential", lambda ds, _s: ds.bin_flux, uses_stat=False),
    MapQuantity("chi2", "χ²_red", "", "sequential", lambda ds, _s: ds.fit_stats[0] if ds.fit_stats else None, uses_stat=False),
    MapQuantity("rms", "Residual rms", "", "sequential", lambda ds, _s: ds.fit_stats[1] if ds.fit_stats else None, uses_stat=False),
    MapQuantity("ell_gp", "GP length scale ℓ", "", "sequential", _param_value("ell_gp")),
    MapQuantity("sigma_gp", "GP amplitude σ_GP", "", "sequential", _param_value("sigma_gp")),
)
MAP_QUANTITY_BY_KEY = {q.key: q for q in MAP_QUANTITIES}


def color_limits(values: np.ndarray, kind: str) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0) if kind == "diverging" else (0.0, 1.0)
    if kind == "diverging":
        vmax = max(float(np.percentile(np.abs(finite), 99.0)), 1e-6)
        return -vmax, vmax
    lo, hi = (float(v) for v in np.percentile(finite, [1.0, 99.0]))
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-3, 1e-6)
    return lo, hi


# =============================================================================
# Dataset
# =============================================================================
@dataclass
class BinView:
    """Everything needed to draw one spatial bin."""

    idx: int
    has_result: bool
    x: float
    y: float
    xvel: np.ndarray
    losvd: np.ndarray  # (7, nvel)
    gh_model: np.ndarray | None  # GH profile from median parameters, same normalisation as losvd
    params: dict[str, np.ndarray]  # name -> (7,)
    wave: np.ndarray | None = None
    spec: np.ndarray | None = None
    noise: np.ndarray | None = None
    model: np.ndarray | None = None  # (7, npix)
    continuum: np.ndarray | None = None  # (7, npix)
    residual: np.ndarray | None = None  # NaN outside the fitted mask
    spans: list[tuple[float, float]] = field(default_factory=list)
    chi2_red: float = np.nan
    rms: float = np.nan
    weights: np.ndarray | None = None  # (7, ntemp)
    bin_snr: float = np.nan
    bin_flux: float = np.nan

    @property
    def radius(self) -> float:
        return float(np.hypot(self.x, self.y))

    @property
    def has_spectrum(self) -> bool:
        return self.wave is not None and self.model is not None and self.has_result


@dataclass
class LosvdDataset:
    path: Path
    source: str
    ndim: int
    xbin: np.ndarray
    ybin: np.ndarray
    xvel: np.ndarray
    velscale: float
    out: dict[str, np.ndarray]
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    pixel_bin: np.ndarray | None = None
    psize: float | None = None
    bin_flux: np.ndarray | None = None
    bin_snr: np.ndarray | None = None
    wave_obs: np.ndarray | None = None
    spec_obs: np.ndarray | None = None  # (npix, nbins)
    sigma_obs: np.ndarray | None = None  # (npix, nbins)
    mask: np.ndarray | None = None  # indices of fitted pixels

    # ---------------------------------------------------------------- basics
    @property
    def run_name(self) -> str:
        return infer_run_name(self.path)

    @property
    def nbins(self) -> int:
        return int(self.xbin.size)

    @cached_property
    def has_result(self) -> np.ndarray:
        return ~np.all(np.isnan(self.out["losvd"][:, Q_MED, :]), axis=1)

    @property
    def result_bins(self) -> np.ndarray:
        return np.flatnonzero(self.has_result)

    @cached_property
    def center_bin(self) -> int:
        candidates = np.flatnonzero(self.has_result & np.isfinite(self.xbin) & np.isfinite(self.ybin))
        if candidates.size == 0:
            return 0
        radius2 = self.xbin[candidates] ** 2 + self.ybin[candidates] ** 2
        return int(candidates[np.argmin(radius2)])

    @cached_property
    def grid(self) -> BinGrid | None:
        """Pixel grid for 2D data; ``None`` for long-slit / single-spectrum runs."""
        if self.ndim < 2 or self.x is None or self.y is None or self.pixel_bin is None:
            return None
        return BinGrid.from_pixels(self.x, self.y, self.pixel_bin, self.nbins, self.psize)

    @cached_property
    def bin_outline(self) -> tuple[list[float | None], list[float | None]] | None:
        return self.grid.outline() if self.grid is not None else None

    @property
    def layout_kind(self) -> str:
        if self.grid is not None:
            return "map"
        return "profile" if self.nbins > 1 else "single"

    @property
    def profile_coordinate(self) -> np.ndarray:
        """Signed position along the slit for 1D data."""
        return self.xbin if np.ptp(self.xbin) >= np.ptp(self.ybin) else self.ybin

    def param(self, name: str) -> np.ndarray | None:
        """Posterior summary ``(nbins, 7)`` of a scalar parameter, if stored."""
        values = self.out.get(name)
        if values is None or values.ndim != 2 or values.shape[1] != N_SUMMARY:
            return None
        return values

    @property
    def has_spectra(self) -> bool:
        return (
            self.wave_obs is not None
            and self.spec_obs is not None
            and self.mask is not None
            and all(key in self.out for key in SPECTRAL_OUT)
        )

    @cached_property
    def good_pixels(self) -> np.ndarray | None:
        if self.wave_obs is None or self.mask is None:
            return None
        good = np.zeros(self.wave_obs.size, dtype=bool)
        good[self.mask[(self.mask >= 0) & (self.mask < good.size)]] = True
        return good

    @cached_property
    def noise(self) -> np.ndarray | None:
        """Per-pixel noise ``(npix, nbins)`` that the likelihood actually used.

        The NumPyro models replace ``in/sigma_obs`` by the constant ``1/min(bin_snr, 100)``,
        so χ² and the residual noise band are computed with that; ``sigma_obs`` is only a
        fallback for files without ``bin_snr``.
        """
        if self.bin_snr is not None and self.wave_obs is not None:
            with np.errstate(divide="ignore"):
                per_bin = 1.0 / np.minimum(self.bin_snr, MODEL_SNR_CAP)
            return np.broadcast_to(per_bin, (self.wave_obs.size, self.nbins))
        return self.sigma_obs

    @cached_property
    def fit_stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-bin reduced χ² and residual rms over the fitted pixels."""
        if not self.has_spectra:
            return None
        good = self.good_pixels
        residual = self.spec_obs.T[:, good] - self.out["model_spec"][:, Q_MED, :][:, good]
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows of unfitted bins
            rms = np.sqrt(np.nanmean(residual**2, axis=1))
            if self.noise is not None:
                chi2 = np.nanmean((residual / self.noise.T[:, good]) ** 2, axis=1)
            else:
                chi2 = np.full(self.nbins, np.nan)
        rms[~self.has_result] = np.nan
        chi2[~self.has_result] = np.nan
        return chi2, rms

    def available_quantities(self) -> list[MapQuantity]:
        result = []
        for quantity in MAP_QUANTITIES:
            values = quantity.getter(self, "median")
            if values is not None and np.any(np.isfinite(values)):
                result.append(quantity)
        return result

    def quantity_values(self, key: str, stat: str = "median", subtract_median: bool = False) -> np.ndarray:
        quantity = MAP_QUANTITY_BY_KEY[key]
        values = quantity.getter(self, stat)
        if values is None:
            return np.full(self.nbins, np.nan)
        values = np.array(values, dtype=float)
        if quantity.uses_stat or quantity.key in {"chi2", "rms"}:
            values[~self.has_result] = np.nan  # fit products only exist for fitted bins
        if subtract_median and np.any(np.isfinite(values)):
            values = values - np.nanmedian(values)
        return values

    # --------------------------------------------------------------- per bin
    def bin_view(self, idx: int) -> BinView:
        idx = int(idx)
        if not 0 <= idx < self.nbins:
            raise IndexError(f"Bin {idx} outside 0..{self.nbins - 1}")

        losvd = self.out["losvd"][idx]
        params = {name: self.out[name][idx] for name in self.out if self.param(name) is not None}

        gh_model = None
        if self.has_result[idx] and all(k in params for k in ("vel_star", "sigma_star", "h3_star", "h4_star")):
            vel, sig, h3, h4 = (params[k][Q_MED] for k in ("vel_star", "sigma_star", "h3_star", "h4_star"))
            if np.all(np.isfinite([vel, sig, h3, h4])) and sig > 0:
                gauss_hermite = pipeline_gauss_hermite()
                if gauss_hermite is not None:
                    area = _trapezoid(losvd[Q_MED], self.xvel)
                    gh_model = gauss_hermite(self.xvel, vel, sig, h3, h4) * area

        view = BinView(
            idx=idx,
            has_result=bool(self.has_result[idx]),
            x=float(self.xbin[idx]),
            y=float(self.ybin[idx]),
            xvel=self.xvel,
            losvd=losvd,
            gh_model=gh_model,
            params=params,
            weights=self.out["weights"][idx] if "weights" in self.out and self.out["weights"].ndim == 3 else None,
            bin_snr=float(self.bin_snr[idx]) if self.bin_snr is not None else np.nan,
            bin_flux=float(self.bin_flux[idx]) if self.bin_flux is not None else np.nan,
        )

        if self.has_spectra:
            good = self.good_pixels
            spec = self.spec_obs[:, idx]
            model = self.out["model_spec"][idx]
            residual = np.where(good, spec - model[Q_MED], np.nan)
            view.wave = self.wave_obs
            view.spec = spec
            view.noise = self.noise[:, idx] if self.noise is not None else None
            view.model = model
            view.continuum = self.out["continuum"][idx]
            view.residual = residual
            view.spans = excluded_spans(self.wave_obs, good)
            if self.fit_stats is not None:
                view.chi2_red = float(self.fit_stats[0][idx])
                view.rms = float(self.fit_stats[1][idx])
        return view

    def summary_lines(self) -> list[tuple[str, str]]:
        rows = [
            ("Source", self.source),
            ("Spatial bins", f"{self.nbins} ({int(self.has_result.sum())} with results)"),
            ("Geometry", {"map": f"2D map, {self.grid.bin_image.shape[1]}×{self.grid.bin_image.shape[0]} px" if self.grid else "",
                          "profile": "1D profile", "single": "single spectrum"}[self.layout_kind]),
            ("Velocity grid", f"{self.xvel.size} bins, {self.xvel.min():.0f} … {self.xvel.max():.0f} km/s, Δv = {self.velscale:.1f} km/s"),
        ]
        if self.wave_obs is not None:
            fitted = int(self.good_pixels.sum()) if self.good_pixels is not None else self.wave_obs.size
            rows.append(("Spectrum", f"{self.wave_obs.min():.0f} … {self.wave_obs.max():.0f} Å, {fitted}/{self.wave_obs.size} px fitted"))
        if self.bin_snr is not None and np.any(np.isfinite(self.bin_snr)):
            rows.append(("Input S/N", f"{np.nanmin(self.bin_snr):.1f} … {np.nanmax(self.bin_snr):.1f} (median {np.nanmedian(self.bin_snr):.1f})"))
        if self.fit_stats is not None and np.any(np.isfinite(self.fit_stats[0])):
            rows.append(("χ²_red", f"median {np.nanmedian(self.fit_stats[0]):.2f}"))
        return rows


# =============================================================================
# Loading
# =============================================================================
def _scalar(group: h5py.Group, key: str, default=None):
    if key not in group:
        return default
    return np.asarray(group[key][()]).reshape(-1)[0].item()


def _array(group: h5py.Group, key: str, dtype=float) -> np.ndarray | None:
    return np.asarray(group[key][...], dtype=dtype) if key in group else None


def _read_out(handle: h5py.File, path: Path, nbins: int) -> tuple[dict[str, np.ndarray], str]:
    if "out" in handle:
        out = {k: np.asarray(v[...], dtype=float) for k, v in handle["out"].items() if isinstance(v, h5py.Dataset)}
        return out, "packed results file"

    files = per_bin_files(path)
    files = {idx: f for idx, f in files.items() if 0 <= idx < nbins}
    if not files:
        raise DatasetError(
            f"No 'out' group in {path.name} and no per-bin files '{path.stem}_bin<N>{path.suffix}' next to it. "
            "Has the run finished?"
        )

    out: dict[str, np.ndarray] = {}
    for idx, bin_file in files.items():
        with h5py.File(bin_file, "r") as bin_handle:
            if "out" not in bin_handle:
                continue
            for name, dataset in bin_handle["out"].items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                if name not in out:
                    out[name] = np.full((nbins, *dataset.shape), np.nan)
                out[name][idx] = dataset[...]
    return out, f"unpacked per-bin files ({len(files)}/{nbins} bins present)"


def load_dataset(hdf5_path: str | Path) -> LosvdDataset:
    path = Path(hdf5_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        if "in" not in handle:
            raise DatasetError(f"{path.name} has no 'in' group – not a Bayes-LOSVD results file.")
        group = handle["in"]
        missing = [key for key in REQUIRED_IN if key not in group]
        if missing:
            raise DatasetError(f"{path.name}: missing required input dataset(s): {', '.join('in/' + m for m in missing)}")

        xbin = _array(group, "xbin")
        nbins = xbin.size
        out, source = _read_out(handle, path, nbins)

        wave_log = _array(group, "wave_obs")
        dataset = LosvdDataset(
            path=path,
            source=source,
            ndim=int(_scalar(group, "ndim", 2 if "binID" in group else 1)),
            xbin=xbin,
            ybin=_array(group, "ybin"),
            xvel=_array(group, "xvel"),
            velscale=float(_scalar(group, "velscale", np.nan)),
            out=out,
            x=_array(group, "x"),
            y=_array(group, "y"),
            pixel_bin=_array(group, "binID", dtype=int),
            psize=_scalar(group, "psize"),
            bin_flux=_array(group, "bin_flux"),
            bin_snr=_array(group, "bin_snr"),
            wave_obs=np.exp(wave_log) if wave_log is not None else None,
            spec_obs=_array(group, "spec_obs"),
            sigma_obs=_array(group, "sigma_obs"),
            mask=_array(group, "mask", dtype=int),
        )

    missing = [key for key in REQUIRED_OUT if key not in out]
    if missing:
        raise DatasetError(f"{path.name}: missing required output dataset(s): {', '.join('out/' + m for m in missing)}")
    expected = (nbins, N_SUMMARY, dataset.xvel.size)
    if out["losvd"].shape != expected:
        raise DatasetError(f"{path.name}: out/losvd has shape {out['losvd'].shape}, expected {expected}")
    if dataset.spec_obs is not None and dataset.spec_obs.ndim == 1:
        dataset.spec_obs = dataset.spec_obs[:, None]
    if dataset.sigma_obs is not None and dataset.sigma_obs.ndim == 1:
        dataset.sigma_obs = dataset.sigma_obs[:, None]
    if not np.isfinite(dataset.velscale) and dataset.xvel.size > 1:
        dataset.velscale = float(np.median(np.diff(dataset.xvel)))
    return dataset
