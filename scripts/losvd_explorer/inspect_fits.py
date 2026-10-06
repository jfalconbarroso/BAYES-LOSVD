"""Command-line replacement for ``bayes_losvd_inspect_fits.py``.

Examples::

    python inspect_fits.py -f ../../results/RUN/RUN_results.hdf5            # show central bin
    python inspect_fits.py -f RUN_results.hdf5 -l 12                         # show bin 12
    python inspect_fits.py -f RUN_results.hdf5 -l all -s                     # PNG for every bin
    python inspect_fits.py -f RUN_results.hdf5 -l 0,5,10-20 -s --format pdf --map sigma

Figures are written as ``<file stem>_bin<N>.<fmt>`` next to the HDF5 file (or into
``--outdir``), exactly like the original script.
"""
from __future__ import annotations

import argparse
import sys
import time

from losvd_data import MAP_QUANTITY_BY_KEY, DatasetError, LosvdDataset, load_dataset


def parse_bins(spec: str, ds: LosvdDataset) -> list[int]:
    """``center``, ``all``, ``12`` or comma-separated ranges like ``0,5,10-20``."""
    spec = spec.strip().lower()
    if spec == "center":
        return [ds.center_bin]
    if spec == "all":
        return list(range(ds.nbins))
    bins: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(v) for v in part.split("-", 1))
            bins.extend(range(start, stop + 1))
        else:
            bins.append(int(part))
    invalid = [b for b in bins if not 0 <= b < ds.nbins]
    if invalid:
        raise ValueError(f"Bin(s) {invalid} outside 0..{ds.nbins - 1}")
    return bins


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Bayes-LOSVD fit diagnostics (map, LOSVD, spectrum, residuals) per bin.",
    )
    parser.add_argument("-f", "--filename", required=True, help="*_results.hdf5 file (packed or with *_bin<N>.hdf5 files)")
    parser.add_argument("-l", "--bin", default="center", help="'center' (default), 'all', a bin ID or a list like '0,5,10-20'")
    parser.add_argument("-s", "--save", action="store_true", help="save figures instead of showing them (implied for >1 bin)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="output format (default: png)")
    parser.add_argument("--outdir", default=None, help="output directory (default: next to the HDF5 file)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--map", default="vel", choices=sorted(MAP_QUANTITY_BY_KEY), help="quantity shown in the map panel")
    parser.add_argument("--stat", default="median", choices=["median", "mean"], help="posterior statistic for the map")
    parser.add_argument("--subtract-vsys", action="store_true", help="subtract the median velocity in the map")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        ds = load_dataset(args.filename)
        bins = parse_bins(args.bin, ds)
    except (FileNotFoundError, DatasetError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"{ds.run_name}: {ds.nbins} bins ({int(ds.has_result.sum())} with results), source: {ds.source}")
    save = args.save or len(bins) > 1

    import matplotlib

    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import losvd_static

    render_kwargs = dict(quantity=args.map, stat=args.stat, subtract_median=args.subtract_vsys)
    skipped: list[int] = []
    written = 0
    start = time.time()
    for count, idx in enumerate(bins, start=1):
        if not ds.has_result[idx]:
            skipped.append(idx)
            continue
        if save:
            path = losvd_static.save_bin_figure(ds, idx, outdir=args.outdir, fmt=args.format, dpi=args.dpi, **render_kwargs)
            written += 1
            if len(bins) <= 20 or count % 25 == 0 or count == len(bins):
                print(f"  [{count}/{len(bins)}] {path}")
        else:
            fig = plt.figure(figsize=losvd_static.FIGSIZE, layout="constrained")
            losvd_static.render_bin_figure(ds, idx, fig=fig, **render_kwargs)
            written += 1
    if skipped:
        listed = ", ".join(map(str, skipped[:10])) + (", …" if len(skipped) > 10 else "")
        print(f"WARNING: skipped {len(skipped)} bin(s) without results: {listed}")
    if save:
        print(f"Wrote {written} figure(s) in {time.time() - start:.1f} s")
    elif written:
        plt.show()
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
