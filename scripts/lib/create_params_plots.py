import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec


def _centers_to_edges(c):
    """Convert sorted bin centers to edges (for pcolormesh)."""
    c = np.asarray(c)
    if c.size < 2:
        dc = 1.0
        return np.array([c[0] - 0.5 * dc, c[0] + 0.5 * dc])
    dc = np.diff(c)
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = (c[:-1] + c[1:]) / 2.0
    edges[0] = c[0] - dc[0] / 2.0
    edges[-1] = c[-1] + dc[-1] / 2.0
    return edges


def _make_norm(values, log_color=False, vmin=None, vmax=None, percentile_clip=(1, 99)):
    """Create a Normalize/LogNorm based on values and optional vmin/vmax."""
    vals = np.asarray(values, dtype=float)

    if log_color:
        vp = vals[np.isfinite(vals) & (vals > 0)]
        if vp.size == 0:
            raise ValueError("All values are <= 0 or non-finite; cannot use log scaling.")
        if vmin is None:
            vmin = np.nanpercentile(vp, percentile_clip[0])
        if vmax is None:
            vmax = np.nanpercentile(vp, percentile_clip[1])
        vmin = max(vmin, np.min(vp))
        return LogNorm(vmin=vmin, vmax=vmax)

    vp = vals[np.isfinite(vals)]
    if vp.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    if vmin is None:
        vmin = np.nanpercentile(vp, percentile_clip[0])
    if vmax is None:
        vmax = np.nanpercentile(vp, percentile_clip[1])
    return Normalize(vmin=vmin, vmax=vmax)


def tiles_weight_and_uncertainty(
    params,
    weights,
    uncertainties,
    labels=None,
    reducer="sum",                  # "sum" or "max" if duplicates exist
    cmap_w="Blues",
    cmap_u="Oranges",
    log_color_w=False,
    log_color_u=False,
    vmin_w=None, vmax_w=None,
    vmin_u=None, vmax_u=None,
    show_colorbar=True,
    figsize=None,
    # Spacing controls (CLI-friendly)
    wspace=0.25,
    hspace=0.35,
    left=0.07,
    right=0.97,
    top=0.92,
    bottom=0.12,
    # Colorbar geometry
    cbar_width=0.05,                # relative width compared to one panel
    cbar_pad=0.10,                  # extra space between last panel and colorbar column (as wspace in GridSpec)
    missing_value=np.nan,           # np.nan recommended so missing combos appear blank
    uncertainty_reducer="sum",       # "sum", "quadrature", or "max" (see below)
):
    """
    Layout: 2 rows x K columns (K = number of unique parameter pairs i>j).
        - Row 0: weights tile heatmap for each pair
        - Row 1: uncertainties tile heatmap for each pair
        - Dedicated colorbar column on the right (no overlap with panels)

    params:        (N, D) parameter combinations (often bin centers)
    weights:       (N,)   weight/intensity for each combination
    uncertainties: (N,)   uncertainty for each combination (aligned with weights)

    reducer: aggregation for WEIGHTS when multiple samples map to the same (x,y) cell:
        - "sum" or "max"

    uncertainty_reducer: aggregation for UNCERTAINTIES when duplicates exist (only relevant if reducer="sum"):
        - "sum":         sum uncertainties directly
        - "quadrature":  sqrt(sum(u^2)) per cell (common for independent errors)
        - "max":         cell-wise max of uncertainties (typically for reducer="max" scenarios)

    Returns: fig, axes(2,K), pairs(list of (i,j))
    """
    params = np.asarray(params)
    weights = np.asarray(weights)
    uncertainties = np.asarray(uncertainties)

    if params.ndim != 2:
        raise ValueError("params must be 2D: (ncomb, npars)")
    if weights.ndim != 1 or weights.shape[0] != params.shape[0]:
        raise ValueError("weights must be 1D with same length as params rows")
    if uncertainties.ndim != 1 or uncertainties.shape[0] != params.shape[0]:
        raise ValueError("uncertainties must be 1D with same length as params rows")

    N, D = params.shape
    if D < 2:
        raise ValueError("Need at least D=2 parameters to form 2D combinations (i>j).")

    if labels is None:
        labels = [f"par{i}" for i in range(D)]
    if len(labels) != D:
        raise ValueError("labels must have length npars")

    if reducer not in ("sum", "max"):
        raise ValueError("reducer must be 'sum' or 'max'")
    if uncertainty_reducer not in ("sum", "quadrature", "max"):
        raise ValueError("uncertainty_reducer must be 'sum', 'quadrature', or 'max'")

    # Column-defining parameter pairs (i>j)
    pairs = [(i, j) for i in range(D) for j in range(D) if i > j]
    K = len(pairs)

    if figsize is None:
        figsize = (3.0 * K, 6.0)

    # Norms per row
    norm_w = _make_norm(weights, log_color=log_color_w, vmin=vmin_w, vmax=vmax_w)
    norm_u = _make_norm(uncertainties, log_color=log_color_u, vmin=vmin_u, vmax=vmax_u)

    fig = plt.figure(figsize=figsize)

    # Reserve an explicit last column for colorbars.
    # width_ratios = [1,1,1,..., cbar_width]
    width_ratios = [1.0] * K + [float(cbar_width)]
    gs = GridSpec(
        2, K + 1,
        figure=fig,
        width_ratios=width_ratios,
        wspace=wspace,
        hspace=hspace,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )

    # If you want extra separation between plots and colorbar column, increase wspace,
    # or add a dedicated "pad column". Here is a simple approach: we increase wspace
    # locally by inserting a narrow spacer column before the cbar column.
    # To keep the API simple, we implement it by scaling the last-gap effect via cbar_pad.
    # If cbar_pad != 0, we emulate a spacer column.
    if cbar_pad and cbar_pad > 0:
        # Rebuild GridSpec with a spacer column before the cbar
        width_ratios = [1.0] * K + [float(cbar_pad), float(cbar_width)]
        gs = GridSpec(
            2, K + 2,
            figure=fig,
            width_ratios=width_ratios,
            wspace=wspace,
            hspace=hspace,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        cbar_col = K + 1
        spacer_col = K
    else:
        cbar_col = K
        spacer_col = None

    axes = np.empty((2, K), dtype=object)

    # Create plot axes
    for col in range(K):
        axes[0, col] = fig.add_subplot(gs[0, col])
        axes[1, col] = fig.add_subplot(gs[1, col])

    # Create dedicated colorbar axes
    cax_w = fig.add_subplot(gs[0, cbar_col])
    cax_u = fig.add_subplot(gs[1, cbar_col])

    # If we inserted a spacer column, turn it off
    if spacer_col is not None:
        ax_spacer_top = fig.add_subplot(gs[0, spacer_col])
        ax_spacer_bot = fig.add_subplot(gs[1, spacer_col])
        ax_spacer_top.axis("off")
        ax_spacer_bot.axis("off")

    last_im_w = None
    last_im_u = None

    w = weights.astype(float)
    u = uncertainties.astype(float)

    for col, (i, j) in enumerate(pairs):
        ax_w = axes[0, col]
        ax_u = axes[1, col]

        x = params[:, j]
        y = params[:, i]

        xu = np.unique(x)
        yu = np.unique(y)

        x_index = {val: idx for idx, val in enumerate(xu)}
        y_index = {val: idx for idx, val in enumerate(yu)}

        if np.isnan(missing_value):
            Mw = np.full((yu.size, xu.size), np.nan, dtype=float)
            Mu = np.full((yu.size, xu.size), np.nan, dtype=float)
        else:
            Mw = np.full((yu.size, xu.size), missing_value, dtype=float)
            Mu = np.full((yu.size, xu.size), missing_value, dtype=float)

        if reducer == "sum":
            # Accumulate weights
            if np.isnan(missing_value):
                Mw[:] = 0.0
                if uncertainty_reducer in ("sum", "quadrature"):
                    Mu[:] = 0.0
                else:
                    Mu[:] = -np.inf

            for xv, yv, ww, uu in zip(x, y, w, u):
                yi = y_index[yv]
                xi = x_index[xv]
                Mw[yi, xi] += ww

                if uncertainty_reducer == "sum":
                    Mu[yi, xi] += uu
                elif uncertainty_reducer == "quadrature":
                    Mu[yi, xi] = np.sqrt(Mu[yi, xi] ** 2 + uu ** 2)
                else:  # "max"
                    Mu[yi, xi] = max(Mu[yi, xi], uu)

            # If you want NaNs for truly-missing cells in sum mode, track counts and mask.
            # This implementation shows 0 where nothing accumulated when missing_value is NaN.

        else:  # reducer == "max"
            Mw[:] = -np.inf
            Mu[:] = -np.inf
            for xv, yv, ww, uu in zip(x, y, w, u):
                yi = y_index[yv]
                xi = x_index[xv]
                if ww > Mw[yi, xi]:
                    Mw[yi, xi] = ww
                    # Often you'd want uncertainty from the same entry as the max-weight entry:
                    Mu[yi, xi] = uu
                else:
                    # Alternatively, keep max uncertainty; uncomment if desired:
                    # Mu[yi, xi] = max(Mu[yi, xi], uu)
                    pass

            Mw[Mw == -np.inf] = missing_value
            Mu[Mu == -np.inf] = missing_value

        xe = _centers_to_edges(xu)
        ye = _centers_to_edges(yu)

        # Top row: weights
        imw = ax_w.pcolormesh(xe, ye, Mw, shading="auto", cmap=cmap_w, norm=norm_w)
        last_im_w = imw
        ax_w.set_xlabel(labels[j])
        ax_w.set_ylabel(labels[i])

        # Bottom row: uncertainties
        imu = ax_u.pcolormesh(xe, ye, Mu, shading="auto", cmap=cmap_u, norm=norm_u)
        last_im_u = imu
        ax_u.set_xlabel(labels[j])
        ax_u.set_ylabel(labels[i])

        # Optional tick de-clutter for many columns
        if K > 6:
            ax_w.tick_params(axis="both", labelsize=8)
            ax_u.tick_params(axis="both", labelsize=8)

    # Dedicated colorbars (won't overlap plots)
    if show_colorbar:
        if last_im_w is not None:
            cbw = fig.colorbar(last_im_w, cax=cax_w)
            cbw.set_label("Weights mean")
        else:
            cax_w.axis("off")

        if last_im_u is not None:
            cbu = fig.colorbar(last_im_u, cax=cax_u)
            cbu.set_label("Weights STD")
        else:
            cax_u.axis("off")
    else:
        cax_w.axis("off")
        cax_u.axis("off")

    return fig, axes, pairs

