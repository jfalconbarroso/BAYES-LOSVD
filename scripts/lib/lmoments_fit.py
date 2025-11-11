import xarray as xr
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial.hermite import hermval
#==============================================================================
def gauss_hermite(v, V, sigma, h3, h4):
    x = (v - V) / sigma
    L = np.exp(-0.5 * x**2) / (sigma * np.sqrt(2 * np.pi))
    # Physicists' Hermite polynomials
    H3 = hermval(x, [0,0,0,1])
    H4 = hermval(x, [0,0,0,0,1])

    return L * (1 + h3 * H3 / np.sqrt(6) + h4 * H4 / np.sqrt(24))

#------------------------------------------------------------------------------
def fit_gauss_hermite(v, losvd, p0=None):
    losvd = losvd / np.trapz(losvd, v)
    if p0 is None:
        V0 = np.average(v, weights=losvd)
        sigma0 = np.sqrt(np.average((v - V0)**2, weights=losvd))
        p0 = [V0, sigma0, 0.0, 0.0]

    lower = [v.min(), 1e-3, -0.3, -0.3]
    upper = [v.max(), np.ptp(v), 0.3, 0.3]

    popt, _ = curve_fit(
        gauss_hermite, v, losvd, p0=p0,
        bounds=(lower, upper), maxfev=5000
    )

    return popt  # [V, sigma, h3, h4]

#==============================================================================
def lmoments_fit(xvel, losvd_samples, idata, n_chains):

    """
    xvel: velocity grid
    losvd_samples: array of shape (n_total_samples, n_vel)
    idata: ArviZ InferenceData
    n_chains: number of MCMC chains
    """

    ncases = losvd_samples.shape[0]
    gh_params = np.zeros((ncases, 4))  # (sample, param)

    print(f"# Fitting {ncases} LOSVD samples with Gauss-Hermite expansion...")

    for i in range(ncases):
        try:
            gh_params[i, :] = fit_gauss_hermite(xvel, losvd_samples[i, :])
        except RuntimeError:
            # If the fit fails, fill with NaN
            gh_params[i, :] = np.nan
            print(f"Warning: fit failed for sample {i}")

    # Split back into (chains, draws)
    draws_per_chain = ncases // n_chains
    gh_params = gh_params.reshape(n_chains, draws_per_chain, 4)

    # Create xarray DataArrays
    coords = {"chain": np.arange(n_chains),
              "draw": np.arange(draws_per_chain)}

    idata.posterior["vel_star"]   = xr.DataArray(gh_params[:, :, 0], dims=("chain", "draw"), coords=coords)
    idata.posterior["sigma_star"] = xr.DataArray(gh_params[:, :, 1], dims=("chain", "draw"), coords=coords)
    idata.posterior["h3_star"]    = xr.DataArray(gh_params[:, :, 2], dims=("chain", "draw"), coords=coords)
    idata.posterior["h4_star"]    = xr.DataArray(gh_params[:, :, 3], dims=("chain", "draw"), coords=coords)

    return idata
