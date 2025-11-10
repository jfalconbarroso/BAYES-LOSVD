import numpyro
import jax.numpy             as jnp
import numpyro.distributions as dist
from jax.scipy.signal        import convolve
from jax.numpy               import sqrt, exp, pi, std
#=============================================================================
def legendre_eval(x, coeffs):

    """
    Evaluate sum_{n=0}^{N-1} coeffs[n] * P_n(x)
    """
    P = coeffs.shape[0]
    L = [jnp.ones_like(x), x]
    for n in range(2, P):
        Ln = ((2 * n - 1) * x * L[-1] - (n - 1) * L[-2]) / n
        L.append(Ln)
    L_stack = jnp.stack(L, axis=0)

    return jnp.tensordot(coeffs, L_stack[:P], axes=1)

#------------------------------------------------------------------------------
def create_kernel(xvel, v_shift, sigma_val, h3_val, h4_val):

    # Defining some variables
    w = (xvel - v_shift) / sigma_val
    w2 = w ** 2

    # Gauss-Hermite polynomials
    H3 = w * (2.0 * w2 - 3.0)
    H4 = 4.0 * w2**2 - 12.0 * w2 + 3.0

    # Normalized Gaussian
    gaussian = exp(-0.5 * w2) / sqrt(2 * pi)

    # Full kernel with Hermite expansion
    poly = 1.0 \
         + h3_val * H3 / sqrt(3.0) \
         + h4_val * H4 / sqrt(24.0)

    kernel = gaussian * poly

    # Normalize over velocity dimension for each spectrum
    kernel /= jnp.sum(kernel, axis=0, keepdims=True)

    return kernel  # shape: [Nvel, Nspec]

#=============================================================================
def bayes_losvd_model_SP(data):

    # Loading all the necessary data
    mean_template = data['mean_template']
    templates     = data['templates']
    spec_obs      = data['spec_obs']
    sigma_obs     = data['sigma_obs']
    porder        = data['porder']
    mask          = data['mask']
    xvel          = data['xvel']
    snr_input     = data['snr']
    NPCA          = data['npca']
    params        = data['params'] 
    Npix, Ntemp   = templates.shape
    xcont         = jnp.linspace(-1, 1, Npix)
    vscale        = xvel[1]-xvel[0] 
    Nvel          = len(xvel)

    # Adjusting input SNR and error spectrum (using SNR as better estimate)
    # Note: I don't believe anything with an error below 1% (i.e. SNR=100)
    snr_input = jnp.clip(snr_input, max=100.0)
    sigma_obs = jnp.ones_like(sigma_obs) * (1.0/snr_input)

    # --- Defining the priors for the model---
    if "alpha" in params:
        alpha = numpyro.deterministic("alpha", params['alpha'])
    else:
        alpha = numpyro.deterministic("alpha", 0.1 + numpyro.sample("xalpha_offset", dist.Gamma(2.0, 5.0)))
    
    weights = numpyro.sample("weights", dist.Dirichlet(alpha * jnp.ones(Ntemp)))

    # --- Create non-parametric LOSVD ---
    if "beta" in params:
        beta = numpyro.deterministic("beta", params['beta'])
    else:
        beta = numpyro.deterministic("beta", 0.1 + numpyro.sample("xbeta_offset", dist.Gamma(2.0, 5.0)))
                                      
    losvd = numpyro.sample("losvd", dist.Dirichlet(beta * jnp.ones(Nvel)))

    # --- Combine templates ---
    if NPCA == 0:    
        combined_temp = jnp.dot(templates, weights.T) # [Npix]
    else:
        combined_temp = mean_template + jnp.dot(templates,weights)

    # --- Convolve templates ---
    convolved_temp = convolve(combined_temp, losvd, mode='same') # [Npix]

    # --- Continuum for each spectrum
    if porder == 0:
        coeff_full = jnp.array([1.0])
    else:
        coeffs = numpyro.sample("coeffs", dist.Normal(0.0, 0.15).expand([porder]))
        coeff_full = jnp.concatenate([jnp.array([1.0]), coeffs])
    continuum  = numpyro.deterministic("continuum", legendre_eval(xcont, coeff_full))  # [Npix]

    # --- Model spectra ---
    model_spec = numpyro.deterministic("model_spec", convolved_temp * continuum.T) 

    # --- Real SNR ---
    numpyro.deterministic("snr_real", 1.0 / jnp.std(spec_obs[mask] - model_spec[mask]))

    # --- Likelihood ---
    numpyro.sample("obs", dist.Normal(model_spec[mask], sigma_obs[mask]), obs=spec_obs[mask])
