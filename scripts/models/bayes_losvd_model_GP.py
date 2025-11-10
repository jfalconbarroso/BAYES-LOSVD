import numpyro
import jax.numpy             as jnp
import numpyro.distributions as dist
from   jax.scipy.signal      import convolve
from   jax.scipy.linalg      import cholesky
from   jax.nn                import softplus
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

#=============================================================================
def bayes_losvd_model_GP(data):

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

    # Adjusting input SNR and error spectrum (using SNR as better estimate)
    # Note: I don't believe anything with an error below 1% (i.e. SNR=100)
    snr_input = jnp.clip(snr_input, max=100.0)
    sigma_obs = jnp.ones_like(sigma_obs) * (1.0/snr_input)

    # --- Defining the priors for the template weights ---
    if NPCA == 0:
        if "tau" in params:    
            tau = numpyro.deterministic("tau", params['tau'])
        else:
            tau = numpyro.sample("tau", dist.TruncatedNormal(1.5, 0.5, low=0.5, high=2.5))
        z = numpyro.sample("z", dist.Normal(0, tau).expand([Ntemp]))
        z = z - jnp.mean(z)
        weights = numpyro.deterministic("weights", jnp.exp(z) / jnp.sum(jnp.exp(z)))
    else:
        weights = numpyro.sample("weights", dist.Normal(0.0,1.0).expand([Ntemp]))

    # --- Create non-parametric LOSVD ---

    # .... Hyperpriors ....
    if "ell_gp" in params:
        ell_gp   = numpyro.deterministic("ell_gp", params['ell_gp']) 
    else:     
        ell_gp   = numpyro.sample("ell_gp", dist.TruncatedNormal(3.5, 0.5, low=1.0, high=6.0))

    if "sigma_gp" in params:     
        sigma_gp = numpyro.deterministic("sigma_gp", params['sigma_gp']) 
    else:
        sigma_gp = numpyro.sample("sigma_gp", dist.LogNormal(-2.0, 0.7)) 
    
    # .... GP kernel .... 
    x  = (xvel / vscale)[:, None] 
    dx = jnp.abs(x - x.T) 
    r  = dx / ell_gp 
    K  = sigma_gp**2 * jnp.where(r < 1.0, (1 - r)**4 * (4*r + 1), 0.0) 
    K += 1e-6 * jnp.eye(len(xvel)) 
    L  = cholesky(K, lower=True) 
    
    # .... GP draw (softplus version) .... 
    mean_func = jnp.ones_like(xvel) / len(xvel) 
    eta       = numpyro.sample("eta", dist.Normal(0, 1).expand([len(xvel)])) 
    f         = mean_func + L @ eta 
    losvd     = softplus(f) 
    losvd    /= jnp.sum(losvd)

    numpyro.deterministic("losvd", losvd)

    # --- Combine templates ---
    if NPCA == 0:    
        combined_temp = jnp.dot(templates, weights.T) # [Npix]
    else:
        combined_temp = mean_template + jnp.dot(templates,weights)

    # --- Convolve templates ---
    convolved_temp = convolve(combined_temp, losvd, mode='same') # [Npix]

    # --- Continuum for each spectrum
    # Note: coeff[0]=1.0, coeff[1:]~N(0.0,0.15)
    if porder == 0:
        coeff_full = jnp.array([1.0])
    else:
        coeffs = numpyro.sample("coeffs", dist.Normal(0.0, 0.15).expand([porder]))
        coeff_full = jnp.concatenate([jnp.array([1.0]), coeffs])
    continuum  = numpyro.deterministic("continuum", legendre_eval(xcont, coeff_full))  # [Npix]

    # --- Model spectrum ---
    model_spec = numpyro.deterministic("model_spec", convolved_temp * continuum.T) # [Npix]

    # --- Real SNR ---
    numpyro.deterministic("snr_real", 1.0 / jnp.std(spec_obs[mask] - model_spec[mask]))

    # --- Likelihood ---
    numpyro.sample("obs", dist.Normal(model_spec[mask], sigma_obs[mask]), obs=spec_obs[mask])

