Current implementation and limitations
======================================

This reference describes the original implementation, retained unchanged in the
documentation-only revision of 6 October 2026. It supplements the existing
manual rather than introducing a new pipeline. Suggestions below are not
implemented fixes.

Execution and command-line reference
------------------------------------

Run from ``scripts/``. Preprocessing uses ``-c`` / ``--config_file``. Inference
accepts the following arguments (use ``-h`` for each script's own help):

.. list-table:: Inference arguments
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Meaning
   * - ``-f --preproc_file``
     - None
     - Supply an existing preprocessing HDF5 file.
   * - ``-l --bin_option``
     - all
     - all, odd, even, or comma-separated zero-based bin IDs; no range notation.
   * - ``-m --maskfile``
     - None
     - ASCII wavelength mask.
   * - ``-p --porder``
     - 5
     - Number of additional multiplicative Legendre coefficients.
   * - ``-n --nsamples``
     - 500
     - Warm-up steps AND retained samples per chain.
   * - ``-c --nchain``
     - 2
     - Sequential chains per bin; retained draws total nchain * nsamples.
   * - ``-j --njobs``
     - 1
     - Number of bin workers in a spawned Pool.
   * - ``-o --outdir``
     - ../results/
     - Per-bin destination; packing still uses ../results/. Keep the default.
   * - ``-t --fit_type``
     - GP
     - Model key from config_files/codes.properties.
   * - ``-s --save_chains``
     - off
     - Save per-bin NetCDF posterior chains.
   * - ``--extra_pars``
     - None
     - Comma-separated key=value overrides; GP example below.

The current ``odd`` option selects bins 0, 2, 4, ... and ``even`` selects
1, 3, 5, ... . These names are reversed relative to literal zero-based parity.
Use explicit IDs when this distinction matters. IDs must be within the saved
``nbins`` range. Each bin uses PRNGKey(0); no seed option exists. NUTS uses
target_accept_prob=0.90 and max_tree_depth=10, with no CLI overrides.

For GP, this fixes two values rather than changing their prior distributions::

   python bayes_losvd_run.py -f ../preproc_data/NEW_RUN.hdf5 -l 0 -t GP --extra_pars ell_gp=3.0,sigma_gp=0.5

Use an unused NEW_RUN filename and keep the default output root. For SP, the
model currently reads ``alpha`` and ``beta`` from ``data['params']`` (template
metadata), whereas the runner places CLI overrides in ``data['pars']``.
Do not rely on SP CLI overrides until that small dictionary mismatch is fixed.

Output names and reruns
-----------------------

The preprocessed filename stem is the run name. For ``NGC0000.hdf5``, both
GP and SP use ``results/NGC0000/NGC0000_results.hdf5``. ``-t`` changes the
model and the packed ``in/suffix`` metadata; it does not change filenames.
Use distinct preprocessing section/run names for distinct models or settings.

.. warning::
   Packing replaces an existing packed file using only the per-bin files
   currently present, then removes those per-bin files. It does not merge
   previous packed bins. Refitting one bin after a successful full run can
   leave all other bins as NaN in the replacement file. Retain the previous
   output directory and use a new run name for a new fit.

The ``-o`` argument changes per-bin writing only, requires a trailing path
separator, and is not forwarded to packing. A custom output root can therefore
leave new files unpacked or cause packing to collect files from the default
root. The examples use the default ``../results/`` throughout.

Worker exceptions print tracebacks, but ``run_fit`` discards the returned
per-bin status and the main script can still print FINISHED. Check actual
outputs and finite fitted-bin values. Old summary or chain files can remain
after reruns; their presence alone is not evidence of a newly successful fit.
Interrupted packing is not atomic. Preserve valuable results before rerunning.

Scientific assumptions and priors
---------------------------------

Both models use independent Gaussian pixels with constant noise
``1 / min(bin_snr, 100)`` on the retained pixels. Input errors affect
preprocessing and S/N estimation but are replaced in the inference likelihood.
Correlations introduced by resampling are not modeled. Large input-error
sentinels therefore do not independently downweight fitting pixels; use the
spectral mask when a region should be excluded. Posterior uncertainty is
conditional on these assumptions, the templates and the selected priors.

* GP uses a Wendland covariance on ``xvel / vscale``. ``ell_gp`` has a
  truncated Normal(3.5, 0.5) prior on [1, 6] in velocity-grid pixels.
  ``sigma_gp`` is LogNormal(-2.0, 0.7), before positivity and normalization.
  A latent GP draw is passed through softplus and normalized to sum to one.
* With PCA, GP template coefficients have independent Normal(0, 1) priors.
  Without PCA, normalized exponential weights are formed from Normal(0, tau)
  latent coefficients, with tau ~ truncated Normal(1.5, 0.5) on [0.5, 2.5].
  GP allows fixed ``tau`` (without PCA), ``ell_gp`` and ``sigma_gp`` via
  ``--extra_pars``.
* SP uses Dirichlet template weights and LOSVD masses. Concentrations alpha
  and beta default to ``0.1 + Gamma(2, 5)`` (concentration/rate convention).
  With PCA, SP still constrains coefficient weights to a simplex; these are
  not the same priors as GP's unconstrained PCA coefficients.
* The multiplicative continuum has constant term 1 and ``porder`` additional
  coefficients with Normal(0, 0.15) priors. At porder=0 it is unity.

The LOSVD is discrete probability mass with ``sum(losvd)=1``; to express a
density per km/s, divide by velocity-bin spacing. ``vel_star``, ``sigma_star``,
``h3_star`` and ``h4_star`` come from Gauss-Hermite fits to each posterior LOSVD
draw, rather than raw moments of a potentially multimodal distribution.
The implemented h3/h4 fitting bounds are +/-0.3; failed fits produce NaN.

Preprocessing and input contracts
---------------------------------

* Wavelength limits are in rest-frame Angstroms, after dividing by
  ``1+redshift``. ``velscale`` is km/s per pixel. ``vmax`` limits the velocity
  grid; non-divisible ranges can stop inside the requested endpoints. Use the
  saved ``xvel`` values as authoritative.
* ``snr <= 0`` disables spatial binning. Positive target S/N uses PowerBin.
  ``snr_min`` estimates a signal isophote, rather than applying a strict
  per-spaxel S/N cut.
* ``npca=0`` retains original templates; positive values use PCA. Choose a
  component count no larger than the smaller template-array dimension and
  inspect explained variance. PCA coefficients are not physical template
  population fractions.
* For pixel-coordinate cube readers, supplied centres use the loader's
  ``center-1`` convention (one-based pixel centres). Reader coordinates and
  pixel size determine the resulting spatial units; confirm them for your
  instrument rather than assuming all readers have identical conventions.
* Resolution files contain named columns ``Lambda`` and ``FWHM`` in Angstroms.
  Check coverage. If templates have worse resolution than the data, a warning
  does not make exact resolution matching possible.
* Mask rows contain centre, full width (Angstroms on the rest-frame grid), and
  comment. Automatic redshift correction for sky-line rows is commented out;
  convert observer-frame sky positions yourself. Approximately 2% at each
  spectral end is excluded automatically. The saved mask contains retained
  pixel indices, rather than Boolean values.

New data readers expose ``read_data(filename)`` returning a dictionary with
``wave, spec, espec, x, y, npix, nspax, psize, ndim``. Flux and standard-deviation
errors have shape (wavelength_pixels, spectra); convert variance/inverse
variance before returning. Wavelengths must increase and match the spectra.
Consult the selected reader's FITS header/extension handling: FITS2D uses
primary flux, an error HDU and ``CRVAL1 + CDELT1 * index`` rather than full WCS.
FITS2D_novar synthesizes errors as ``sqrt(spec)/100``; negative flux is
problematic. Sanitize non-finite values, and mask invalid fitting pixels.

Template readers expose ``read_templates(template_lib)`` returning
``wave, temp, ntemp, npix, params``, with temp shaped (npix, ntemp) and parameter
metadata shaped (nparameters, ntemp). Follow the existing reader's filename
patterns and verify the library contents and resolution before selecting it.

The registry includes incomplete entries: EMILES_SSP and SINFONI template
reader files are absent; XSL_Stars exposes ``read_data`` rather than the expected
``read_templates``; the testdata instrument's referenced MILES_SSP.lsf is absent
from instruments/ (it exists under templates/). These entries are unchanged
and should not be treated as working examples. MILES SSP/stellar readers and
sMILES readers require their corresponding data directories. Registry presence
does not establish availability, provenance or validation of a library. Record
the source, release, selection and citation of templates you use; their complete
acquisition history is not recorded in the original repository.

Results and inspection
----------------------

Packed scalar summaries have shape (nbins, 7), ``losvd`` has (nbins, 7, nvel),
and ``model_spec`` / ``continuum`` have (nbins, 7, npix_obs). Per-bin files omit
the leading bin axis. Summary positions are 0.1%, 15.9%, 50%, 84.1%, 99.9%,
mean and standard deviation. These give 68.2% and 99.8% equal-tailed intervals,
not the ArviZ text summary's HDIs. Bins without saved fits are NaN.

``wave_obs`` and ``lwave_temp`` store natural logarithms of rest-frame
wavelength: exponentiate them for Angstroms. Most ``in`` fields originate in
preprocessing; mask, porder and suffix are added at packing. For GP without
PCA, ``mean_params`` reflects template-parameter row order; the MILES SSP reader
uses age followed by metallicity. It is not a fitted independent parameter.

Example (run from scripts/)::

   import numpy as np
   from bayes_losvd_load_hdf5 import load_hdf5
   res = load_hdf5('../results/NGC0000/NGC0000_results.hdf5')
   good = np.isfinite(res['vel_star'][:, 2])
   median = res['vel_star'][good, 2]
   lower_error = median - res['vel_star'][good, 1]
   upper_error = res['vel_star'][good, 3] - median

Saved NetCDF chains preserve correlations unavailable from marginal HDF5
summaries. The original loader function is usable as above, but its standalone
CLI contains undefined variables after the function call; use the Python
function or h5py rather than that CLI. The function also leaves its HDF5 handle
without an explicit close; use an h5py context manager for repeated direct reads.

The Explorer is a newer alternative to the existing inspection script. Its
setup and batch commands are in ``scripts/losvd_explorer/README.md``. Directory
discovery expects a base ``*_results.hdf5``; unpacked loading expects that base
input plus adjacent per-bin files. Standalone interrupted-run files are not
guaranteed to be discovered or loaded. Keep these distinct from packed products.

Diagnostics and reproducibility
-------------------------------

The checker uses ``-d --dirname``, ``-r --rhat_max`` (default 1.1) and
``-e --ess_min`` (default 100). It flags summary rows above the R-hat threshold
or below either bulk/tail ESS threshold. Consider a stricter threshold such
as ``-r 1.01``, assess sampling stability and inspect posterior fits.

It does not verify every expected bin, flag NaN diagnostics through its
comparisons, or check divergences. Its text parsing assumes familiar summary
column layouts. An empty/malformed row or stale summary can evade meaningful
checking. Exit status and a good report do not certify completeness or
convergence. The runner prints a separate warning for >1% divergences; examine
saved chain sample_stats and problematic fits directly, even below that limit.

Record the original commit, full preprocessing configuration, template library,
mask, model, fixed values, command, seed convention, package versions and JAX
device. These are not comprehensively captured by the original HDF5 schema.
For a scientific analysis, vary velocity range/scale, wavelength coverage,
PCA count and continuum order, check fit residuals and divergences, and increase
chain length until the conclusions are stable. No universal setting guarantees
recovery or convergence.

Small changes suggested for later
---------------------------------

The following remain proposals so that each can be understood and reviewed
individually before changing the code:

* Correct SP's override dictionary from params to pars.
* Forward the chosen output root to packing and normalize its trailing separator.
* Propagate failed worker statuses to the final report and exit status.
* Close the loader handle explicitly and repair its standalone CLI.
* Repair or disable the incomplete registry targets after selecting the intended
  readers. Sort template file lists for reproducible ordering.
* Decide whether to change odd/even semantics; this affects existing scripts.

Preserving prior packed bins and making packing atomic need more careful
changes than documentation. Keep those as a separate future task, with the
existing functions and naming conventions retained where possible.
