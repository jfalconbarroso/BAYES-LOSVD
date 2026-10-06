.. _usage: 

Usage
=====================

.. warning::
   The BAYES-LOSVD package commands have to be executed from the 'scripts' directory.
   
   See :ref:`dir_structure` for relevant details. 


Basic steps
-----------------

Running the code involves the following steps:

Step 1: Pre-processing of the input data
   * Before execution, the data has to be prepared/preprocessed. This is needed to chose, e.g., the wavelength range for the fitting, the level of spatial binning, number of PCA components or template library, among other things.

Step 2: Running the code
   * This is the main step of the process that leads to the extraction of the LOSVD.

Step 3: Analysis of the outputs
   * In this step the spectral fits, the recovered LOSVD and model convergence diagnostics can be checked.

Step 4: Checking the outputs
   * In this step the output .txt files with the results of the inference are analysed. Bins exhibiting non-convergenced are flagged.

See :ref:`tutorial` for a full example and a Jupyter notebook.


Preproc data configuration files
---------------------------------------

The standard location to place the required configuration file for the 
preprocessing of a particular dataset is the ``config_files`` directory. 

The configuration file for the preprocessing follows the 
`TOML (Tom's Obvious Minimal Language) <https://en.wikipedia.org/wiki/TOML>`_. 
An example of such file is provided at  the ``config_files/example_preproc.properties file``::

  [NGC0000]
  filename     = "NGC0000.fits"
  instrument   = "MUSE-WFM"
  redshift     = 0.008764
  lmin         = 4750.0
  lmax         = 5500.0
  vmax         = 700.0
  velscale     = 60.0
  snr          = 0.0
  snr_min      = 1.0
  template_lib = "MILES_SSP"
  npca         = 5
  xcen         = 15  # Optional centre, as in the supplied example
  ycen         = 15

* ``[<run name>]``: name to identify the run
* ``filename`` filename in data dir
* ``instrument``: instrument mode [see instruments.properties]
* ``redshift``: redshift of the target
* ``lmin``: minimum wavelength to be used in the fit (in Angstroms)
* ``lmax``: maximum wavelength to be used in the fit (in Angstroms)
* ``vmax``: maximum value of velocity allowed for the LOSVD extraction
* ``velscale``: desired velocity scale km/s/pix
* ``snr``: target signal-to-noise ratio (Note: if not required set to 0 or a negative value)
* ``snr_min``: S/N used to estimate a signal isophote for spatial selection; not a strict per-spaxel S/N cut
* ``template_lib``: template library to use from those available in 'templates' directory
* ``npca``: number of PCA components to use as templates
* ``<xcen,ycen>``: are optional to indicate the central pixel coordinates of the datacube (in pixels)

The same file can have as many ``[<run name>]`` configuration blocks as needed.

Instruments configuration file
------------------------------

Our current distribution includes reading routines for some of the most popular 
IFUs/surveys (e.g. CALIFA, MANGA, MUSE-WFM, SAMI, SAURON, FITS2D, ...). This is 
defined in a `TOML  <https://en.wikipedia.org/wiki/TOML>`_ file ``ìnstruments.properties`` 
placed in the ``config_files``::

  [CALIFA-V1200]
  read_file = 'CALIFA.py'
  lsf_file  = 'CALIFA-V1200.lsf'
  
  [CALIFA-V500]
  read_file = 'CALIFA.py'
  lsf_file  = 'CALIFA-V500.lsf'

  [MANGA]
  read_file = 'MANGA.py'
  lsf_file  = 'MANGA.lsf'

  [MUSE-WFM]
  read_file = 'MUSE-WFM.py'
  lsf_file  = 'MUSE-WFM.lsf'
  
  [MUSE-WFM_2D]
  read_file = 'FITS2D.py'
  lsf_file  = 'MUSE-WFM.lsf'
  
  [SAMI-BLUE]
  read_file = 'SAMI.py'
  lsf_file  = 'SAMI-BLUE.lsf'

  [SAMI-RED]
  read_file = 'SAMI.py'
  lsf_file  = 'SAMI-RED.lsf'

  [SAURON_E3D]
  read_file = 'SAURON_E3D.py'
  lsf_file  = 'SAURON_E3D.lsf'
  
Each instrument is defined with a ``[<instrument name>]`` heading.
This is the name to be used in the ``instrument`` keyword of the preprocessing configuration 
file. For each instrument, two files are required: a Python routine to read the instrument 
data (``read_file``), and an ASCII file describing the Line-Spread Function (i.e. the 
instrumental resolution as a function of wavelength) for the instrument (``lsf_file``). Both 
files are placed in the ``config_files/instruments`` directory for the default instruments. 

Templates configuration file
------------------------------

Our current distribution includes reading routines for some of the MILES template libraries. 
This is defined in a `TOML  <https://en.wikipedia.org/wiki/TOML>`_ file ``templates.properties`` 
placed in the ``config_files``::

  [MILES_SSP]
  read_file = 'MILES_SSP.py'
  lsf_file  = 'MILES_SSP.lsf'
  
  [MILES_Stars]
  read_file = 'MILES_Stars.py'
  lsf_file  = 'MILES_Stars.lsf'

.. hint::
   Adding new instruments or template libraries is as simple as including, following the scheme above, their definition 
   in the ``config_files/instruments.properties`` and ``config_files/templates.properties`` file and adding the required 
   two new files to the ``config_files/instruments/`` and ``config_files/templates/`` directory respectively. The user 
   should use existing files for reference on the required input and output variables. Please make sure there are no 
   NaNs in the data by setting up the flux values to zero and the errors to a very large value.

Models configuration file
-----------------------------

This BAYES-LOSVD allows different Numpyro/JAX models to perform the LOSVD fitting. The different implementations 
describe the LOSVD in distinct ways: (1) a pure Simplex definition (with a Dirichlet prior), (2) a Gaussian Process
with a Wendland kernel. The list of available models is listed in the ``config_files/codes.properties`` file::

  [SP]
  codefile = "bayes_losvd_model_SP.py"
  
  [GP]
  codefile = "bayes_losvd_model_GP.py"
  
Like previous `TOML  <https://en.wikipedia.org/wiki/TOML>`_ files the code identification is set in the ``[<code name>]`` keyword. 

We require the ``codefile`` with the actual name of the file with the Numpyro/JAX model. In addition, it is possible to pass the other variables to the model through the ``--extra_pars`` option (GP fixed-value overrides; see the SP limitation in :doc:`current_implementation`) in bayes_losvd_run.py.


Adding new models
""""""""""""""""""""""

Adding a new code is as simple as including, following the scheme above, its definition in the ``config_files/codes.properties file`` and adding the required model file to the ``scripts/models/`` directory. For the new model to work properly, it requires that the main function has the same name as the filename of the code.

The user needs to make sure the model accepts a 'data' dictionary. By default the dictionary must contain the following keys:: 
   def <model name>(data):
   
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
       params        = data['params']

Note that the input spectrum, error spectrum, mean_template and templates are log-rebinned to the same wavelength and velocity scale.

Additional posterior variables are saved automatically, but the runner requires a posterior site named ``losvd`` with length ``len(xvel)`` for its Gauss-Hermite fits. See :doc:`current_implementation` for the reader and model contracts.

Output files format
-------------------

Output results from bayes_losvd_run.py
""""""""""""""""""""""""""""""""""""""

The output file with the results is an HDF5 file with the following structure and variables::

   Group ['in']: # Preprocessing data plus metadata added at packing
      - binID
      - bin_flux
      - bin_snr
      - flux
      - lmax
      - lmin
      - lwave_temp
      - mask
      - mean_template
      - nbins
      - ndim
      - npca
      - npix
      - npix_obs
      - npix_temp
      - nspec
      - ntemp
      - nvel
      - params
      - porder
      - psize
      - redshift
      - sigma_obs
      - snr
      - spec_obs
      - templates
      - velscale
      - wave
      - wave_obs
      - x
      - xbin
      - xvel
      - y
      - ybin

   Group ['out']: # for the GP model case
      - coeffs
      - continuum
      - ell_gp
      - eta
      - h3_star
      - h4_star
      - losvd
      - model_spec
      - sigma_gp
      - sigma_star
      - snr_real
      - vel_star
      - weights
  if npca = 0 (GP) the group will contain:
      - mean_params

This information can be loaded into a dictionary using the bayes_losvd_load_hdf5.py script::

   from bayes_losvd_load_hdf5 import load_hdf5
   tab = load_hdf5("../results/NGC0000/NGC0000_results.hdf5")

Output chains from bayes_losvd_run.py
"""""""""""""""""""""""""""""""""""""

If --save_chains is activated in bayes_losvd_run.py then NETCDF files will be stored in the 'results' directory for each spectrum. These files can be opened and manipulated with ArviZ with Arviz::

   import arviz as az
   idata = az.from_netcdf("../results/NGC0000/NGC0000_chains_bin0.netcdf")

Shapes, units, credible intervals, masks and the exact CLI options are described
in :doc:`current_implementation`. Registry entries do not guarantee that a
working reader or the template data is available; consult that page before
selecting an optional library.
