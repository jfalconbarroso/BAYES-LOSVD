.. _usage: 

Usage
=====================

.. warning::
   The BAYES-LOSVD package commands have to be executed from the 'scripts' directory.
   
   See :ref:`dir_structure` for relevant details. 


Basic steps
-----------------

Running the code involves the following steps:

Step 1: Compilation of the Stan codes. 
   * This step is necessary to run the main fitting code. Stan models will be 
   internally converted into C++ and then compiled. This step only needs to be executed once.

Step 2: Pre-processing of the input data
   * Before execution, the data has to be prepared/preprocessed. This is needed 
   to chose, e.g., the wavelength range for the fitting, the level of spatial binning, 
   number of PCA components or template library, among other things.

Step 3: Running the code
   * This is the main step of the process that leads to the extraction of the LOSVD.

Step 4: Analysis of the outputs
   * In this step the spectral fits, the recovered LOSVD and Stan convergence 
   diagnostics can be checked.


Preproc data configuration files
---------------------------------------

The standard location to place the required configuration file for the 
preprocessing of a particular dataset is the ``config_files`` directory. 

The configuration file for the preprocessing follows the 
`TOML (Tom's Obvius Minimal Language) <https://en.wikipedia.org/wiki/TOML>`_. 
An example of such file is provided at  the ``config_files/example_preproc.properties file``::

  [NGC000]
  filename     = "NGC000.fits"
  instrument   = "MUSE-WFM"
  redshift     = 0.00
  lmin         = 4825.0
  lmax         = 5300.0
  vmax         = 700.0
  velscale     = 60.0
  snr          = 50.0
  snr_min      = 3.0
  porder       = 5
  template_lib = "MILES_SSP"
  npca         = 5
  mask_file    = "emission_lines.mask"

* ``[<run name>]``: name to identify the run
* ``filename`` filename in data dir
* ``instrument``: intrument mode [see instruments.properties]
* ``redshift``: redshift of the target
* ``lmin``: minimum wavelength to be used in the fit (in Angstroms)
* ``lmax``: maximum wavelength to be used in the fit (in Angstroms)
* ``vmax``: maximum value of velocity allowed for the LOSVD extraction
* ``velscale``: desired velocity scale km/s/pix
* ``snr``: target signal-to-noise ratio (Note: if not required set to 0 or a * negative value)
* ``snr_min``: minimum signal-to-noise to use for the spatial binning
* ``porder``: polynomial order to be used in spectral fitting
* ``template_lib``: template library to use from those available in 'templates' directory
* ``npca``: number of PCA components to use as templates
* ``mask_file``: emission line mask file. Set to "None" if no masking is desired

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

.. hint::
   Adding new instruments is as simple as including, following the scheme above,  their definition 
   in the ``config_files/instruments.properties`` file and adding the required two new files to the 
   ``config_files/instruments/`` directory. The user should use existing files for reference on the 
   required input and output variables. Please make sure there are no NaNs in the data by setting up
   the flux values to zero and the errors to a very large value. See SAMI.py for an example.

Stan codes configuration file
-----------------------------

BAYES-LOSVD allows different Stan models to perform the LOSVD fitting. The different implementations 
describe the LOSVD in distinct ways: from a pure Simplex definition (with no prior assumptions), to 
several forms of regularization using priors (e.g. Random Walk, Auto-Regresive, or penalised B-splines). 
The list of available models is listed in the ``config_files/codes.properties`` file::

  [SP]
  codefile = "bayes-losvd_model_SP.stan"
  
  [RW]
  codefile = "bayes-losvd_model_RW.stan"
  
  [AR1]
  codefile = "bayes-losvd_model_AR.stan"
  order    = 1
  
  [AR2]
  codefile = "bayes-losvd_model_AR.stan"
  order    = 2
  
  [AR3]
  codefile = "bayes-losvd_model_AR.stan"
  order    = 3
  
  [Bsplines3]
  codefile = "bayes-losvd_model_Bsplines.stan"
  spline_order = 3
  
  [Bsplines4]
  codefile = "bayes-losvd_model_Bsplines.stan"
  spline_order = 4
  
  [GHfree]
  codefile = "bayes-losvd_model_GH_full_series.stan"

Like previous `TOML  <https://en.wikipedia.org/wiki/TOML>`_ files the code identification is set in the ``[<code name>]`` keyword. We require the ``codefile`` with the actual name of the file with the Stan model. In addition, it is possible to pass the Stan code other variables for execution (see, e.g.,  AR and Bsplines models above).


Adding new Stan models
""""""""""""""""""""""

Adding a new Stan code is as simple as including, following the scheme above,  its definition in the ``config_files/codes.properties file`` and adding the required Stan model file to the ``scripts/stan_model/`` directory. For the new model to work properly, it requires the following input variables in the Stan's data block::

  data {
     int<lower=1> npix_obs;      // Number of pixels of input spectrum
     int<lower=1> ntemp;         // Number of PC components
     int<lower=1> npix_temp;     // Number of pixels of each PC components
     int<lower=1> nvel;          // Number of pixels of the LOSVD
     int<lower=1> nmask;         // Number of pixels of the mask
     int<lower=1> mask[nmask];   // Mask with pixels to be fitted
     int<lower=0> porder;        // Polynomial order to be used
     vector[npix_obs]            spec_obs;      // Array with observed spectrum 
     vector<lower=0.0>[npix_obs] sigma_obs;     // Array with error spectrum
     matrix[npix_temp,ntemp]     templates;     // Array with PC components spectra
     vector[npix_temp]           mean_template; // Array with mean template of the  PCA decomposion
     vector[nmask]               spec_masked;   // masked input spectrum
  }  

These variables will be generated automatically during the preprocessing process. Note that the input spectrum, error spectrum, mean_template and templates are log-rebinned to the same wavelength and velocity scale.

In addition, the generated quantites block should contain the following variables::

  generated quantities {
  
    vector[npix_temp] spec      = mean_template + templates * weights;
    vector[npix_obs]  conv_spec = convolve_data(spec,losvd,npix_temp,nvel);
    vector[npix_obs]  poly      = leg_pols * coefs;
    vector[npix_obs]  bestfit   = poly + conv_spec;
    
  }

The parameters of the model can be anything. BAYES-LOSVD will capture them automatically and process them appropiately.

