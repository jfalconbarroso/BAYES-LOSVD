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
   * Before execution, the data has to be prepared/preprocessed. This is needed 
   to chose, e.g., the wavelength range for the fitting, the level of spatial binning, 
   number of PCA components or template library, among other things.

Step 2: Running the code
   * This is the main step of the process that leads to the extraction of the LOSVD.

Step 3: Analysis of the outputs
   * In this step the spectral fits, the recovered LOSVD and model convergence 
   diagnostics can be checked.

See :ref:`tutorial` for a full example and a Jupyter notebook.


Preproc data configuration files
---------------------------------------

The standard location to place the required configuration file for the 
preprocessing of a particular dataset is the ``config_files`` directory. 

The configuration file for the preprocessing follows the 
`TOML (Tom's Obvius Minimal Language) <https://en.wikipedia.org/wiki/TOML>`_. 
An example of such file is provided at  the ``config_files/example_preproc.properties file``::

  [NGC0000]
  filename     = "NGC0000.fits"
  instrument   = "MUSE-WFM"
  redshift     = 0.00
  lmin         = 4825.0
  lmax         = 5300.0
  vmax         = 700.0
  velscale     = 60.0
  snr          = 50.0
  snr_min      = 3.0
  template_lib = "MILES_SSP"
  npca         = 5
 <xcen         = 50>
 <ycen         = 50>

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
* ``template_lib``: template library to use from those available in 'templates' directory
* ``npca``: number of PCA components to use as templates
* ``xcen,ycen``: are optional to indicate the central pixel coordinates of the dacube (in pixels)

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

Models configuration file
-----------------------------

This BAYES-LOSVD allows different Numpyro/JAX models to perform the LOSVD fitting. The different implementations 
describe the LOSVD in distinct ways: (1) a pure Simplex definition (with no prior assumptions), (2) a Gaussian Process
with a Wedland kernel. The list of available models is listed in the ``config_files/codes.properties`` file::

  [SP]
  codefile = "bayes_losvd_model_SP.py"
  
  [GP]
  codefile = "bayes_losvd_model_GP.py"
  
Like previous `TOML  <https://en.wikipedia.org/wiki/TOML>`_ files the code identification is set in the ``[<code name>]`` keyword. 

We require the ``codefile`` with the actual name of the file with the Numpyro/JAX model. In addition, it is possible to pass the other variables to the mdel for execution through the 'extra-params' keyword in bayes_losvd_run.py.


Adding new models
""""""""""""""""""""""

Adding a new  code is as simple as including, following the scheme above,  its definition in the ``config_files/codes.properties file`` and adding the required model file to the ``scripts/models/`` directory. For the new model to work properly, it requires that the main function has the same name as the filename of the code::

The user needs to make sure the model accepts a 'data' dictionary. By default the dictionary contains the following keys:: 

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

Note that the input spectrum, error spectrum, mean_template and templates are log-rebinned to the same wavelength and velocity scale.

The parameters of the model can be anything. BAYES-LOSVD will capture them automatically and process them appropiately.

