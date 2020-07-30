.. _dir_structure:

Directory structure
===============================

We provide below the basic directory structure of the package along with the main files::

   BAYES-LOSVD/
      │ 
      ├─ config_files/
      |   ├─ example_preproc.properties
      |   ├─ codes.properties
      |   ├─ instruments.properties     
      |   └─ instruments/     
      |         ├─ CALIFA.py
      |         ├─ CALIFA-V1200.lsf
      |         ├─ CALIFA-V500.lsf
      |         ├─ EMILES_SSP.lsf
      |         ├─ FITS2D.py
      |         ├─ MANGA.py
      |         ├─ MANGA.lsf
      |         ├─ MILES_SSP.lsf
      |         ├─ MILES_Stars.lsf
      |         ├─ MUSE-WFM.lsf
      |         ├─ MUSE-WFM.py
      |         ├─ SAMI.py
      |         ├─ SAMI-BLUE.lsf
      |         ├─ SAMI-RED.lsf
      |         ├─ SAURON_E3D.lsf
      |         └─ SAURON_E3D.py
      |
      ├─ data/
      │   └─ NGC0000.fits
      |
      ├─ preproc_data/
      |
      ├─ results/
      |
      ├─ scripts/
      │   ├─ bayes_losvd_compile_codes.py
      │   ├─ bayes_losvd_ghfit.py
      │   ├─ bayes_losvd_inspect_fits.py
      │   ├─ bayes_losvd_inspect_ghfit.py
      │   ├─ bayes_losvd_load_hdf5.py
      │   ├─ bayes_losvd_monitor.py
      │   ├─ bayes_losvd_notebook.ipynb
      │   ├─ bayes_losvd_preproc_data.py
      │   ├─ bayes_losvd_run.py
      │   ├─ lib/
      │   └─ stan_model/
      |         ├─ bayes-losvd_ghfit.stan
      |         ├─ bayes-losvd_model_AR.stan
      |         ├─ bayes-losvd_model_Bsplines.stan
      |         ├─ bayes-losvd_model_GH_full_series.stan
      |         ├─ bayes-losvd_model_RW.stan
      |         └─ bayes-losvd_model_SP.stan
      |
      └─ templates/
          ├─ MILES_Stars/
          ├─ MILES_SSP/
          └─ EMILES_SSP/


The main purposes of each directory are:

* ``config_files``: holds all the necessary configuration files to run the code 
* ``data``: contains the data to be analysed (e.g. MUSE datacube)
* ``preproc_data``: it will contain the preprocessed data (in HDF5) for analysis
* ``results``: it will store the results from our main code (i.e. bayes_losvd_run.py)
* ``scripts``: contains the main procedures to run the code
* ``templates``: holds the templates to be used to fit the input spectra

We provide details about the configuration files and execution of the code in :ref:`usage`.


