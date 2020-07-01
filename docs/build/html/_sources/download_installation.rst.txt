Download & Installation
=======================

On this page you can download the latest version of the pipeline, along with a 
description of the system requirements and python packages needed.

System requirements
"""""""""""""""""""""""
The code should run on all platforms (OSX, Linux, Windows) provided the user can 
install `Python 3 <https://www.python.org>`_ and all the Python dependecies listed below.

Download
"""""""""""""""""""""""
The BAYES-LOSVD code is installed by cloning the following repository 
from the command line::

   git clone https://github.com/jfalconbarroso/BAYES-LOSVD.git

**We strongly recommend to install the BAYES-LOSVD package in a separate and new conda 
environment, using Python3.6!** For further instructions on the use and management 
of conda environments, please see the `Conda Documentation <https://conda.io>`_.

Python dependencies
"""""""""""""""""""""""

The code requires the following Python packages to be installed in the system. 
Most of these packages should be readily avaialble with the standard conda installation, 
but use 'conda install' to install them. Independent installations using 'pip' 
should also work.

+--------------+-----------+
| Package      | Version   |
+==============+===========+
| pystan       | 2.19.00   | 
+--------------+-----------+
| astropy      | 3.1.2     |
+--------------+-----------+
| arviz        | 0.6.1     |
+--------------+-----------+
| numpy        | 1.18.1    |
+--------------+-----------+
| matplotlib   | 3.1.3     |
+--------------+-----------+
| scipy        | 1.4.1     |
+--------------+-----------+
| h5py         | 2.10.0    |
+--------------+-----------+
| scikit_learn | 0.23.1    |
+--------------+-----------+
| tqdm         | 4.46.1    |
+--------------+-----------+
| toml         | 0.10.1    |
+--------------+-----------+

No additional configuration (e.g. environmental paths, etc) is required to run the code.

Directory structure & Execution
"""""""""""""""""""""""""""""""""""""""

The BAYES-LOSVD package commands have to be executed from the 'scripts' directory.
We provide below the basic directory structure of the package along with the main files::

   BAYES-LOSVD/
      │ 
      ├─ config_files/
      |   ├─ example_preproc.properties
      |   ├─ codes.properties
      |   ├─ instruments.properties     
      |   └─ instruments/     
      |         ├─ CALIFA-V1200.lsf
      |         ├─ CALIFA-V1200.py
      |         ├─ CALIFA-V500.lsf
      |         ├─ CALIFA-V500.py
      |         ├─ EMILES_SSP.lsf
      |         ├─ FITS2D.py
      |         ├─ MILES_SSP.lsf
      |         ├─ MILES_Stars.lsf
      |         ├─ MUSE-WFM.lsf
      |         ├─ MUSE-WFM.py
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
      │   ├─ bayes_losvd_load_results.py
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



Parallelisation
"""""""""""""""""""""""
The parallelisation of the pipeline uses the multiprocessing module of Python's Standard Library. In particular, it uses
multiprocessing.Queue and multiprocessing.Process providing a maximum of stability and control over the parallel
processes.  In addition, the threading/parallelisation of other Python native modules, such as numpy or scipy, is
suppressed. Thus, the number of active processes should never exceed the number of cores defined in the configuration
file, nor should any process be able to claim more than 100% CPU usage. 

The drawback of the Python multiprocessing module is that it does not natively support the use of multiple nodes on
large computing clusters. However, at this point the use of one node (with e.g. 32 cores) should be sufficient for most
kinds of analysis. Implementing a distributed memory parallelisation in BAYES-LOSVD is nonetheless a long-term
objective. 

The implemented parallelisation has been tested on various machines: This includes hardware from laptop up to
cluster systems, as well as Linux and MacOS operating systems. 

|
