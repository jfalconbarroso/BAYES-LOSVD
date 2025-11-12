.. _tutorial:

Workflow & Tutorial
===================

As explained in :ref:`usage`, the basic workflow of the code consists of 4 steps:

* Step 1: Pre-processing of the input data
* Step 2: Running the code
* Step 3: Analysis of the outputs
* Step 4: Checking the outputs

**Remember that all the codes have to be run from the** ``scripts`` **directory.**

The sequence of commands to run the code is the following::

  python bayes_losvd_preproc_data.py -c ../config_files/example_preproc.properties
  python bayes_losvd_run.py -f ../preproc_data/NGC0000.hdf5 -l all -t SP
  python bayes_losvd_inspect_fits.py -f ../results/NGC0000_SP/NGC0000_SP_results.hdf5 -l 0
  python bayes_losvd_check_results.py -d ../results/NGC0000_SP
  
In order to help the user to understand better the logic of this workflow as well as all the possible switches and options each code has, we have prepared a `Jupyter Notebook <https://jupyter.org/>`_  showing all possibilites. This notebook is located in the ``scripts/`` directory and can be executed as::
  
  jupyter-notebook bayes_losvd_notebook.ipynb

Note that Jupyter tools have to be installed in the system for this to work.