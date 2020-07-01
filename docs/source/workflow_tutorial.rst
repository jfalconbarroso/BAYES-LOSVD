Workflow & Tutorial
===================

As explained in :ref:`usage`, the basic workflow of the code consists of 4 steps:

* Step 1: Compilation of the Stan codes. 
* Step 2: Pre-processing of the input data
* Step 3: Running the code
* Step 4: Analysis of the outputs

**Remember that all the codes have to be run from the** ``scripts`` **directory.**

The sequence of commands to run the code is the following::

  python bayes_losvd_compile_codes.py
  python bayes_losvd_preproc_data.py -c ../config_files/examples_preproc.properties
  python bayes_losvd_run.py -f ../preproc_data/NGC000.hdf5 -b all -v 1 -t SP
  python bayes_losvd_inspect_fits.py -r NGC000-SP -b 0
  
In order to help the user to understand better the logic of this workflow as well as all the possible switches and options each code has, we have prepared a `Jupyter Notebook <https://jupyter.org/>`_  showing all possibilites. This notebook is located in the ``scripts/`` directory and can be executed as::
  
  jupyter-notebook bayes_losvd_notebook.ipynb

Note that Jupyter tools have to be installed in the system for this to work.