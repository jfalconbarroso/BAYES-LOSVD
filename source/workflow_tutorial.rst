.. _tutorial:

Workflow & Tutorial
===================

As explained in :ref:`usage`, the basic workflow of the code consists of 4 steps:

* Step 1: Pre-processing of the input data
* Step 2: Running the code
* Step 3: Analysis of the outputs
* Step 4: Checking the outputs

**Remember that all the codes have to be run from the** ``scripts`` **directory.**

.. warning::
   These commands use the run name ``NGC0000`` and can replace existing outputs.
   Before executing them, copy the example configuration to a new file and
   rename its ``[NGC0000]`` section to an unused name. Use that name in all
   preprocessing and result paths below. Do not partially rerun a valuable
   packed result: previous bins are not merged automatically.

The following one-bin example uses the supplied configuration and default output
root. It illustrates execution; it does not establish scientific convergence::

  python bayes_losvd_preproc_data.py -c ../config_files/example_preproc.properties
  python bayes_losvd_run.py -f ../preproc_data/NGC0000.hdf5 -l 0 -t GP -s
  python bayes_losvd_inspect_fits.py -f ../results/NGC0000/NGC0000_results.hdf5 -l 0
  python bayes_losvd_check_results.py -d ../results/NGC0000
  
In order to help the user to understand better the logic of this workflow as well as all the possible switches and options each code has, we have prepared a `Jupyter Notebook <https://jupyter.org/>`_  showing the main steps. This notebook is located in the ``scripts/`` directory and can be executed as::
  
  jupyter lab bayes_losvd_workflow.ipynb

Note that Jupyter tools have to be installed in the system for this to work.
The output directory is ``results/<preprocessed filename stem>/``; ``-t GP``
or ``-t SP`` does not append a model suffix. Use different preprocessing names
for model comparisons. The older ``bayes_losvd_notebook.ipynb`` remains a short
alternative. See :doc:`current_implementation` before refitting or changing output
paths, and ``scripts/losvd_explorer/README.md`` for the newer viewer.
