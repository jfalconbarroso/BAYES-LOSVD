Download & Installation
=======================

On this page you can download the latest version of the software, along with a 
description of the system requirements and python packages needed.

System requirements
"""""""""""""""""""""""
The supplied environments select Python 3.12. CPU and GPU availability depends on
the JAX installation and platform. Check the
`official JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for supported combinations; the GPU YAML requests ``jax[cuda13]``. Do not assume
that this CUDA environment enables an Apple GPU.

Download
"""""""""""""""""""""""

We recommend to install the BAYES-LOSVD package in a separate and new conda 
environment, using the supplied Python 3.12 specification. For further instructions on the use and management 
of conda environments, please see the `Conda Documentation <https://conda.io>`_.

The BAYES-LOSVD code is installed by cloning the following `Github <https://github.com>`_ repository 
from the command line::

   git clone https://github.com/jfalconbarroso/BAYES-LOSVD.git



Python dependencies
"""""""""""""""""""""""

The code requires the packages in the supplied environment files::

   environment.yaml       # CPU environment
   environment-gpu.yaml   # CUDA environment, on supported systems 

This file can be used to install all those packages within your environment::

   conda env create -f environment.yaml
   conda activate bayes_losvd

Run commands from ``scripts/``. Check the backend before inference::

   python -c "import jax; print(jax.devices())"

The environment files are unchanged in this documentation revision and do not
pin the whole dependency stack. The Gauss-Hermite routine uses
``numpy.trapezoid``, so install NumPy 2 or later if your environment resolves an
older version. Optional notebook and Explorer dependencies can be installed with::

   conda install -c conda-forge "numpy>=2" jupyterlab "plotly>=6" "ipywidgets>=8" anywidget voila

To rebuild the manual, install Sphinx and the book theme, then run from the
repository root::

   python -m pip install "sphinx>=7" sphinx-book-theme
   python -m sphinx -b html docs/source docs/build/html

These are installation instructions, not a claim that every platform or optional
viewer has been tested.

Parallelisation
"""""""""""""""""""""""
The parallelisation of the pipeline uses the multiprocessing module of Python's Standard Library. In particular, it uses
a spawned ``multiprocessing.Pool`` to run bins in parallel (``-j``). NumPyro
chains within a bin run sequentially. Start with ``-j 1``; more processes may
increase compilation overhead and memory use, particularly on a GPU. 

The drawback of the Python multiprocessing module is that it does not natively support the use of multiple nodes on
large computing clusters. However, at this point the use of one node (with e.g. 32 cores) should be sufficient for most
kinds of analysis. Implementing a distributed memory parallelisation in BAYES-LOSVD is nonetheless a long-term
objective. 

The implemented parallelisation has been tested on various machines: This includes hardware from laptop up to
cluster systems, as well as Linux and MacOS operating systems. 

|
