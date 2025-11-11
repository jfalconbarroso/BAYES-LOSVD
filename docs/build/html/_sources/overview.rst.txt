Overview
========

BAYES-LOSVD is a python framework for the non-parametric extraction of the Line-Of-Sight Velocity Distributions in galaxies. It makes use of Stan (https://mc-stan.org/) to perform all the computations and provide reliable uncertainties for all the parametes of the model chosen for the fit. The code comes with a large number of features, including read-in routines for some of the most popular IFU spectrographs and surveys: ATLAS3D, CALIFA, MaNGA, MUSE-WFM, SAMI, SAURON. 


Example figures
"""""""""""""""""""""""
.. figure:: bayes-losvd_fig_snr.png
   :width: 300
  
   LOSVD recovery for no regularisation with a Simplex (v1 of the code) and a Gaussian Process (this version). Red line on all panels are the input test LOSVD. The recovered median values of the LOSVDs are indicated with a thick black line. 16%-84% and 1%-99% confidence limits at each point are indicated in dark and light blue, respectively. Our new approach with Gaussian Processes overcomes the inherent limitations of defining the LOSVD with a Simplex (i.e. prone to jumpy solutions). 

