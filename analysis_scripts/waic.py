import numpy as np
import h5py
import glob
import os
from astropy.io import ascii
import matplotlib.pyplot as plt
import arviz as az
#==============================================================================
if (__name__ == '__main__'):

   case = 'Gaussian'
   s0 = az.from_netcdf("../results/testcases_"+case+"-S0/testcases_"+case+"-S0_chains_bin50.netcdf")
   s1 = az.from_netcdf("../results/testcases_"+case+"-S1/testcases_"+case+"-S1_chains_bin50.netcdf")
   a1 = az.from_netcdf("../results/testcases_"+case+"-A1/testcases_"+case+"-A1_chains_bin50.netcdf")
   a2 = az.from_netcdf("../results/testcases_"+case+"-A2/testcases_"+case+"-A2_chains_bin50.netcdf")
   a3 = az.from_netcdf("../results/testcases_"+case+"-A3/testcases_"+case+"-A3_chains_bin50.netcdf")
   b3 = az.from_netcdf("../results/testcases_"+case+"-B3/testcases_"+case+"-B3_chains_bin50.netcdf")
   b4 = az.from_netcdf("../results/testcases_"+case+"-B4/testcases_"+case+"-B4_chains_bin50.netcdf")

   compare_dict = {"S0": s0, "S1": s1, "A1": a1, "A2": a2, "A3": a3, "B3": b3, "B4": b4}
   comp = az.compare(compare_dict,ic='loo', scale='log')
   print(comp)

   az.plot_compare(comp)
   plt.title(case)
   plt.show()

