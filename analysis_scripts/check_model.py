import numpy as np
import h5py
import glob
import os
from astropy.io import ascii
import matplotlib.pyplot as plt
#==============================================================================
if (__name__ == '__main__'):

 case = 'Gaussian'
#  case = 'Double_Gaussian'
#  case = 'Marie1'
#  case = 'Marie2'

 sigma = {'Gaussian':[0.06,0.02], 'Double_Gaussian':[0.06,0.03], 'Marie1':[0.06,0.01], 'Marie2':[0.06,0.02]}

 snr = np.array([10.,25.,50.,100.,200.])
 tab = ascii.read("../losvd/"+case+"_velscale50.dat")

 S0_list = glob.glob("../results_deimos_v2/testcases_"+case+"-S0/*bin*.hdf5")
 S1_list = glob.glob("../results_deimos_v2/testcases_"+case+"-S1/*bin*.hdf5")

#  S0_list = glob.glob("../results/testcases_"+case+"-S0/*bin*.hdf5")
#  S1_list = glob.glob("../results/testcases_"+case+"-S1/*bin*.hdf5")

 for i in range(len(S0_list)):

     f = h5py.File(S0_list[i],"r")
     binID = np.int(os.path.basename(S0_list[i]).split('bin')[1].split('.hdf5')[0])
     check = np.floor_divide(binID,26)

     if  check == 0:
         bin_snr = np.array(snr[0],ndmin=1)
     elif check == 1:
         bin_snr = np.array(snr[1],ndmin=1)
     elif check == 2:
         bin_snr = np.array(snr[2],ndmin=1)
     elif check == 3:
         bin_snr = np.array(snr[3],ndmin=1)
     elif check == 4:
         bin_snr = np.array(snr[4],ndmin=1)
                             
     losvd = np.array(f['losvd']) 
     losvd_lo,losvd_med,losvd_hi = np.percentile(losvd,q=[16,50,84],axis=0)

     err       = np.average(0.5*(losvd_hi-losvd_lo), weights=tab['col2'])
     zscore    = np.array(np.average(np.fabs(losvd_med - tab['col2'])/err, weights=tab['col2']),ndmin=1)
     shrinkage = np.array(1.0 - (err/sigma[case][0])**2,ndmin=1)
 
     print(err)

     plt.scatter(shrinkage, zscore, c=bin_snr, cmap='Blues', vmin=np.amin(snr), vmax=np.amax(snr), alpha=0.75, edgecolors='gray')

 for i in range(len(S1_list)):

     f = h5py.File(S1_list[i],"r")
     binID = np.int(os.path.basename(S1_list[i]).split('bin')[1].split('.hdf5')[0])
     check = np.floor_divide(binID,26)

     if  check == 0:
         bin_snr = np.array(snr[0],ndmin=1)
     elif check == 1:
         bin_snr = np.array(snr[1],ndmin=1)
     elif check == 2:
         bin_snr = np.array(snr[2],ndmin=1)
     elif check == 3:
         bin_snr = np.array(snr[3],ndmin=1)
     elif check == 4:
         bin_snr = np.array(snr[4],ndmin=1)

     losvd = np.array(f['losvd']) 

     losvd_lo,losvd_med,losvd_hi = np.percentile(losvd,q=[16,50,84],axis=0)

     err       = np.average(0.5*(losvd_hi-losvd_lo), weights=tab['col2'])
     zscore    = np.array(np.average(np.fabs(losvd_med - tab['col2'])/err, weights=tab['col2']),ndmin=1)
     shrinkage = np.array(1.0 - (err/sigma[case][1])**2,ndmin=1)

     plt.scatter(shrinkage, zscore, c=bin_snr, cmap='Reds', vmin=np.amin(snr), vmax=np.amax(snr), alpha=0.75, edgecolors='gray')


#  plt.xlim([0,1])
#  plt.ylim([0,5])
#  plt.axhline(y=1.0, color='k', linestyle=':')
 plt.xlabel("Posterior Shrinkage")
 plt.ylabel("Posterior z-score")
 plt.title(case)
#  plt.colorbar()
#  plt.show()
 plt.savefig("Figures/checks_"+case+".png")

