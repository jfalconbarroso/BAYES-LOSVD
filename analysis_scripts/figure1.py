import os
import sys
import glob
import h5py
from astropy.io import ascii
import numpy                 as np
import matplotlib.pyplot     as plt
#==============================================================================
if (__name__ == '__main__'):

   # Loading files with results
   outdir = "../results_denso/"
   files  = ["testcase_Gaussian_Mixed_SNR100-No_prior_results.hdf5",
             "testcase_Gaussian_Mixed_SNR100-Border1_results.hdf5",
             "testcase_Gaussian_Mixed_SNR100-Border2_results.hdf5",
             "testcase_Gaussian_Mixed_SNR100-Border3_results.hdf5",
             "testcase_Gaussian_Mixed_SNR100-Border4_results.hdf5"]

   nfiles = len(files)
    
   # Loading input LOSVD
   tab = ascii.read("../losvd/losvd_Gaussian_velscale50.dat")
   tab['col2'] /= np.trapz(tab['col2'],tab['col1'])
  
   # Plotting figure
   fig, ax = plt.subplots(nrows=nfiles, ncols=1, sharex=True, sharey=True, figsize=(4,8))
   ax = ax.ravel()
   plt.subplots_adjust(left=0.01, bottom=0.06, right=0.99, top=0.99, wspace=0.0, hspace=0.0)

  
   for i in range(nfiles):

       rname = str.split(files[i],'_results.hdf5')[0]
       f     = h5py.File(outdir+rname+"/"+files[i],'r')
       xvel  = np.array(f['in/xvel'])
       losvd = np.array(f['out/0/losvd'])
       f.close()

       losvd_3s_lo  = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
       losvd_3s_hi  = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
       losvd_1s_lo  = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
       losvd_1s_hi  = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
       losvd_med    = losvd[2,:]/np.trapz(losvd[2,:],-xvel)

       if i == 0:
           ax[i].text(700,0.004,'No priors',horizontalalignment='right', fontweight='bold',fontsize=11)
       else:
           ax[i].text(700,0.004,'B-splines order '+str(i),horizontalalignment='right', fontweight='bold',fontsize=11)

       ax[i].fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
       ax[i].fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
       ax[i].plot(xvel,losvd_med,'k.-', ds='steps-mid')
       ax[i].plot(tab['col1'], tab['col2'],'r.-', ds='steps-mid') 
       ax[i].axhline(y=0.0,color='k', linestyle='--')
       ax[i].axvline(x=0.0, color='k', linestyle="--")

       ax[i].plot(xvel, -0.001 + losvd_med - tab['col2'],color='darkgray', ds='steps-mid')
       ax[i].axhline(y=-0.001,color='k', linestyle=':')


       ax[i].set_yticks([])
       ax[i].set_ylim([-0.0015,0.005])

   ax[-1].set_xlabel("Velocity (km s$^{-1}$)")

#    plt.show()   
   fig.savefig("bayes-losvd_fig1.pdf",dpi=300)