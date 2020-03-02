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
   files  = ["testcase_0-SNR200_LOSVD1_results.hdf5",
            "testcase_5-SNR200_LOSVD2_results.hdf5",
            "testcase_1-SNR100_LOSVD1_results.hdf5",
            "testcase_6-SNR100_LOSVD2_results.hdf5",
            "testcase_2-SNR50_LOSVD1_results.hdf5",
            "testcase_7-SNR50_LOSVD2_results.hdf5",
            "testcase_3-SNR25_LOSVD1_results.hdf5",
            "testcase_8-SNR25_LOSVD2_results.hdf5",
            "testcase_4-SNR10_LOSVD1_results.hdf5",
            "testcase_9-SNR10_LOSVD2_results.hdf5"]

   nfiles = len(files)

   # Loading input LOSVDs
   tab1 = ascii.read("../losvd/losvd_1_velscale25.dat")
   tab2 = ascii.read("../losvd/losvd_2_velscale50.dat")
  

   # Making plot
   fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(6,6))
   ax = ax.ravel()
   plt.subplots_adjust(left=0.01, bottom=0.08, right=0.99, top=0.99, wspace=0.0, hspace=0.0)

 
   for i in range(nfiles):
      
       rname = str.split(files[i],'_results.hdf5')[0]
       l     = str.split(rname,'SNR')[1]
       snr   = str.split(l,'_')[0]
       f     = h5py.File(outdir+rname+"/"+files[i],'r')
       xvel  = np.array(f['in/xvel'])
       losvd = np.array(f['out/0/losvd'])
       f.close()

       losvd_3s_lo  = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
       losvd_3s_hi  = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
       losvd_1s_lo  = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
       losvd_1s_hi  = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
       losvd_med    = losvd[2,:]/np.trapz(losvd[2,:],-xvel)

       if np.mod(i,2) == 0:
          col1 = tab1['col1'] 
          col2 = tab1['col2'] 
          ax[i].text(700,0.003,'SNR: '+snr,horizontalalignment='right', fontweight='bold',fontsize=10)
       else:
          col1 = tab2['col1'] 
          col2 = tab2['col2']
       col2 /= np.trapz(col2,col1)


       ax[i].axhline(y=0.0,color='k', linestyle=':')
       ax[i].axvline(x=0.0, color='k', linestyle=":")
       ax[i].fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
       ax[i].fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
       ax[i].plot(xvel,losvd_med,'k.-', ds='steps-mid')
       ax[i].plot(col1, col2,'r.-', ds='steps-mid') 

       ax[i].set_yticks([])
       ax[i].set_ylim([-0.0002,0.004])
       ax[i].set_xlim([-749,749])

   ax[-2].set_xlabel("Velocity (km s$^{-1}$)")      
   ax[-1].set_xlabel("Velocity (km s$^{-1}$)")      


   # plt.show()   
   fig.savefig("bayes-losvd_fig3.pdf",dpi=300)