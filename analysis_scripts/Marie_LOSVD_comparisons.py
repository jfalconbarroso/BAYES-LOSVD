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
   snr = ['200','100','50','25','10']
   suffix = ''
   suffix = '_penalised'
   vscale = '50'
   vscale = '25'


   # Loading input LOSVDs
   tab1 = ascii.read("../losvd/losvd_1_velscale"+vscale+".dat")
   tab2 = ascii.read("../losvd/losvd_2_velscale"+vscale+".dat")
  
   # Plotting figure
   fig, ax = plt.subplots(nrows=len(snr), ncols=2, sharex=True, sharey=True, figsize=(6,7))
   plt.subplots_adjust(left=0.02, bottom=0.07, right=0.99, top=0.99, wspace=0.0, hspace=0.0)

   for i in range(len(snr)):
       
       rname1 = "testcase_1?-SNR"+snr[i]+"_LOSVD1"+suffix
       list1 = glob.glob(outdir+rname1+"/"+rname1+"_results.hdf5")
       rname2 = "testcase_1?-SNR"+snr[i]+"_LOSVD2"+suffix
       list2 = glob.glob(outdir+rname2+"/"+rname2+"_results.hdf5")

       f1 = h5py.File(list1[0],'r')
       f2 = h5py.File(list2[0],'r')
       xvel1  = np.array(f1['in/xvel'])
       losvd1 = np.array(f1['out/0/losvd'])
       xvel2  = np.array(f2['in/xvel'])
       losvd2 = np.array(f2['out/0/losvd'])

       ax[i,0].set_ylim([-0.01,0.25])
       ax[i,0].text(700,0.2,'SNR: '+snr[i],horizontalalignment='right', fontweight='bold',fontsize=10)

       ax[i,0].fill_between(xvel1,losvd1[0,:],losvd1[4,:], color='blue', alpha=0.15, step='mid')
       ax[i,0].fill_between(xvel1,losvd1[1,:],losvd1[3,:], color='blue', alpha=0.50, step='mid')
       ax[i,0].plot(xvel1,losvd1[2,:],'b.-', ds='steps-mid', label='Free')
       ax[i,0].plot(tab1['col1'],tab1['col2'],'r.-', ds='steps-mid')

       ax[i,1].fill_between(xvel2,losvd2[0,:],losvd2[4,:], color='blue', alpha=0.15, step='mid')
       ax[i,1].fill_between(xvel2,losvd2[1,:],losvd2[3,:], color='blue', alpha=0.50, step='mid')
       ax[i,1].plot(xvel2,losvd2[2,:],'b.-', ds='steps-mid', label='Free')
       ax[i,1].plot(tab2['col1'],tab2['col2'],'r.-', ds='steps-mid')

       ax[i,0].axhline(y=0.0,color='k', linestyle='--')
       ax[i,0].axvline(x=0.0, color='k', linestyle="--")
       ax[i,0].set_yticks([])
       ax[i,1].axhline(y=0.0,color='k', linestyle='--')
       ax[i,1].axvline(x=0.0, color='k', linestyle="--")
       ax[i,1].set_yticks([])
     
       f1.close()
       f2.close()

   [ax[len(snr)-1,k].set_xlabel("Velocity (km s$^{-1}$)") for k in range(2)]

#    plt.show()   
   fig.savefig("Marie_LOSVD_comparison"+suffix+".pdf",dpi=300)