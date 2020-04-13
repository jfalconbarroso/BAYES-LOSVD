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
   border = ['0','1','2','3','4']
   binID  = ['0','169','189']
   rootname = 'NGC4550_SAURON' 

   # Plotting figure
   fig, ax = plt.subplots(nrows=len(border), ncols=3, sharex=True, sharey=True, figsize=(8,8))
   plt.subplots_adjust(left=0.03, bottom=0.06, right=0.99, top=0.96, wspace=0.0, hspace=0.0)

   for i in range(len(border)):
       
       rname     = rootname+"-free_Border"+border[i]+"_11"
       filename1 = outdir+rname+"/"+rname+"_results.hdf5"
       rname     = rootname+"-penalised_Border"+border[i]+"_11"
       filename2 = outdir+rname+"/"+rname+"_results.hdf5"

       f1 = h5py.File(filename1,'r')
       f2 = h5py.File(filename2,'r')
       for j in range(3):
           xvel1  = np.array(f1['in/xvel'])
           losvd1 = np.array(f1['out/'+binID[j]+'/losvd'])
           xvel2  = np.array(f2['in/xvel'])
           losvd2 = np.array(f2['out/'+binID[j]+'/losvd'])
           xvel1 += 60.0
           xvel2 += 60.0
       
           ax[i,j].fill_between(xvel1,losvd1[0,:],losvd1[4,:], color='blue', alpha=0.15, step='mid')
           ax[i,j].fill_between(xvel1,losvd1[1,:],losvd1[3,:], color='blue', alpha=0.50, step='mid')
           ax[i,j].plot(xvel1,losvd1[2,:],'b.-', ds='steps-mid', label='Free')
           ax[i,j].fill_between(xvel2,losvd2[0,:],losvd2[4,:], color='red', alpha=0.15, step='mid')
           ax[i,j].fill_between(xvel2,losvd2[1,:],losvd2[3,:], color='red', alpha=0.50, step='mid')
           ax[i,j].plot(xvel2,losvd2[2,:],'r.-', ds='steps-mid', label='RW prior')
           ax[i,j].axhline(y=0.0,color='k', linestyle='--')
           ax[i,j].axvline(x=0.0, color='k', linestyle="--")
           ax[i,j].set_yticks([])

       f1.close()
       f2.close()


   [ax[0,k].set_title("BinID: "+binID[k]) for k in range(len(binID))]
   ax[0,0].set_ylabel("Simplex")
   [ax[k+1,0].set_ylabel("B-spline order "+border[k+1]) for k in np.arange(len(border)-1)]
   [ax[len(border)-1,k].set_xlabel("Velocity (km s$^{-1}$)") for k in range(3)]
   ax[0,0].set_xlim([-650,650])
   ax[0,-1].legend(fontsize=8)

#    plt.show()   
   fig.savefig("NGC4550_comparison_11.pdf",dpi=300)