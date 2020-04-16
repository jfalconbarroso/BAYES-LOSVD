import os
import sys
import glob
import h5py
from astropy.io import ascii
import numpy                 as np
import matplotlib.pyplot     as plt
#==============================================================================
if (__name__ == '__main__'):

    dir = '../results_deimos/'

    cases  = ['Gaussian','Double_Gaussian','Marie1','Marie2']
    snr    = ['10','25','50','100','200']
    ssp    = ['Mixed','Old','Young']
    # ftype  = ['Border0_free','Border0_penalised','Border1_penalised','Border2_penalised','Border3_penalised','Border4_penalised']
    # labels = ['Simplex','Simplex AR','B-spline 1','B-spline 2','B-spline 3','B-spline 4']
    ftype  = ['Border0_free','Border0_penalised','Border3_penalised','Border4_penalised']
    labels = ['Simplex','Simplex AR','B-spline 3','B-spline 4']
    ncases = len(cases)
    nsnr   = len(snr)
    nssp   = len(ssp)
    nftype = len(ftype)

    for i in range(ncases):

        # Loading input LOSVD
        tab = ascii.read("../losvd/"+cases[i]+"_velscale50.dat")
        tab['col2'] /= np.trapz(tab['col2'],tab['col1'])
        
        for j in range(nssp):

            idx = 0
            # Defining the figure parameters 
            fig, ax = plt.subplots(nrows=nsnr, ncols=nftype, sharex=True, sharey=True, figsize=(12,8))
            plt.subplots_adjust(left=0.025, bottom=0.06, right=0.99, top=0.97, wspace=0.0, hspace=0.0)
     
            for k in range(nsnr):
                for o in range(nftype):
                    basename = 'testcase_'+cases[i]+'_'+ssp[j]+'_SNR'+snr[k]+'-'+ftype[o]
                    filename = dir+basename+'/'+basename+'_results.hdf5'
                    if os.path.exists(filename):
                        f     = h5py.File(filename,'r')
                        xvel  = np.array(f['in/xvel'])
                        losvd = np.array(f['out/0/losvd'])
                        f.close()

                        losvd_3s_lo  = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
                        losvd_3s_hi  = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
                        losvd_1s_lo  = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
                        losvd_1s_hi  = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
                        losvd_med    = losvd[2,:]/np.trapz(losvd[2,:],-xvel)
                
                        ax[k,o].set_ylim([0,1.5*np.amax(tab['col2'])])
                        ax[k,o].fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
                        ax[k,o].fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
                        ax[k,o].plot(xvel,losvd_med,'k.-', ds='steps-mid')
                        ax[k,o].plot(tab['col1'], tab['col2'],'r.-', ds='steps-mid') 
                        ax[k,o].axvline(x=0.0, color='k', linestyle="--")

                        if k == 0:
                            ax[k,o].set_title(labels[o])
                        if k == nsnr-1:
                            ax[k,o].set_xlabel("Velocity (km s$^{-1}$)")
                        if o == 0:
                            ax[k,o].set_ylabel('S/N='+snr[k])
                            ax[k,o].set_yticks([])    
                
            plt.savefig("Figures/testcases_LOSVDS-"+cases[i]+"_"+ssp[j]+".png", dpi=300)           

    exit()  