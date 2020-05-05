import os
import sys
import glob
import h5py
from astropy.io import ascii
import numpy                 as np
import matplotlib.pyplot     as plt
#==============================================================================
def compute_losvd(rootname, idx):

    nbins = len(idx)
    
    losvd = []
    for i in range(nbins):

        filename = rootname+str(idx[i])+'.hdf5'
        if not os.path.exists(filename):
            continue
        # print(filename)

        f   = h5py.File(filename,'r')
        tmp = np.array(f['losvd'])
        f.close()
        if len(losvd) == 0:
            losvd = tmp
        else:
            losvd = np.vstack((losvd,tmp))
       
    losvd_summary = np.percentile(losvd, q=[1,16,50,84,99], axis=0)

    return losvd_summary     

#==============================================================================
if (__name__ == '__main__'):

    dir = '../results_deimos_v2/'

    # cases  = ['Gaussian','Double_Gaussian','Marie1','Marie2']
    cases  = ['Marie2']
    snr    = ['10','25','50','100','200']
    ftype  = ['S0','S1','A1','A2','A3','B3','B4']

    ncases = len(cases)
    nsnr   = len(snr)
    nftype = len(ftype)

    for i in range(ncases):

        print('# Plotting: testcase_'+cases[i])     

        # Loading input LOSVD
        tab = ascii.read("../losvd/"+cases[i]+"_velscale50.dat")
        tab['col2'] /= np.trapz(tab['col2'],tab['col1'])
        
        # Defining the figure parameters 
        fig, ax = plt.subplots(nrows=nsnr, ncols=nftype, sharex=True, sharey=True, figsize=(12,8))
        plt.subplots_adjust(left=0.025, bottom=0.06, right=0.99, top=0.97, wspace=0.0, hspace=0.0)
     
        for o in range(nftype):
            bname    = 'testcases_'+cases[i]+'-'+ftype[o]
            fname    = dir+bname+'/'+bname+'_results.hdf5'
            print(bname)
            if not os.path.exists(fname):
                continue
            f        = h5py.File(fname,'r')
            xvel     = np.array(f['in/xvel'])
            bin_snr  = np.array(f['in/bin_snr'])
            nvel     = len(xvel)        
            rootname = dir+bname+'/'+bname+'_chains_bin'

            for k in range(nsnr):

                    idx   = np.flatnonzero((bin_snr == np.float(snr[k])))
                    losvd = compute_losvd(rootname,idx)

                    losvd_3s_lo  = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
                    losvd_3s_hi  = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
                    losvd_1s_lo  = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
                    losvd_1s_hi  = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
                    losvd_med    = losvd[2,:]/np.trapz(losvd[2,:],-xvel)
            
                    ax[k,o].set_ylim([0,1.5*np.amax(tab['col2'])])
                    ax[k,o].fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
                    ax[k,o].fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
                    ax[k,o].plot(xvel,losvd_med,'k-', ds='steps-mid')
                    ax[k,o].plot(tab['col1'], tab['col2'],'r-', ds='steps-mid') 
                    ax[k,o].axvline(x=0.0, color='k', linestyle=":")

                    if k == 0:
                        ax[k,o].set_title(ftype[o])
                    if k == nsnr-1:
                        ax[k,o].set_xlabel("Velocity (km s$^{-1}$)")
                    if o == 0:
                        ax[k,o].set_ylabel('S/N='+snr[k])
                        ax[k,o].set_yticks([])    
            
        plt.savefig("Figures/testcases_LOSVDS-"+cases[i]+".png", dpi=300)           

    exit()  