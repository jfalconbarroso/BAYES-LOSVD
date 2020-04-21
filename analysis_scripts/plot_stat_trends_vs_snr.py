import os
import sys
import glob
import h5py
from astropy.io import ascii
import numpy                 as np
import matplotlib.pyplot     as plt
from   compute_stats         import compute_stats
#==============================================================================
if (__name__ == '__main__'):

    # Loading file list
#    dir = "../results_denso/"
    dir = "../results_deimos/"
    filelist = glob.glob(dir+"testcase_*/testcase_*_results.hdf5")

    # Loading files with results
    print("# Loading results")
    tab = compute_stats(filelist)

    # Defining cases to plot
    cases  = ['Gaussian','Double_Gaussian','Marie1','Marie2']
    snr    = ['10','25','50','100','200']
    # ftype  = ['Border0_free','Border0_penalised','Border1_penalised','Border2_penalised','Border3_penalised','Border4_penalised']
    # labels = ['Simplex','Simplex AR','B-spline 1','B-spline 2','B-spline 3','B-spline 4']
    ftype  = ['Border0_free','Border0_penalised','Border3_penalised','Border4_penalised']
    labels = ['Simplex','Simplex AR','B-spline 3','B-spline 4']
    ncases = len(cases)
    nsnr   = len(snr)
    nftype = len(ftype)

    # Plotting results
    print("# Plotting results")
    colors  = ['red','blue','green','coral','brown','magenta','coralblue']
    fig, ax = plt.subplots(nrows=3, ncols=ncases, sharex=True, sharey='row', figsize=(12,7))
    plt.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.96, wspace=0.0, hspace=0.0)

    ax[0,0].set_xlim([-9.,210])
    for j in range(ncases):
        for k in range(nftype):
            lo, hi, med = np.zeros((3,nftype,nsnr)), np.zeros((3,nftype,nsnr)), np.zeros((3,nftype,nsnr))
            for o in range(nsnr):
                idx = (tab['case'] == cases[j]) & (tab['ftype_name'] == ftype[k]) & (tab['S/N'] == float(snr[o]))
                # idx = (tab['ftype_name'] == ftype[k]) & (tab['S/N'] == float(snr[o]))
                # print(cases[j],ftype[k], snr[o], np.sum(idx))
                if np.sum(idx) == 0:
                    continue
                lo[0,k,o], med[0,k,o], hi[0,k,o] = np.percentile(tab['Bias'][idx],q=[16,50,84])
                lo[1,k,o], med[1,k,o], hi[1,k,o] = np.percentile(tab['Rel. Error'][idx],q=[16,50,84])
                lo[2,k,o], med[2,k,o], hi[2,k,o] = np.percentile(tab['Accuracy'][idx],q=[16,50,84])
            
            ax[0,j].fill_between(np.array(snr,dtype=float), lo[0,k,:], hi[0,k,:], alpha=0.5, color=colors[k])
            ax[0,j].plot(np.array(snr,dtype=float), med[0,k,:], '.-', color=colors[k], label=labels[k])
            ax[1,j].fill_between(np.array(snr,dtype=float), lo[1,k,:], hi[1,k,:], alpha=0.5, color=colors[k])
            ax[1,j].plot(np.array(snr,dtype=float), med[1,k,:], '.-', color=colors[k], label=labels[k])
            ax[2,j].fill_between(np.array(snr,dtype=float), lo[2,k,:], hi[2,k,:], alpha=0.5, color=colors[k])
            ax[2,j].plot(np.array(snr,dtype=float), med[2,k,:], '.-', color=colors[k], label=labels[k])

            ax[0,j].axhline(y=0.0, color='k', linestyle=":")
            ax[1,j].axhline(y=0.0, color='k', linestyle=":")
            ax[2,j].axhline(y=1.0, color='k', linestyle=":")

    [ax[0,i].set_title(cases[i]) for i in range(ncases)]
    ax[0,0].set_ylabel('Bias')        
    ax[1,0].set_ylabel('Relative Error')
    ax[2,0].set_ylabel('Accuracy')
    [ax[-1,i].set_xlabel('S/N') for i in range(ncases)]        
    ax[0,-1].legend(loc='lower right')
    ax[0,0].set_ylim([-0.35,0.075])
    ax[1,0].set_ylim([-0.05,0.75])
    ax[2,0].set_ylim([0.0,3.45])
    # plt.show() 
    plt.savefig("Figures/stat_trends_vs_snr.png", dpi=300)


    exit()
