import os
import sys
import glob
import h5py
from astropy.io import ascii
import numpy                 as np
import matplotlib.pyplot     as plt
from   compute_stats         import compute_stats
#==============================================================================
def plot_ftype_vs_SNR_split(tab):
   
    # Defining cases to plot
    cases  = ['Gaussian','Double_Gaussian','Marie1','Marie2','Wings']
    snr    = [10,25,50,100,200]
    ftype  = ['S0','S1','A1','A2','A3','A4']#,'B3','B4']
    labels = ['Free','RW','AR(1)','AR(2)','AR(3)','AR(4)','B-spline 3','B-spline 4']
    ncases = len(cases)
    nsnr   = len(snr)
    nftype = len(ftype)

    # Plotting results
    print("# Plotting results")
    # colors  = ['red','blue','green','coral','brown','magenta','navy']
    colors = plt.cm.jet(np.linspace(0,1,nftype))

    fig, ax = plt.subplots(nrows=2, ncols=ncases, sharex=True, sharey='row', figsize=(12,7))
    plt.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.96, wspace=0.0, hspace=0.0)

    ax[0,0].set_xlim([-9.,210])
    for j in range(ncases):
        for k in range(nftype):
            lo, hi, med = np.zeros((4,nftype,nsnr)), np.zeros((4,nftype,nsnr)), np.zeros((4,nftype,nsnr))
            for o in range(nsnr):
                idx = (tab['case'] == cases[j]) & (tab['Fit type'] == ftype[k]) & (tab['S/N'] == snr[o])
                
                # print(cases[j],ftype[k],snr[o],np.sum(idx))
                if np.sum(idx) == 0:
                    continue
                lo[0,k,o], med[0,k,o], hi[0,k,o] = np.percentile(tab['Bias'][idx],q=[16,50,84])
                lo[1,k,o], med[1,k,o], hi[1,k,o] = np.percentile(tab['Rel. Error'][idx],q=[16,50,84])
                lo[2,k,o], med[2,k,o], hi[2,k,o] = np.percentile(tab['Accuracy'][idx],q=[16,50,84])
                # lo[3,k,o], med[3,k,o], hi[3,k,o] = np.percentile(tab['Correlation'][idx],q=[16,50,84])
            
            # ax[0,j].fill_between(np.array(snr,dtype=float), lo[0,k,:], hi[0,k,:], alpha=0.25, color=colors[k])
            # ax[0,j].plot(np.array(snr,dtype=float), med[0,k,:], '.-', color=colors[k], label=labels[k])
            ax[0,j].fill_between(np.array(snr,dtype=float), lo[1,k,:], hi[1,k,:], alpha=0.25, color=colors[k])
            ax[0,j].plot(np.array(snr,dtype=float), med[1,k,:], '.-', color=colors[k], label=labels[k])
            ax[1,j].fill_between(np.array(snr,dtype=float), lo[2,k,:], hi[2,k,:], alpha=0.25, color=colors[k])
            ax[1,j].plot(np.array(snr,dtype=float), med[2,k,:], '.-', color=colors[k], label=labels[k])
            # ax[3,j].fill_between(np.array(snr,dtype=float), lo[3,k,:], hi[3,k,:], alpha=0.5, color=colors[k])
            # ax[3,j].plot(np.array(snr,dtype=float), med[3,k,:], '.-', color=colors[k], label=labels[k])

            # ax[0,j].axhline(y=0.0, color='k', linestyle=":")
            ax[0,j].axhline(y=0.0, color='k', linestyle=":")
            ax[1,j].axhline(y=1.0, color='k', linestyle=":")

    [ax[0,i].set_title(cases[i]) for i in range(ncases)]
    # ax[0,0].set_ylabel('Bias')        
    ax[0,0].set_ylabel('Relative Error')
    ax[1,0].set_ylabel('Accuracy')
    # ax[3,0].set_ylabel('Correlation')
    [ax[-1,i].set_xlabel('S/N') for i in range(ncases)]        
    ax[0,-1].legend(loc='upper right', fontsize=9)
    # ax[0,0].set_ylim([-0.35,0.075])
    ax[0,0].set_ylim([-0.05,0.75])
    ax[1,0].set_ylim([0.0,3.45])
    # plt.show() 
    plt.savefig("Figures/stat_trends_vs_snr_split.png", dpi=300)

    return
#==============================================================================
def plot_ftype_vs_SNR_all(tab):
   
    # Defining cases to plot
    cases  = ['Gaussian','Double_Gaussian','Marie1','Marie2','Wings']
    snr    = [10,25,50,100,200]
    ftype  = ['S0','S1','A1','A2','A3','A4']#,'B3','B4']
    labels = ['Free','RW','AR(1)','AR(2)','AR(3)','AR(4)','B-spline 3','B-spline 4']
    nsnr   = len(snr)
    nftype = len(ftype)

    # Plotting results
    print("# Plotting results")
    # colors  = ['red','blue','green','coral','brown','magenta','cyan']
    colors = plt.cm.jet(np.linspace(0,1,nftype))
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey='row', figsize=(5,7))
    plt.subplots_adjust(left=0.15, bottom=0.06, right=0.99, top=0.99, wspace=0.0, hspace=0.0)

    ax[0].set_xlim([-9.,210])
    for k in range(nftype):
        lo, hi, med = np.zeros((4,nftype,nsnr)), np.zeros((4,nftype,nsnr)), np.zeros((4,nftype,nsnr))
        for o in range(nsnr):
         
            idx = (tab['Fit type'] == ftype[k]) & (tab['S/N'] == snr[o])
            
            if np.sum(idx) == 0:
                continue
            lo[0,k,o], med[0,k,o], hi[0,k,o] = np.nanpercentile(tab['Bias'][idx],q=[16,50,84])
            lo[1,k,o], med[1,k,o], hi[1,k,o] = np.nanpercentile(tab['Rel. Error'][idx],q=[16,50,84])
            lo[2,k,o], med[2,k,o], hi[2,k,o] = np.nanpercentile(tab['Accuracy'][idx],q=[16,50,84])
            # lo[3,k,o], med[3,k,o], hi[3,k,o] = np.percentile(tab['Correlation'][idx],q=[16,50,84])
        
        # ax[0].fill_between(np.array(snr,dtype=float), lo[0,k,:], hi[0,k,:], alpha=0.25, color=colors[k])
        # ax[0].plot(np.array(snr,dtype=float), med[0,k,:], '.-', color=colors[k], label=labels[k])
        ax[0].fill_between(np.array(snr,dtype=float), lo[1,k,:], hi[1,k,:], alpha=0.25, color=colors[k])
        ax[0].plot(np.array(snr,dtype=float), med[1,k,:], '.-', color=colors[k], label=labels[k])
        ax[1].fill_between(np.array(snr,dtype=float), lo[2,k,:], hi[2,k,:], alpha=0.25, color=colors[k])
        ax[1].plot(np.array(snr,dtype=float), med[2,k,:], '.-', color=colors[k], label=labels[k])
        # ax[3].fill_between(np.array(snr,dtype=float), lo[3,k,:], hi[3,k,:], alpha=0.5, color=colors[k])
        # ax[3].plot(np.array(snr,dtype=float), med[3,k,:], '.-', color=colors[k], label=labels[k])

    # ax[0].axhline(y=0.0, color='k', linestyle=":")
    # ax[0].axhline(y=0.0, color='k', linestyle=":")
    ax[1].axhline(y=1.0, color='k', linestyle=":")

    # ax[0].set_ylabel('Bias')        
    ax[0].set_ylabel('Relative Error')
    ax[1].set_ylabel('Accuracy')
    # ax[3].set_ylabel('Correlation')
    ax[-1].set_xlabel('S/N')       
    ax[0].legend(loc='upper right', fontsize=9)
    # ax[0].set_ylim([-0.35,0.075])
    ax[0].set_ylim([0.0,1.00])
    ax[1].set_ylim([0.0,4.49])
    # plt.show() 
    plt.savefig("Figures/stat_trends_vs_snr_all.png", dpi=300)


    return
#==============================================================================
if (__name__ == '__main__'):


    # Loading file list
    dir = "../results_deimos_v4/"
    filelist = glob.glob(dir+"testcases_*/testcases_*_results.hdf5")

    # Loading files with results
    print("# Loading results")
    tab = compute_stats(filelist)
    
    # Plot Ftype vs SNR for different cases
    plot_ftype_vs_SNR_split(tab)

    # Plot Ftype vs SNR for all cases at the same time
    plot_ftype_vs_SNR_all(tab)