import os
import sys
import glob
import h5py
import numpy             as np
import matplotlib.pyplot as plt
from   astropy.io        import ascii
from   astropy.table     import Table
#==============================================================================
def compute_stats(filelist, nsim=150):

    xvel_lims = [-650.0,650.0]

    nfiles = len(filelist)
    # nfiles = 50
    print(" - "+str(nfiles)+" files found")
    ntot = nsim * nfiles

    struct = {'case':list(np.repeat('',ntot)), 'S/N':np.repeat(np.nan,ntot),'Fit type':list(np.repeat('',ntot)), 
              'Bias':np.repeat(np.nan,ntot),'Accuracy':np.repeat(np.nan,ntot), 'Rel. Error':np.repeat(np.nan,ntot)}

    o = 0
    for i in range(nfiles):

        # Defining some basics
        dname  = os.path.dirname(filelist[i])
        bname  = os.path.basename(filelist[i])
        case   = bname.split('-')[0].split('_')[1]
        ftype  = bname.split('-')[1].split('_results.hdf5')[0]
        if case == 'Double':
            case = 'Double_Gaussian'

        # Reading the true inpur LOSVD
        tab = ascii.read("../losvd/"+case+"_velscale50.dat")
        vel = tab['col1']
        tmp_losvd = tab['col2'] / np.sum(tab['col2'])   

        # Opening the HDF5 file with general results
        f          = h5py.File(filelist[i],'r')
        xvel       = np.array(f['in/xvel'])
        snr        = np.array(f['in/bin_snr'])
        good       = (xvel >= xvel_lims[0]) & (xvel <= xvel_lims[1])
        true_losvd = np.interp(xvel, vel, tmp_losvd)
        f.close()       
 
        # Handling individual bin results
        flist = glob.glob(dname+'/testcases_'+case+'*chains_bin*.hdf5')
        for item in flist:

            # Loading the infor from HDF5 file
            if not os.path.exists(item):
                continue
            
            f   = h5py.File(item,'r')
            tmp = np.array(f['losvd'])
            f.close()       
            binid = np.int(item.split('bin')[1].split('.hdf5')[0])
            binsnr = snr[binid]

            # # Normalizing the LOSVDs
            losvd = np.percentile(tmp,q=[1,16,50,84,99],axis=0)
            losvd_3s_lo = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
            losvd_3s_hi = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
            losvd_1s_lo = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
            losvd_1s_hi = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
            losvd_med   = losvd[2,:]/np.trapz(losvd[2,:],-xvel)
            true_losvd  = true_losvd/np.trapz(true_losvd,-xvel)
    
            # # Computing statistics
            diff       = losvd_med - true_losvd
            error      = 0.5*(losvd_1s_hi-losvd_1s_lo)
            bias       = np.average(diff[good],weights=true_losvd[good])/np.amax(true_losvd) 
            rel_error  = np.average(np.fabs(diff[good])/true_losvd[good], weights=true_losvd[good])
            accuracy   = np.average(np.fabs(diff[good])/error[good], weights=true_losvd[good])
    
            # print(case+' '+str(binsnr)+' '+ftype," --- ", bias," --- ", rel_error, " --- ", accuracy)
    
            # # # plt.plot(xvel[good],np.fabs(diff[good])/error[good])
            # plt.plot(xvel[good],np.fabs(diff[good])/true_losvd[good])
            # plt.axhline(y=0, linestyle=':')
            # plt.axhline(y=1, linestyle=':')
            # plt.axhline(y=rel_error)
            # plt.axvline(x=xvel_lims[0],color='k', linestyle=':')
            # plt.axvline(x=xvel_lims[1], color='k', linestyle=":")
            # plt.ylim([-1,1])
            # plt.show()
            # continue
    
            struct['Fit type'][o]   = ftype
            struct['case'][o]       = case
            struct['S/N'][o]        = binsnr
            struct['Fit type'][o]   = ftype
            struct['Bias'][o]       = bias 
            struct['Accuracy'][o]   = accuracy
            struct['Rel. Error'][o] = rel_error
            o += 1
    
            # plt.fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
            # plt.fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
            # plt.plot(xvel,losvd_med,'k.-', ds='steps-mid')
            # plt.plot(xvel, true_losvd,'r.-', ds='steps-mid') 
            # plt.axhline(y=0.0,color='k', linestyle='--')
            # plt.axvline(x=0.0, color='k', linestyle="--")
            # plt.axvline(x=xvel_lims[0],color='k', linestyle=':')
            # plt.axvline(x=xvel_lims[1], color='k', linestyle=":")
            # plt.plot(xvel, diff, 'green', ds='steps-mid', linestyle='--')
            # plt.axhline(y=np.mean(error),color='green', linestyle='--')
            # plt.axhline(y=stat_std,color='green', linestyle=':')
            # plt.title(case+' '+str(snr)+' '+str(border)+' '+ftype+' '+ssp)
            # plt.show()
            # exit() 

    idx = np.isfinite(struct['S/N'])
    
    return Table(struct)[idx]
#==============================================================================
if (__name__ == '__main__'):

   dir = "../results_deimos_v2/"
   filelist = glob.glob(dir+"testcases*/*results.hdf5")

   tab = compute_stats(filelist)
