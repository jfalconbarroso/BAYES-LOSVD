import os
import sys
import glob
import h5py
import warnings
import numpy             as np
import matplotlib.pyplot as plt
import scipy.spatial.distance    as dist
from   astropy.io        import ascii
from   astropy.table     import Table
#==============================================================================
def compute_stats(filelist):

    warnings.filterwarnings("ignore")

    xvel_lims = [-650.0,650.0]
    nfiles = len(filelist)
    print(" - "+str(nfiles)+" files found")

    f      = h5py.File(filelist[0],'r')
    nid    = len(f['in/binID'])
    ntot   = nfiles*nid

    struct = {'id':np.repeat(np.nan,ntot), 'case':list(np.repeat('',ntot)), 
              'S/N':np.repeat(np.nan,ntot),'Fit type':list(np.repeat('',ntot)), 
              'Bias':np.repeat(np.nan,ntot),'Accuracy':np.repeat(np.nan,ntot), 
              'Rel. Error':np.repeat(np.nan,ntot), 'Correlation':np.repeat(np.nan,ntot)}

    k = 0
    for i in range(nfiles):

        f      = h5py.File(filelist[i],'r')
        bname  = os.path.basename(filelist[i])
        case   = bname.split('-')[0].split('testcases_')[1]
        ftype  = bname.split('-')[1].split('_results.hdf5')[0]
        snr    = np.array(f['in/snr'])
        id     = np.array(f['in/binID'])
       
        # Loading and preparing True LOSVD
        tab        = ascii.read("../losvd/"+case+"_velscale50.dat")
        vel        = tab['col1']
        tmp_losvd  = tab['col2']  
        xvel       = np.array(f['in/xvel'])
        good       = (xvel >= xvel_lims[0]) & (xvel <= xvel_lims[1])
        true_losvd = np.interp(xvel, vel, tmp_losvd)
        # true_losvd = true_losvd/np.trapz(true_losvd,-xvel)
        true_losvd = true_losvd/np.sum(true_losvd)

        for j in range(nid):

            check = f.get('out/'+str(id[j])+'/losvd')
            if check == None:
                k += 1
                continue
 
            # Normalizing the LOSVDs
            losvd       = np.array(f['out/'+str(id[j])+'/losvd'])
            # losvd_3s_lo = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
            # losvd_3s_hi = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
            # losvd_1s_lo = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
            # losvd_1s_hi = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
            # losvd_med   = losvd[2,:]/np.trapz(losvd[2,:],-xvel)

            losvd_3s_lo = losvd[0,:]/np.sum(losvd[2,:])
            losvd_3s_hi = losvd[4,:]/np.sum(losvd[2,:])
            losvd_1s_lo = losvd[1,:]/np.sum(losvd[2,:])
            losvd_1s_hi = losvd[3,:]/np.sum(losvd[2,:])
            losvd_med   = losvd[2,:]/np.sum(losvd[2,:])

            # Computing statistics
            diff       = losvd_med - true_losvd
            error      = 0.5*(losvd_1s_hi-losvd_1s_lo)
            bias       = np.average(diff[good],weights=true_losvd[good])/np.amax(true_losvd) 
            rel_error  = np.sum(np.fabs(diff/true_losvd)*true_losvd)/np.sum(true_losvd)
            accuracy   = np.sum(np.fabs(diff/error)*true_losvd)/np.sum(true_losvd)
            correl     = dist.canberra(true_losvd,losvd_med, w=true_losvd)

            struct['id'][k]         = id[j]
            struct['case'][k]       = case
            struct['S/N'][k]        = snr[id[j]]
            struct['Fit type'][k]   = ftype
            struct['Bias'][k]       = bias 
            struct['Accuracy'][k]   = accuracy
            struct['Rel. Error'][k] = rel_error
            struct['Correlation'][k] = correl
            k += 1

        f.close()       

    return Table(struct)
#==============================================================================
if (__name__ == '__main__'):

   dir = "../results_deimos_v3/"
   filelist = glob.glob(dir+"testcases*/*results.hdf5")
   
   tab = compute_stats(filelist)

   print(tab['Fit type'][np.nonzero(tab['case'])])
