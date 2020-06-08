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

    nfiles = len(filelist)
    print(" - "+str(nfiles)+" files found")

    f      = h5py.File(filelist[0],'r')
    nid    = len(f['in/binID'])
    ntot   = nfiles*nid

    struct = {'id':np.repeat(np.nan,ntot), 'case':list(np.repeat('',ntot)), 
              'S/N':np.repeat(np.nan,ntot),'Fit type':list(np.repeat('',ntot)), 
              'Bias':np.repeat(np.nan,ntot),'Accuracy':np.repeat(np.nan,ntot), 
              'Rel. Error':np.repeat(np.nan,ntot), 'Fit':np.repeat(np.nan,ntot), 
              'RMSE':np.repeat(np.nan,ntot)}

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
        true_losvd = np.interp(xvel, vel, tmp_losvd)
        true_losvd = true_losvd/np.sum(true_losvd)

        for j in range(nid):

            check = f.get('out/'+str(id[j])+'/losvd')
            if check == None:
                k += 1
                continue
 
            # Normalizing the LOSVDs
            losvd       = np.array(f['out/'+str(id[j])+'/losvd'])
            losvd_3s_lo = losvd[0,:]/np.sum(losvd[2,:])
            losvd_3s_hi = losvd[4,:]/np.sum(losvd[2,:])
            losvd_1s_lo = losvd[1,:]/np.sum(losvd[2,:])
            losvd_1s_hi = losvd[3,:]/np.sum(losvd[2,:])
            losvd_med   = losvd[2,:]/np.sum(losvd[2,:])

            # Computing statistics
            diff      = losvd_med - true_losvd
            error     = 0.5*(losvd_1s_hi-losvd_1s_lo)
            rel_error = np.fabs(diff/true_losvd)
            zscore    = np.fabs(diff/error)
            good      = np.isfinite(rel_error)

            bias         = np.average(diff[good],weights=true_losvd[good])/np.amax(true_losvd) 
            av_rel_error = np.average(rel_error[good], weights=true_losvd[good])
            av_accuracy  = np.average(zscore[good],    weights=true_losvd[good])
            fit          = 100.0 * (1.0 - np.sqrt(np.sum(diff[good]**2))/np.sqrt(np.sum((true_losvd[good]-np.mean(losvd_med[good]))**2)/np.sum(good)))
            rmse         = np.sqrt(np.sum(diff[good]**2)/np.sum(good))

            struct['id'][k]         = id[j]
            struct['case'][k]       = case
            struct['S/N'][k]        = snr[id[j]]
            struct['Fit type'][k]   = ftype
            struct['Bias'][k]       = bias 
            struct['Accuracy'][k]   = av_accuracy
            struct['Rel. Error'][k] = av_rel_error
            struct['Fit'][k]        = fit
            struct['RMSE'][k]       = rmse
            k += 1

            # print(ftype, id[j], fit, rmse)
            
            # plt.plot(xvel,rel_error,'k')
            # plt.plot(xvel,true_losvd,'r')
            # plt.plot(xvel,losvd_med,'b')
            # plt.axhline(y=av_rel_error, color='k', linestyle=':')
            # plt.ylim([0,1.0])
            # plt.title(str(id[j])+" "+ftype)
            # plt.show()

        f.close()       

    return Table(struct)
#==============================================================================
if (__name__ == '__main__'):

   dir = "../results_deimos_v4/"
   filelist = glob.glob(dir+"testcases*Marie1-S1/*results.hdf5")
   
   tab = compute_stats(filelist)

   print(tab['Fit type'][np.nonzero(tab['case'])])
