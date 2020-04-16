import os
import sys
import glob
import h5py
import numpy             as np
import matplotlib.pyplot as plt
import seaborn as sns
from   astropy.io        import ascii
from   astropy.table     import Table
from chainconsumer import ChainConsumer
#==============================================================================
def compute_bias_precision(filelist):

    xvel_lims = [-650.0,650.0]

    nfiles = len(filelist)
    # nfiles = 10
    print(" - "+str(nfiles)+" files found")

    struct = {'case':list(np.repeat('',nfiles)), 'B-spline order':np.repeat(np.nan,nfiles), 
              'S/N':np.repeat(np.nan,nfiles),'Fit type':list(np.repeat('',nfiles)), 'Model':list(np.repeat('',nfiles)), 
              'Bias':np.repeat(np.nan,nfiles),'Precision':np.repeat(np.nan,nfiles), 'ftype_name':list(np.repeat('',nfiles))}

    for i in range(nfiles):

        bname  = os.path.basename(filelist[i])
        case   = bname.split('_')[1]
        snr    = bname.split('SNR')[1].split('-')[0]
        border = bname.split('Border')[1].split('_')[0]
        ftype  = bname.split('_')[-2]
        ssp    = bname.split('_')[2]
        ftype_name = bname.split('-')[1].split('_results.hdf5')[0]

        if case == 'Double':
            case = 'Double_Gaussian'
            ssp = bname.split('_')[3]

        if ftype == 'penalised':
            ftype = 'RW'
        else:
            ftype = 'Free'        

        tab = ascii.read("../losvd/"+case+"_velscale50.dat")
        vel = tab['col1']
        tmp_losvd = tab['col2'] / np.sum(tab['col2'])   

        # Loading the infor from HDF5 file
        f          = h5py.File(filelist[i],'r')
        xvel       = np.array(f['in/xvel'])
        good       = (xvel >= xvel_lims[0]) & (xvel <= xvel_lims[1])
        true_losvd = np.interp(xvel, vel, tmp_losvd)
        losvd      = np.array(f['out/0/losvd'])
        f.close()       

        # Normalizing the LOSVDs
        # losvd_3s_lo = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
        # losvd_3s_hi = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
        # losvd_1s_lo = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
        # losvd_1s_hi = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
        # losvd_med   = losvd[2,:]/np.trapz(losvd[2,:],-xvel)
        # true_losvd  = true_losvd/np.trapz(true_losvd,-xvel)

        losvd_3s_lo = losvd[0,:]/np.sum(losvd[2,:])
        losvd_3s_hi = losvd[4,:]/np.sum(losvd[2,:])
        losvd_1s_lo = losvd[1,:]/np.sum(losvd[2,:])
        losvd_1s_hi = losvd[3,:]/np.sum(losvd[2,:])
        losvd_med   = losvd[2,:]/np.sum(losvd[2,:])
        true_losvd  = true_losvd/np.sum(true_losvd)

        # Computing statistics
        diff  = losvd_med - true_losvd
        error = 0.5*(losvd_1s_hi-losvd_1s_lo)
        bias  = np.mean(diff[good])/np.amax(true_losvd) 
        prec  = np.std(diff[good])/np.mean(error[good])

        # print(case+' '+str(snr)+' '+str(border)+' '+ftype+' '+ssp," --- ", bias," --- ", prec)

        struct['ftype_name'][i] = ftype_name
        struct['B-spline order'][i] = border
        struct['case'][i]      = case
        struct['S/N'][i]       = snr
        struct['Fit type'][i]  = ftype
        struct['Model'][i]     = ssp
        struct['Bias'][i]      = bias 
        struct['Precision'][i] = prec

        # plt.fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
        # plt.fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
        # plt.plot(xvel,losvd_med,'k.-', ds='steps-mid')
        # plt.plot(xvel, true_losvd,'r.-', ds='steps-mid') 
        # plt.axhline(y=0.0,color='k', linestyle='--')
        # plt.axvline(x=0.0, color='k', linestyle="--")
        # plt.axvline(x=xvel_lims[0],color='k', linestyle=':')
        # plt.axvline(x=xvel_lims[1], color='k', linestyle=":")
        # plt.plot(xvel, diff, 'gray', ds='steps-mid', linestyle='--')
        # plt.axhline(y=np.mean(error),color='gray', linestyle='--')
        # plt.title(case+' '+str(snr)+' '+str(border)+' '+ftype+' '+ssp)
        # plt.show()
        # exit() 

    return Table(struct)
