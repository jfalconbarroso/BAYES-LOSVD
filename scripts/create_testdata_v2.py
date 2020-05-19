import os
import sys
import glob
import h5py
import warnings
import optparse
import numpy                 as np
import matplotlib.pyplot     as plt
import lib.misc_functions    as misc
import lib.cap_utils         as cap
from   astropy.io            import ascii, fits
from   astropy.convolution   import convolve
#==============================================================================
def create_testdata(cases, models, snr, nsim=25):

    # Defining basic stuff
    ncases     = len(cases)
    model_list = np.array(glob.glob("../templates/"+models+"/*.fits"))
    nmodels    = len(model_list)
    nsnr       = len(snr)
    velscale   = 50.0

    # Randomly shuffling the models    
    tmp = np.arange(nmodels)
    np.random.shuffle(tmp)
    idx = tmp[0:nsim]
    model_list = model_list[idx]

    # Defining the size of some arrays
    hdu   = fits.open(model_list[0])
    hdr   = hdu[0].header
    spec  = hdu[0].data
    npix  = len(spec)
    wave  = hdr['CRVAL1']+np.arange(npix)*hdr['CDELT1']
    lamRange = np.array([np.amin(wave),np.amax(wave)])
    lspec, lwave, dummy = cap.log_rebin(lamRange, spec, velscale=velscale)
    npix = len(lspec)
    
    for i in range(ncases):
        
        misc.printRUNNING(cases[i]) 

        # Reading the LOSVD
        losvd_file = '../losvd/'+cases[i]+'_velscale50.dat'
        tab        = ascii.read(losvd_file)
        xvel       = np.array(tab['col1'])
        losvd      = np.array(tab['col2'])
        velscale   = xvel[1]-xvel[0]
 
        # Defining output filename and array sizes
        # outhdf5   = '../data/testcases_'+cases[i]+'_'+models+'.hdf5'
        outhdf5   = '../data/testcases_'+cases[i]+'.hdf5'
        out_spec  = np.zeros((npix,nsnr*nsim)) * np.nan
        out_espec = np.zeros((npix,nsnr*nsim)) * np.nan
        spec_snr  = np.zeros(nsnr*nsim) * np.nan
        o = 0
        for j in range(nsnr):
            
            for k in range(nsim):

                # Reading the input spectrum
                hdu   = fits.open(model_list[k])
                hdr   = hdu[0].header
                spec  = hdu[0].data
                npix  = len(spec)
                wave  = hdr['CRVAL1']+np.arange(npix)*hdr['CDELT1']
                spec /= np.mean(spec)

                # Log-rebinning the data
                lamRange = np.array([np.amin(wave),np.amax(wave)])
                lspec, lwave, dummy = cap.log_rebin(lamRange, spec, velscale=velscale)

                # Convolving the input spectrum with the LOSVD
                lspec = convolve(lspec, losvd, boundary='fill', fill_value=0.0)
                npix  = lwave.shape[0]
            
                # Estimating noise based on SNR and adding it
                sig   = (1.0/snr[j])
                noise = np.random.normal(loc=0.0,scale=sig,size=npix)
                out_espec[:,o] = sig*np.ones_like(lspec)
                out_spec[:,o]  = lspec + noise
                spec_snr[o]    = snr[j]
                o += 1

        if not os.path.exists("../data"):
              os.mkdir("../data")
        if os.path.exists(outhdf5):
              os.remove(outhdf5)
        
        print("  - Saving result: "+outhdf5)
        print("")
        f = h5py.File(outhdf5,"w")
        f.create_dataset('WAVE',     data=lwave,     compression="gzip")
        f.create_dataset('SPEC',     data=out_spec,  compression="gzip")
        f.create_dataset('ESPEC',    data=out_espec, compression="gzip")
        f.create_dataset('SNR',      data=spec_snr,  compression="gzip")
        f.create_dataset('SOURCE',   data=[n.encode("ascii", "ignore") for n in model_list], dtype='S100', compression="gzip" )
        f.create_dataset('VELSCALE', data=velscale)
        f.close()

        misc.printDONE(outhdf5+" created.")

    return

#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("             CREATE_TESTDATA               ")
    print("===========================================")
    print("")

    # cases  = ['Gaussian','Double_Gaussian','Marie1','Marie2']
    cases  = ['Wings']
    models = 'MILES_SSP'
    snr    = [10,25,50,100,200]

    create_testdata(cases, models, snr)

    exit()
