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
def create_testdata(tab, idx):

    misc.printRUNNING(tab['FILENAME'][idx]+" SNR: "+str(tab['SNR'][idx])+" LOSVD: "+tab['LOSVD_FILE'][idx]) 

    # Reading input parameters
    filename   = '../templates/'+tab['FILENAME'][idx]
    snr        = tab['SNR'][idx]
    losvd_file = '../losvd/'+tab['LOSVD_FILE'][idx]
    outfits    = '../data/'+tab['OUTNAME'][idx]

    # Reading the input spectrum
    hdu   = fits.open(filename)
    hdr   = hdu[0].header
    spec  = hdu[0].data
    npix  = len(spec)
    wave  = hdr['CRVAL1']+np.arange(npix)*hdr['CDELT1']
    spec /= np.mean(spec)

    # Reading the LOSVD
    tab      = ascii.read(losvd_file)
    xvel     = np.array(tab['col1'])
    losvd    = np.array(tab['col2'])
    velscale = xvel[1]-xvel[0]
    print("  - Velscale: "+str(velscale))

    # Log-rebinning the data
    lamRange = np.array([np.amin(wave),np.amax(wave)])
    lspec, lwave, dummy = cap.log_rebin(lamRange, spec, velscale=velscale)

    # Convolving the input spectrum with the LOSVD
    lspec = convolve(lspec, losvd, boundary='fill', fill_value=0.0)
    npix  = lwave.shape[0]

    # Estimating noise based on SNR and adding it
    sig    = (1.0/snr)
    lespec = sig*np.ones_like(lspec)
    lspec  = lspec + np.random.normal(loc=0.0,scale=sig,size=len(lspec))

    if not os.path.exists("../data"):
          os.mkdir("../data")
    if os.path.exists(outfits):
          os.remove(outfits)
    
    print("  - Saving result: "+outfits)
    print("")
    c1 = fits.Column(name='WAVE',  format='D', array=lwave)
    c2 = fits.Column(name='SPEC',  format='D', array=lspec)
    c3 = fits.Column(name='ESPEC', format='D', array=lespec)
    t  = fits.BinTableHDU.from_columns([c1, c2, c3])
    t.writeto(outfits)
    fits.setval(outfits, 'VELSCALE', value=velscale)
    fits.setval(outfits, 'SNR',      value=snr)
    fits.setval(outfits, 'LOSVD',    value=losvd_file)

    misc.printDONE(outfits+" created.")

    return

#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("             CREATE_TESTDATA               ")
    print("===========================================")
    print("")

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -c cfile")
    parser.add_option("-c", "--config",  dest="config_file", type="string", default="../config_files/default_testdata.conf", help="Filename of the general config file")

    (options, args) = parser.parse_args()
    config_file = options.config_file
     
    # Load config file
    tab    = ascii.read(config_file, names=['FILENAME','SNR','LOSVD_FILE','OUTNAME'], comment='#')
    nfiles = len(tab['FILENAME']) 

    for i in range(nfiles):
        create_testdata(tab,i)

    exit()
