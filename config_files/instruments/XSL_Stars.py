import numpy              as np
from   astropy.io         import fits
#===============================================================================
def read_data(filename):

    hdu = fits.open(filename)
    if len(hdu) < 1:
        print("ERROR: The datafile needs at least 1 extension: [0] Data")
        exit()

    #---------------------------
    hdr   = hdu[0].header
    spec  = hdu[1].data
    espec = spec * 0.0
    #---------------------------
    x     = np.arange(spec.shape[1])
    y     = np.zeros(spec.shape[1])
    psize = 1.0
    npix  = spec.shape[0]
    nspax = spec.shape[1]
    wave  = hdr['CRVAL1'] + hdr['CDELT1']*np.arange(npix)
    wave  /= 1E4 # to convert to microns

    struct = {'wave':wave, 'spec':spec, 'espec':espec, 'x':x, 'y':y, 'npix':npix, 'nspax':nspax, 'psize':psize, 'ndim':1}

    return struct