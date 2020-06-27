import numpy              as np
from   astropy.io         import fits
#===============================================================================
def read_data(filename):

    hdu = fits.open(filename)
    if len(hdu) < 2:
        print("ERROR: The CALIFA datacube needs 2 extensions: [0] Data, [1] Dispersion")
        exit()

    #---------------------------
    hdr   = hdu[0].header
    spec  = hdu[0].data
    espec = hdu[1].data 
    #---------------------------
    x     = np.arange(spec.shape[2])
    y     = np.zeros(spec.shape[2])
    psize = 1.0
    npix  = spec.shape[0]
    nspax = spec.shape[2]
    wave  = hdr['CRVAL1'] + hdr['CDELT1']*np.arange(npix)

    struct = {'wave':wave, 'spec':spec, 'espec':espec, 'x':x, 'y':y, 'npix':npix, 'nspax':nspax, 'psize':psize, 'ndim':1}

    return struct       