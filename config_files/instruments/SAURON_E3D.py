import numpy              as np
from   astropy.io         import fits
#===============================================================================
def read_data(filename):

    hdu = fits.open(filename)
    hdr  = hdu[1].header
    data = hdu[1].data
    #---------------------------
    spec  = data['DATA_SPE'].T
    espec = np.sqrt(data['STAT_SPE'].T)
    x     = data['XPOS']
    y     = data['YPOS']
    npix  = spec.shape[0]
    nspax = spec.shape[1]
    wave  = hdr['CRVALS'] + hdr['CDELTS']*np.arange(npix)
    psize = 1.0

    struct = {'wave':wave, 'spec':spec, 'espec':espec, 'x':x, 'y':y, 'npix':npix, 'nspax':nspax, 'psize':psize,'ndim':2}

    return struct