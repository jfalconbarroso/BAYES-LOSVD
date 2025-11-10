import numpy              as np
from   astropy.io         import fits
#===============================================================================
def read_data(filename):

    hdu = fits.open(filename)
    if len(hdu) < 3:
        print("The MUSE datacube needs 3 extensions: [0] Primary, [1] Data, [2] Variance")
        exit()

    #---------------------------
    hdr   = hdu[1].header
    spec  = hdu[1].data
    espec = np.sqrt(hdu[2].data) # We need the stddev here, not the variance
    #---------------------------
    xaxis = np.arange(spec.shape[2])*hdr['CD2_2']*3600.0
    yaxis = np.arange(spec.shape[1])*hdr['CD2_2']*3600.0
    x, y  = np.meshgrid(xaxis,yaxis)
    x, y  = x.ravel(), y.ravel()
    npix  = spec.shape[0]
    nspax = spec.shape[1]*spec.shape[2]
    wave  = hdr['CRVAL3'] + hdr['CD3_3']*np.arange(npix)
    psize = 0.2 # arcsec per spaxel
    
    # Reshaping the 3D cube to 2D
    spec  = np.reshape(spec,(npix,nspax))
    espec = np.reshape(espec,(npix,nspax))

    struct = {'wave':wave, 'spec':spec, 'espec':espec, 'x':x, 'y':y, 'npix':npix, 'nspax':nspax, 'psize':psize, 'ndim':2}

    return struct       