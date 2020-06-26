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
    xaxis = np.arange(spec.shape[2])*hdr['CD2_2']*3600.0
    yaxis = np.arange(spec.shape[1])*hdr['CD2_2']*3600.0
    x, y  = np.meshgrid(xaxis,yaxis)
    x, y  = x.ravel(), y.ravel()
    psize = np.abs(x[1]-x[0])
    npix  = spec.shape[0]
    nspax = spec.shape[1]*spec.shape[2]
    wave  = hdr['CRVAL3'] + hdr['CDELT3']*np.arange(npix)

    # Reshaping the 3D cube to 2D
    spec  = np.reshape(spec,(npix,nspax))
    espec = np.reshape(espec,(npix,nspax))

    struct = {'wave':wave, 'spec':spec, 'espec':espec, 'x':x, 'y':y, 'npix':npix, 'nspax':nspax, 'psize':psize}

    return struct       