import numpy              as np
from   astropy.io         import fits
#===============================================================================
def read_data(filename):

    hdu = fits.open(filename)
    if len(hdu) < 2:
        print("ERROR: The SAMI datacube needs 2 extensions: [0] Data, [1] Dispersion")
        exit()

    #---------------------------
    hdr   = hdu[0].header
    spec  = hdu[0].data
    espec = np.sqrt(hdu[1].data)
    #---------------------------
    xaxis = np.arange(spec.shape[2])*hdr['CDELT2']*3600.0
    yaxis = np.arange(spec.shape[1])*hdr['CDELT2']*3600.0
    x, y  = np.meshgrid(xaxis,yaxis)
    x, y  = x.ravel(), y.ravel()
    psize = np.abs(x[1]-x[0])
    npix  = spec.shape[0]
    nspax = spec.shape[1]*spec.shape[2]
    wave  = hdr['CRVAL3'] + hdr['CDELT3']*(np.arange(npix)-hdr['CRPIX3'])

    # Reshaping the 3D cube to 2D
    spec  = np.reshape(spec,(npix,nspax))
    espec = np.reshape(espec,(npix,nspax))

    # Removing remaining spectra with all NaNs
    check = np.nansum(spec,axis=0)
    good  = (check > 0)
    if np.sum(good) > 0:
       spec  = spec[:,good]
       espec = espec[:,good]
       x, y  = x[good], y[good]
       nspax = np.sum(good)

    # Setting those pixels with NaNs to zero with large variance
    # NOTE: This means that even if not formally masked, they 
    #       will hardly affect the final fit.
    for i in range(nspax):
        check = np.isnan(spec[:,i])
        spec[check,i]  = 0.0
        espec[check,i] = 1E10

    struct = {'wave':wave, 'spec':spec, 'espec':espec, 'x':x, 'y':y, 'npix':npix, 'nspax':nspax, 'psize':psize, 'ndim':2}

    return struct       