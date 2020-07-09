import numpy              as np
from   astropy.io         import fits
import matplotlib.pyplot as plt
#===============================================================================
def read_data(filename):

    hdu = fits.open(filename)
    if len(hdu) < 15:
        print("ERROR: The datacube does not conform the MaNGA LINCUBE datamodel standard")
        exit()

    #---------------------------
    hdr   = hdu[1].header
    spec  = hdu[1].data
    espec = np.sqrt(1.0/hdu[2].data)
    mask  = hdu[3].data
    wave  = hdu[4].data
    #---------------------------
    xaxis = (np.arange(spec.shape[1])-(hdr['CRPIX1']-1))*np.fabs(hdr['CD1_1'])*3600.0
    yaxis = (np.arange(spec.shape[2])-(hdr['CRPIX2']-1))*np.fabs(hdr['CD2_2'])*3600.0
    x, y  = np.meshgrid(xaxis,yaxis)
    x, y  = x.ravel(), y.ravel()
    psize = np.abs(x[1]-x[0])
    npix  = spec.shape[0]
    nspax = spec.shape[1]*spec.shape[2]

    # Reshaping the 3D cube to 2D
    spec  = np.reshape(spec,(npix,nspax))
    espec = np.reshape(espec,(npix,nspax))
    mask  = np.reshape(mask,(npix,nspax))

    # Removing remaining spectra with all NaNs
    check = np.nansum(spec,axis=0)
    good  = (check > 0)
    if np.sum(good) > 0:
       spec  = spec[:,good]
       espec = espec[:,good]
       mask  = mask[:,good]
       x, y  = x[good], y[good]
       nspax = np.sum(good)

    # Setting those pixels with non-zero values in the mask with large variance
    # NOTE: This means that even if not formally masked, they 
    #       will hardly affect the final fit.
    for i in range(nspax):
        check = (mask[:,i] > 0.0)
        spec[check,i]  = 0.0
        espec[check,i] = 1E10

    # Creating the relevant LSF for the particular file
    # NOTE: this is important as the LSF in MaNGA changes
    #       with the galaxy.
    res = hdu[5].data
    fwhm = wave / res
    out  = np.zeros((npix,2))
    out[:,0] = wave
    out[:,1] = fwhm
    np.savetxt("../config_files/instruments/MANGA.lsf", out, header="Lambda  FWHM")
   
    struct = {'wave':wave, 'spec':spec, 'espec':espec, 'x':x, 'y':y, 'npix':npix, 'nspax':nspax, 'psize':psize, 'ndim':2}

    return struct       