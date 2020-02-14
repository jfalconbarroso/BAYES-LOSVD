import numpy                 as np
from   astropy.io            import fits
#==============================================================================
if (__name__ == '__main__'):

    frac    = 0.5
    ssp1    = "../MILES_SSP/Mkb1.30Zp0.06T00.5000_iTp0.00_baseFe.fits"
    ssp2    = "../MILES_SSP/Mkb1.30Zp0.06T05.0000_iTp0.00_baseFe.fits"
    outfits = "Mixed_pops_50pc_0.5Gyr_50pc_5.0Gyr_solar.fits"

    # Reading and mixing templates
    hdu   = fits.open(ssp1)
    spec1 = hdu[0].data 
    hdr   = hdu[0].header
    hdu   = fits.open(ssp2)
    spec2 = hdu[0].data 

    spec  = frac * (spec1/np.mean(spec1)) + (1.0-frac)*(spec2/np.mean(spec2))


    # Writing file on disk
    hdu = fits.PrimaryHDU(spec)
    hdu.writeto(outfits)
    fits.setval(outfits, 'CRPIX1', value=1)
    fits.setval(outfits, 'CRVAL1', value=hdr['CRVAL1'])
    fits.setval(outfits, 'CDELT1', value=hdr['CDELT1'])

    exit()