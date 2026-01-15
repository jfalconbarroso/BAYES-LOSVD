import numpy      as np
from   glob       import glob
from   tqdm.auto  import trange
from   astropy.io import fits
#===============================================================================
def read_templates(template_lib):

    # Loading the list of templates
    filelist = glob('../templates/'+template_lib+'/*fits')
    ntemp    = len(filelist) 
    print(" - "+str(ntemp)+" templates found in "+template_lib+" library")

    # Reading the templates into the output arrays
    hdu  = fits.open(filelist[0])
    tmp  = hdu[0].data.ravel()
    hdr  = hdu[0].header
    npix = len(tmp)
    wave = hdr['CRVAL1'] + hdr['CDELT1']*np.arange(npix)
    temp = np.zeros((npix,ntemp))
    wgood = (wave >= 4800) & (wave <= 4900)
    for i in trange(ntemp, ascii=True, leave=False):
        hdu = fits.open(filelist[i])
        temp[:,i]  = hdu[0].data.ravel()
        scale      = np.mean(temp[:,i])
        temp[:,i] /= scale      

    # Extracting parameter information
    # NOTE 1: Use zeros if not known
    # NOTE 2: The array must have size (nparam, ntemp)
    params = np.zeros((3,ntemp))

    return wave, temp, ntemp, npix, params


