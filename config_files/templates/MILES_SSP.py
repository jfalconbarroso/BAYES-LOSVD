import re
import numpy      as np
from   glob       import glob
from   tqdm.auto  import trange
from   astropy.io import fits
#===============================================================================
def extract_ages_and_metallicities(filenames):
    ages = []
    metallicities = []

    for fname in filenames:
        # Extract metallicity
        z_match = re.search(r'Z([mp])(\d+\.\d+)', fname)
        if z_match:
            z_sign = '-' if z_match.group(1) == 'm' else '+'
            metallicity = float(z_sign + z_match.group(2))
        else:
            metallicity = None

        # Extract age
        t_match = re.search(r'T(\d+\.\d+)', fname)
        age = float(t_match.group(1)) if t_match else None

        ages.append(age)
        metallicities.append(metallicity)

    return ages, metallicities

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
    params = np.zeros((2,ntemp))
    age, met = extract_ages_and_metallicities(filelist)
    params[0,:] = age
    params[1,:] = met

    return wave, temp, ntemp, npix, params


