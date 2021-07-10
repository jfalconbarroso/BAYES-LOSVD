import os
import sys
import glob
import h5py
import toml
import arviz                 as az
import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
from   astropy.io            import fits, ascii
from   astropy.convolution   import convolve
from   scipy.interpolate     import interp1d
#===============================================================================
def printDONE(outstr=""):

    print("")
    print("")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r[ "+'\033[0;32m'+"DONE "+'\033[0;39m'+"] "+outstr)
    sys.stdout.flush()
    print("")
    print("")

    return

#==============================================================================
def printFAILED(outstr=""):

    print("")
    print("")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r[ "+'\033[0;31m'+"FAILED "+'\033[0;39m'+"] "+outstr)
    sys.stdout.flush()
    print("")
    print("")

    return
#==============================================================================
def printRUNNING(outstr=""):

    print("")
    print("")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r[ "+'\033[0;33m'+"RUNNING "+'\033[0;39m'+"] "+outstr)
    sys.stdout.flush()
    print("")
    print("")

    return

#==============================================================================
def printWARNING(outstr=""):

    print("")
    print("")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r[ "+'\033[0;36m'+"WARNING "+'\033[0;39m'+"] "+outstr)
    sys.stdout.flush()
    print("")
    print("")

    return


#==============================================================================
def create_gh_losvd(pars,velscale):
    
    # Defining parameters in pixels 
    vel   = pars[0]/velscale     # in pixels
    sigma = pars[1]/velscale     # in pixels
    dx    = np.ceil(np.abs(vel)+6.0*sigma) # Sample the Gaussian and GH at least to vel+6*sigma
    xvel  = np.linspace(-dx,dx,2*dx+1)     # Evaluate the Gaussian using steps of 1 pixel.
    w     = (xvel - vel)/sigma
    w2    = w**2

    # Defining the Gaussian part of the LOSVD
    losvd = np.exp(-0.5*w2)/(np.sqrt(2.0*np.pi)*sigma) # Normalized total(Gaussian)=1

    # Creating the Gauss-Hermits part
    H3   = pars[2]/np.sqrt(3.0)*(w*(2.0*w2-3.0))
    H4   = pars[3]/np.sqrt(24.0)*(w2*(4.0*w2-12.0)+3.0)
    poly = 1.0 + H3 + H4
    
    # Defining the final LOSVD
    losvd = losvd*poly
    
    # Forcing the LOSVD to be positive
    bad = (losvd <= 0.0)
    losvd[bad] = 1E-10
    
    # Normalizing the LOSVD so that the sum=1
    losvd /= np.sum(losvd)

    return losvd, xvel*velscale, xvel.shape

#==============================================================================
def mirror_vector(maxval, inc=1):
    x = np.arange(inc, maxval, inc)
    if x[-1] != maxval:
        x = np.r_[x, maxval]

    return np.r_[-x[::-1], 0, x]

#==============================================================================
def read_lsf(wave,lsf_file):
    
    # Reading the LSF file
    tab = ascii.read(lsf_file)
     
    # Interpolating LSF at input wavelengths
    f   = interp1d(tab['Lambda'],tab['FWHM'], bounds_error=False, fill_value=np.nan)
    out = f(wave)
    
    return out

#==============================================================================
def read_code(fit_type):

    codes_file = "../config_files/codes.properties"
    config     = toml.load(codes_file)
    codefile   = "stan_model/"+config[fit_type]['codefile']    
    
    extrapars = {}
    for key, val in config[fit_type].items():
        if key != 'codefile':
            extrapars[key] = val

    return codefile, extrapars

#==============================================================================
def load_configfile(file=''):

    colnames = ["Runname","Survey","Redshift","Lmin","Lmax","Vmax","Velscale","SNR","SNR_min","Porder","Templates","NPCA","Mask","Mask_width"]

    tab = ascii.read(file, names=colnames, comment="#")

    return tab

#==============================================================================
def pad_spectra(tab):

    pad = int(np.floor(tab['nvel']/2))
    templates_padded     = np.pad(tab['templates'],((pad,0),(0,0)), mode="constant", constant_values=0.0)
    mean_template_padded = np.pad(tab['mean_template'],(pad,0),     mode="constant", constant_values=0.0)
    npix_temp = templates_padded.shape[0]
    
    # Adjusting the models wavelength accordingly
    dwav = tab['lwave_temp'][1]-tab['lwave_temp'][0]
    lam0 = tab['lwave_temp'][0]-pad*dwav
    lam1 = tab['lwave_temp'][0]-dwav
    wave_temp        = np.zeros(npix_temp)
    wave_temp[0:pad] = np.linspace(lam0,lam1,pad)
    wave_temp[pad:]  = tab['lwave_temp']
    wave_temp        = wave_temp[0:-2]

    return wave_temp, templates_padded, mean_template_padded, npix_temp, pad
    
#==============================================================================
def log_unbinning(lamRange, spec, oversample=1, flux=True):
    """
    This function transforms logarithmically binned spectra back to linear
    binning. It is a Python translation of Michele Cappellari's
    "log_rebin_invert" function. Thanks to Michele Cappellari for his permission
    to include this function in the pipeline.
    """
    # Length of arrays
    n = len(spec)
    m = n * oversample

    # Log space
    dLam = (lamRange[1]-lamRange[0]) / (n - 1)             # Step in log-space
    lim = lamRange + np.array([-0.5, 0.5])*dLam            # Min and max wavelength in log-space
    borders = np.linspace( lim[0], lim[1], n+1 )           # OLD logLam in log-space

    # Wavelength domain
    logLim     = np.exp(lim)                               # Min and max wavelength in Angst.
    lamNew     = np.linspace( logLim[0], logLim[1], m+1 )  # new logLam in Angstroem
    newBorders = np.log(lamNew)                            # new logLam in log-space

    # Translate indices of arrays so that newBorders[j] corresponds to borders[k[j]]
    k = np.floor( (newBorders-lim[0]) / dLam ).astype('int')

    # Construct new spectrum
    specNew = np.zeros(m)
    for j in range(0, m-1):
        a = (newBorders[j]   - borders[k[j]])   / dLam
        b = (borders[k[j+1]] - newBorders[j+1]) / dLam

        specNew[j] = np.sum( spec[k[j]:k[j+1]] ) - a*spec[k[j]] - b*spec[k[j+1]]

    # Rescale flux
    if flux == True:
        specNew = specNew / ( newBorders[1:] - newBorders[:-1] ) * np.mean( newBorders[1:] - newBorders[:-1] ) * oversample

    # Shift back the wavelength arrays
    lamNew = lamNew[:-1] + 0.5 * (lamNew[1]-lamNew[0])

    return( specNew, lamNew )

#==============================================================================
def save_stan_summary(samples, unwanted=None, summary_filename=None, verbose=None):

    # samples   = fit.extract(permuted=True) # Extracting parameter samples
    var_all   = list(samples.keys())
    var_names = [e for e in var_all if e not in unwanted]
    stan_summary = az.summary(samples,var_names=var_names)
    if (verbose == True):
        print("")
        print(stan_summary)

    if os.path.exists(summary_filename):
       os.remove(summary_filename)
    f = open(summary_filename, 'w')
    f.write(stan_summary.to_string())
    f.close()       

    return
#==============================================================================
def process_stan_output_per(struct, samples, outhdf5=None, stridx=None):

    lims = [0.1,15.9,50.0,84.1,99.9] # Saving 1-sigma, 3-sigma, and median

    if os.path.exists(outhdf5):
       os.remove(outhdf5)

    # Opening the pointer to the output file
    f = h5py.File(outhdf5, "w")

    # Adding all the input and exisiting output variables
    struct.copy('in',f)
    if not (struct.get('out') == None):
        struct.copy('out/'+stridx, f, name='out/'+stridx)

    # Adding all the new outputs
    for key in samples.keys():
        if np.ndim(samples[key]) == 1:
            result = np.percentile(np.array(samples[key]), q=lims)
        elif np.ndim(samples[key]) > 1:
            result = np.percentile(np.array(samples[key]).T, q=lims, axis=1)
        if not (key == 'lp__'):
            f.create_dataset("out/"+stridx+"/"+key, data=result, compression="gzip")
    f.close()

    return

#==============================================================================
def process_stan_output_hdp(struct, samples, outhdf5=None, stridx=None):

    lims = [0.50, 0.68, 0.999] # Saving 1-sigma, 3-sigma, and median

    if os.path.exists(outhdf5):
       os.remove(outhdf5)

    # Opening the pointer to the output file
    f = h5py.File(outhdf5, "w")

    # Adding all the input and exisiting output variables
    struct.copy('in',f)
    if not (struct.get('out') == None):
        struct.copy('out/'+stridx, f, name='out/'+stridx)

    # Adding all the processed outputs
    for key in samples.keys():
        result = compute_hdp(np.array(samples[key]).T, lims)
        if not (key == 'lp__'):
           f.create_dataset("out/"+stridx+"/"+key, data=result, compression="gzip")               

    f.close()

    return
#==============================================================================
def compute_hdp(samples,lims):
    
    ndim = np.ndim(samples)
    if ndim == 1:
       result = np.zeros(5)
    else:
       size   = samples.shape 
       result = np.zeros((5,size[1]))

    if ndim == 1:
       for i in range(len(lims)):
           kk = az.hdi(samples, hdi_prob=lims[i])
        #    kk = az.hpd(samples, credible_interval=lims[i])
           if i == 0:
               result[2] = np.mean(kk)
           elif i == 1:
               result[1] = kk[0]
               result[3] = kk[1]
           elif i == 2:
               result[0] = kk[0]
               result[4] = kk[1]
    else:
       for j in range(size[1]): 
           for i in range(len(lims)):
               kk = az.hdi(samples[:,j], hdi_prob=lims[i])
            #    kk = az.hpd(samples[:,j], credible_interval=lims[i])
               if i == 0:
                   result[2,j] = np.mean(kk)
               elif i == 1:
                   result[1,j] = kk[0]
                   result[3,j] = kk[1]
               elif i == 2:
                   result[0,j] = kk[0]
                   result[4,j] = kk[1]

    return result
#==============================================================================
def delete_files(inputfile, extension=None):

    dirname, filename = os.path.split(inputfile) 
    basename = os.path.splitext(filename)[0]
    fileList = glob.glob(dirname+'/'+basename+'*.'+extension)

    for filePath in fileList:
        if os.path.exists(filePath):
           os.remove(filePath)

    return       

#==============================================================================
def pack_results(rootname,suffix='', dir='../results/'):

    input_list = glob.glob(dir+rootname+"/"+rootname+suffix+"_results_bin*.hdf5")
    nlist = len(input_list)
    if nlist == 0:
       print(" - Nothing to pack!")
       print("")
       print(dir+rootname+"/"+rootname+suffix+"_results_bin*.hdf5")
       return 
    else:   
       print(" - "+str(nlist)+" files found.")

    # Preparing outputfile
    outfile = dir+rootname+"/"+rootname+suffix+"_results.hdf5"
    if os.path.exists(outfile):
       os.remove(outfile)
    g = h5py.File(outfile,'w')

    print(" - Copying contents to output file")
    for infile in input_list:
       f = h5py.File(infile,'r')

       # Copying the input data       
       if (infile == input_list[0]):
           f.copy("in",g)
    
       # Copying the output results
       members = []
       f.visit(members.append)
       for i in range(len(members)):
           check = members[i].split("/")
           if (check[0] == 'out') & (len(check) == 3):
              f.copy(members[i],g,name=members[i])
       f.close()
    g.close()

    # Checking file is in place and cleaning up
    if os.path.exists(outfile):    
        for filePath in input_list:
            if os.path.exists(filePath):
               os.remove(filePath)
         
    printDONE(outfile+" written.")
 
    return

#============================================================================== 
def print_attrs(name,obj):

    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
 
    return
  
 #============================================================================== 
def check_hdf5_tree(filename):
        
    f = h5py.File(filename,'r')
    f.visititems(print_attrs)
 
    return

 #============================================================================== 
def check_configuration(struct):

    l = ['filename','instrument','redshift','lmin','lmax','vmax','velscale',
         'snr','snr_min','porder','template_lib','npca','mask_file']

    for key in l:
        if key not in struct.keys():
            printFAILED("keyword '"+key+"' not found in configuration file")
            sys.exit()

    return True

 #============================================================================== 
def check_codes(fit_type):

    codes_file = "../config_files/codes.properties"
    if not os.path.exists(codes_file):
        printFAILED("codes.properties not found in 'config_files' directory")
        sys.exit()

    config = toml.load(codes_file)
    
    if fit_type not in config.keys():
        printFAILED("Fit type '"+fit_type+"' not found in codes file.")
        print("Available options are:")
        for key in config.keys():
            print(" - ",key)
        sys.exit()

    return True

 #============================================================================== 
def spectralMasking(maskfile, logLam,redshift):

    """ Mask spectral region in the fit. 
        Adapted from GIST pipeline
    """
    # Read file
    mask        = np.genfromtxt(maskfile, usecols=(0,1))
    maskComment = np.genfromtxt(maskfile, usecols=(2), dtype=str )
    goodPixels  = np.arange( len(logLam) )

    # In case there is only one mask
    if len( mask.shape ) == 1  and  mask.shape[0] != 0:
        mask        = mask.reshape(1,2)
        maskComment = maskComment.reshape(1)

    for i in range( mask.shape[0] ):

        # Check for sky-lines
        if maskComment[i] == 'sky'  or  maskComment[i] == 'SKY'  or  maskComment[i] == 'Sky':
            mask[i,0] = mask[i,0] / (1+redshift)

        # Define masked pixel range
        minimumPixel = int( np.round( ( np.log( mask[i,0] - mask[i,1]/2. ) - logLam[0] ) / (logLam[1] - logLam[0]) ) )
        maximumPixel = int( np.round( ( np.log( mask[i,0] + mask[i,1]/2. ) - logLam[0] ) / (logLam[1] - logLam[0]) ) )

        # Handle border of wavelength range
        if minimumPixel < 0:            minimumPixel = 0
        if maximumPixel < 0:            maximumPixel = 0
        if minimumPixel >= len(logLam): minimumPixel = len(logLam)-1
        if maximumPixel >= len(logLam): maximumPixel = len(logLam)-1

        # Mark masked spectral pixels
        goodPixels[minimumPixel:maximumPixel+1] = -1

    goodPixels = goodPixels[ np.where( goodPixels != -1 )[0] ]

    return(goodPixels)

 #============================================================================== 
def create_bins_list(bin, nbins, mask_bin, outdir, restart):

    if (bin == "all"):
       bin_list = list(np.arange(nbins))
       print("# ENTIRE list of bins selected")

    elif (bin == "odd"):
       bin_list = list(np.arange(0,nbins,2)) 
       print("# ODD bins selected")

    elif (bin == "even"):
       bin_list = list(np.arange(1,nbins,2)) 
       print("# EVEN bins selected")

    else:
       bin_list = list(np.array(bin.split(","),dtype=int))
       nbins    = len(bin_list)
       print("# Selected bins: "+bin)
    
    # Masking undesired bins
    if mask_bin != "None":
        print("# Masking bins: "+mask_bin)
        bad_bins = list(np.array(mask_bin.split(","),dtype=int))
        bin_list = np.setdiff1d(bin_list, bad_bins, assume_unique=False)
        nbins    = len(bin_list)
    else:
        print("# No mask to be applied to input bin list")

    # Updating list if restart option is activated
    if restart:
        print("# Restart flag is on. Updating the input list")
        flist = glob.glob(outdir+"/*bin*.hdf5")
        bins_on_disk = []
        for file in flist:
            bins_on_disk = np.append(bins_on_disk,file.split("bin")[1].split(".hdf5")[0])
        
        bins_on_disk = np.array(np.sort(bins_on_disk), dtype=int)
        bin_list = np.setdiff1d(bin_list,bins_on_disk, assume_unique=False)
        nbins_on_disk = len(bins_on_disk)
        nbins = len(bin_list)
        print(" - "+str(nbins_on_disk)+" bins found on disk. Running "+str(nbins)+" bins.")

    return bin_list, nbins

#============================================================================== 
def create_xvel_vector(struct):

    # Creating the LOSVD velocity vector
    vmax     = struct['vmax']
    velscale = struct['velscale']
    xvel     = mirror_vector(vmax,inc=velscale)
    if (xvel[1]-xvel[0] < velscale):
        xvel = xvel[1:-1]
    xvel = np.flip(xvel) # The flipping is necessary because of the way the convolution is done
    nvel = len(xvel)    

    return xvel, nvel

#============================================================================== 
def pad_templates(struct, nvel):

    # Loading basic info from input structure
    lwave     = struct['lwave_temp']
    mean_temp = struct['mean_template']
    templates = struct['templates']
    npix_temp = struct['npix_temp']

    # Padding the templates
    npad      = int(np.floor(nvel/2))
    mean_temp = np.pad(mean_temp, pad_width=(npad,npad+1),         mode='edge')
    templates = np.pad(templates, pad_width=((npad,npad+1),(0,0)), mode='edge')
    dwave     = lwave[1]-lwave[0]
    lwave_pre = lwave[0]  - dwave*(np.arange(npad)+1)
    lwave_pos = lwave[-1] + dwave*(np.arange(npad+1)+1)
    lwave_new = np.concatenate((lwave_pre,lwave,lwave_pos))

    # Updating input structure
    new_struct = struct.copy()
    new_struct['lwave_temp']    = lwave_new
    new_struct['mean_template'] = mean_temp
    new_struct['templates']     = templates
    new_struct['npix_temp']     = len(lwave_new)

    return new_struct

#============================================================================== 
def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Parameters

    Taken from: https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py
    ----------
    new_wavs : np.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.
    spec_wavs : np.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.
    spec_fluxes : np.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.
    spec_errs : np.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.
    fill : float (optional)
        Value for all new_fluxes and new_errs that fall outside the
        wavelength range in spec_wavs. These will be nan by default.
    Returns
    -------
    new_fluxes : np.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.
    new_errs : np.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Arrays of left hand sides and widths for the old and new bins
    old_lhs = np.zeros(old_wavs.shape[0])
    old_widths = np.zeros(old_wavs.shape[0])
    old_lhs = np.zeros(old_wavs.shape[0])
    old_lhs[0] = old_wavs[0]
    old_lhs[0] -= (old_wavs[1] - old_wavs[0])/2
    old_widths[-1] = (old_wavs[-1] - old_wavs[-2])
    old_lhs[1:] = (old_wavs[1:] + old_wavs[:-1])/2
    old_widths[:-1] = old_lhs[1:] - old_lhs[:-1]
    old_max_wav = old_lhs[-1] + old_widths[-1]

    new_lhs = np.zeros(new_wavs.shape[0]+1)
    new_widths = np.zeros(new_wavs.shape[0])
    new_lhs[0] = new_wavs[0]
    new_lhs[0] -= (new_wavs[1] - new_wavs[0])/2
    new_widths[-1] = (new_wavs[-1] - new_wavs[-2])
    new_lhs[-1] = new_wavs[-1]
    new_lhs[-1] += (new_wavs[-1] - new_wavs[-2])/2
    new_lhs[1:-1] = (new_wavs[1:] + new_wavs[:-1])/2
    new_widths[:-1] = new_lhs[1:-1] - new_lhs[:-2]

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_lhs[j] < old_lhs[0]) or (new_lhs[j+1] > old_max_wav):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill

            if j == 0:
                print("\nSpectres: new_wavs contains values outside the range "
                      "in spec_wavs. New_fluxes and new_errs will be filled "
                      "with the value set in the 'fill' keyword argument (nan "
                      "by default).\n")
            continue

        # Find first old bin which is partially covered by the new bin
        while old_lhs[start+1] <= new_lhs[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_lhs[stop+1] < new_lhs[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_lhs[start+1] - new_lhs[j])
                            / (old_lhs[start+1] - old_lhs[start]))

            end_factor = ((new_lhs[j+1] - old_lhs[stop])
                          / (old_lhs[stop+1] - old_lhs[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

            if old_errs is not None:
                e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

                new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        return new_fluxes

 #============================================================================== 
       