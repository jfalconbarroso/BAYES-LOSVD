import os
import sys
import glob
import h5py
import arviz                 as az
import numpy                 as np
import matplotlib.pyplot     as plt
from   astropy.io            import fits, ascii
from   astropy.convolution   import convolve
from   scipy.interpolate     import interp1d
#===============================================================================
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s Progress [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")

#==============================================================================
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
def read_lsf(wave,survey):
    
    # Reading the LSF file
    tab = ascii.read("../config_files/"+survey+".lsf")
     
    # Interpolating LSF at input wavelengths
    f   = interp1d(tab['Lambda'],tab['FWHM'])
    out = f(wave)
    
    return out

#==============================================================================
def load_configfile(file=''):

    colnames = ["Runname","Survey","Redshift","Lmin","Lmax","Vmax","Velscale","SNR","SNR_min","Porder","Border","Templates","NPCA","Mask","Mask_width"]

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
def save_stan_summary(fit, unwanted=None, summary_filename=None, verbose=None):

    samples   = fit.extract(permuted=True) # Extracting parameter samples
    var_all   = list(samples.keys())
    var_names = [e for e in var_all if e not in unwanted]
    stan_summary = fit.stansummary(pars=var_names)
    if (verbose == True):
        print("")
        print(stan_summary)

    if os.path.exists(summary_filename):
       os.remove(summary_filename)
    f = open(summary_filename, 'w')
    f.write(stan_summary)
    f.close()       

    return
#==============================================================================
def save_stan_chains(samples, chains_filename=None):

    if os.path.exists(chains_filename):
       os.remove(chains_filename)

    f = h5py.File(chains_filename, "w")
    for key, val in samples.items():
        f.create_dataset(key, data=val, compression="gzip")
    f.close()

    return
#==============================================================================
def process_stan_output_per(struct, samples, outhdf5=None, stridx=None):

    lims = [0.1,15.9,50.0,84.1,99.9] # Saving 1-sigma, 3-sigma, and median

    if os.path.exists(outhdf5):
       os.remove(outhdf5)

    f = h5py.File(outhdf5, "w")
    #------------
    struct.copy('in',f)

    # If structure already contains results from a previous step, copy results also
    if not (struct.get('out') == None):
       struct.copy('out',f)
    #------------
    for key in samples.keys():
        if np.ndim(samples[key]) == 1:
           result = np.percentile(np.array(samples[key]), q=lims)
        elif np.ndim(samples[key]) > 1:
           result = np.percentile(np.array(samples[key]).T, q=lims, axis=1)

        if not (key == 'lp__'):
           f.create_dataset("out/"+stridx+"/"+key, data=result, compression="gzip")
    #------------
    f.close()

    return

#==============================================================================
def process_stan_output_hdp(struct, samples, outhdf5=None, stridx=None):

    lims = [0.50, 0.68, 0.999] # Saving 1-sigma, 3-sigma, and median

    if os.path.exists(outhdf5):
       os.remove(outhdf5)

    f = h5py.File(outhdf5, "w")
    #------------
    struct.copy('in',f)

    # If structure already contains results from a previous step, copy results also
    if not (struct.get('out') == None):
       struct.copy('out',f)
    #------------
    for key in samples.keys():
        result = compute_hdp(np.array(samples[key]), lims)
        if not (key == 'lp__'):
           f.create_dataset("out/"+stridx+"/"+key, data=result, compression="gzip")
    #------------
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
           kk = az.hpd(samples, credible_interval=lims[i])
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
               kk = az.hpd(samples[:,j], credible_interval=lims[i])
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
def pack_results(rootname,suffix=''):

    input_list = glob.glob("../results/"+rootname+"/"+rootname+suffix+"_results_bin*.hdf5")
    nlist = len(input_list)
    if nlist == 0:
       print(" - Nothing to pack!")
       print("")
       print("../results/"+rootname+"/"+rootname+suffix+"_results_bin*.hdf5")
       return 
    else:   
       print(" - "+str(nlist)+" files found.")

    # Preparing outputfile
    outfile = "../results/"+rootname+"/"+rootname+suffix+"_results.hdf5"
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
