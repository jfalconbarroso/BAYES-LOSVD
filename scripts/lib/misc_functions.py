import os
import re
import sys
import glob
import h5py
import toml
import arviz             as az
import numpy             as np
from   astropy.io        import ascii
from   scipy.interpolate import interp1d
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
    codefile   = "models/"+config[fit_type]['codefile']

    # Checking the model exists
    if os.path.exists(codefile):
       codefile = str.split(os.path.basename(codefile),'.py')[0] 
    else:
       printFAILED("Code "+os.path.basename(codefile)+" not found in scripts/models directory.")
       sys.exit()

    return codefile

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

    l = ['filename','instrument','redshift','lmin','lmax','velscale',
         'snr','snr_min','template_lib']

    for key in l:
        if key not in struct.keys():
            printFAILED("keyword '"+key+"' not found in configuration file")
            sys.exit()

    return True

#============================================================================== 
def spectral_masking(preproc_file, maskfile):

    # Reading necessary info from preproc_file
    struct   = h5py.File(preproc_file,"r")
    redshift = np.array(struct['in/redshift'])
    logLam   = np.array(struct['in/wave_obs'])
    struct.close()

    # Define the mask array
    goodPixels  = np.arange(len(logLam))

    if maskfile == None:

        print(" - No maskfile loaded")
    
    else:

        if not os.path.exists(maskfile):
            printFAILED("Cannot find mask file in 'config_files' directory")
            sys.exit()

        # Read file
        mask        = np.genfromtxt(maskfile, usecols=(0,1))
        maskComment = np.genfromtxt(maskfile, usecols=(2), dtype=str )

        print(" - "+maskfile+" loaded")

        # In case there is only one mask
        if len( mask.shape ) == 1  and  mask.shape[0] != 0:
            mask        = mask.reshape(1,2)
            maskComment = maskComment.reshape(1)

        for i in range( mask.shape[0] ):
        
            # Check for sky-lines (correcting for redshift because data has already been de-redshifted)
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

    # Masking edges
    mn = int(0.02*len(logLam)) 
    goodPixels[:mn+1] = -1
    goodPixels[-(mn-1):] = -1

    # Extracting the good values
    goodPixels = goodPixels[ np.where( goodPixels != -1 )[0] ]

    return goodPixels

#============================================================================== 
def create_bins_list(preproc_file, bin):

    # Reading necessary info from preproc_file
    struct = h5py.File(preproc_file,"r")
    nbins = np.array(struct['in/nbins'])
    struct.close()

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
       print("# Selected bins: "+bin)

    nbins = len(bin_list)

    return bin_list, nbins

#============================================================================== 
def create_xvel_vector(vmax=1000.0, velscale=60.0):

    # Creating the LOSVD velocity vector
    xvel = mirror_vector(vmax,inc=velscale)
    if (xvel[1]-xvel[0] < velscale):
        xvel = xvel[1:-1]
    nvel = len(xvel)    

    return xvel, nvel

#============================================================================== 
def pad_templates(struct, nvel):

    # Loading basic info from input structure
    lwave     = struct['lwave_temp']
    templates = struct['templates']
    npix_temp = struct['npix_temp']

    # Padding the templates
    npad      = int(np.floor(nvel/2))
    templates = np.pad(templates, pad_width=((npad,npad+1),(0,0)), mode='edge')
    dwave     = lwave[1]-lwave[0]
    lwave_pre = lwave[0]  - dwave*(np.arange(npad)+1)
    lwave_pos = lwave[-1] + dwave*(np.arange(npad+1)+1)
    lwave_new = np.concatenate((lwave_pre,lwave,lwave_pos))

    # Updating input structure
    new_struct = struct.copy()
    new_struct['lwave_temp']    = lwave_new
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
def check_mcmc_diagnostics(idata, divergence_threshold=0.01, rhat_threshold=1.01, ess_threshold=100):
    """
    Check MCMC diagnostics from an ArviZ InferenceData object.
    
    Args:
        idata: InferenceData from ArviZ / NumPyro
        divergence_threshold: acceptable maximum fraction of divergences
        rhat_threshold: maximum acceptable R-hat
        ess_threshold: minimum effective sample size (bulk ESS)
        outfile: file to store the diagnostics (plain text file)
    """
    # Capture printed output into a list of lines
    output_lines = []

    # Summary table
    summary = az.summary(idata, round_to=4)
    
    output_lines.append("\n==== MCMC Diagnostics ====\n")
    
    # 1. Divergences
    divergences = idata.sample_stats.diverging.values.sum()
    total_samples = idata.sample_stats.diverging.values.size
    frac_divergences = divergences / total_samples
    output_lines.append(f"Divergences: {divergences}/{total_samples} ({100*frac_divergences:.2f}%)\n")

    if frac_divergences > divergence_threshold:
        output_lines.append("WARNING: Too many divergences! Consider reparametrizing or increasing target_accept_prob.\n")
    else:
        output_lines.append("Divergences are under control.\n")

    output_lines.append("\n==========================\n")
    
    # Join all lines into a big text block
    full_output = ''.join(line if line.endswith('\n') else line + '\n' for line in output_lines)

    return        
#==============================================================================
def load_hdf5(filename, verbose=True):

    printRUNNING("Loading "+filename+" data")

    # Checking file exists
    if not os.path.exists(filename):
        printFAILED("Cannot find file "+filename)
        sys.exit()
 
    # Opening file
    f = h5py.File(filename,'r')

    # Defining output dictionary     
    struct = {}

    # Filling up dictionary
    if verbose:
        print("# Loading input data:")

    input_data = f['in']
    for key,values in input_data.items():
        struct[key] = np.array(values)
        if verbose:
            print(' - '+key, struct[key].shape)

    if 'out' in f:

        if verbose:
            print("")
            print("# Loading (non)parametric results:")
    
        output_data = f['out']
        for key,values in output_data.items():
            struct[key] = np.array(values)
            if verbose:
                print(' - '+key, struct[key].shape)

    printDONE()

    return struct

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

#===============================================================================
def pack_results(rootname, suffix='', dir='../results/', **kwargs):

    file_list = sorted(glob.glob(f"{dir}{rootname}_{suffix}/{rootname}_{suffix}_results_bin*.hdf5"))
    nfiles = len(file_list)

    if nfiles == 0:
        print(" - Nothing to pack!")
        print(f"   No {dir}{rootname}_{suffix}/{rootname}_{suffix}_results_bin*.hdf5 found")
        return
    else:
        print(f" - {nfiles} files found.")

    outfile = f"{dir}{rootname}_{suffix}/{rootname}_{suffix}_results.hdf5"
    if os.path.exists(outfile):
        os.remove(outfile)
    g = h5py.File(outfile, 'w')

    print(" - Copying contents to output file")

    # --- Probe first file to initialize
    with h5py.File(file_list[0], 'r') as f:

        # Copying the original in/ group
        f.copy("in", g)
        nbins = np.array(f['in/nbins'])
 
        # Adding any extra arrays as kwargs in this function
        for name, value in kwargs.items():
            g.create_dataset(f'in/{name}', data=value)

        # Read variable names from flat 'out/' group ---------
        var_names = list(f['out'].keys())
        shape_sample = {
            var: f[f'out/{var}'].shape
            for var in var_names
        }

        # Allocate full results arrays
        results = {}
        for var in var_names:
            shape = shape_sample[var]
            full_shape = (nbins,) if shape == () else (nbins, *shape)
            results[var] = np.full(full_shape, np.nan)

        # --- Getting the indices of the bins to store results 
        indices = []
        for fname in file_list:
            match = re.search(r'_results_bin(\d+)\.hdf5', os.path.basename(fname))
            if match:
                indices.append(int(match.group(1)))

        # --- Read each file's 'out/<var>' values
        for i, fname in enumerate(file_list):
            with h5py.File(fname, 'r') as f:
                for var in var_names:
                    results[var][indices[i]] = f[f'out/{var}'][:]

        # --- Write results to new combined file
        for var in var_names:
            g.create_dataset(f'out/{var}', data=results[var])
        
    g.close()

    # Clean up individual files
    if os.path.exists(outfile):
        for filePath in file_list:
            if os.path.exists(filePath):
                os.remove(filePath)

    print(f" - Packed results saved to {outfile}")

    return

#===============================================================================
def save_percentiles_to_hdf5(infilename, idata, outfilename, decomp=False, quantile_values=[0.1, 15.9, 50.0, 84.1, 99.9]):

    # Input data
    inputstruct = h5py.File(infilename, 'r')

    # Extract posterior and stack samples
    posterior = idata.posterior
    posterior_stacked = posterior.stack(sample=("chain", "draw"))

    # Compute quantiles
    quant_array = np.array(quantile_values, dtype=float) / 100.0
    quantiles = posterior_stacked.quantile(q=quant_array, dim="sample")

    # Compute mean and std
    means = posterior_stacked.mean(dim="sample")
    stds  = posterior_stacked.std(dim="sample")

    # Open output HDF5
    with h5py.File(outfilename, 'w') as f:

        # Copying in/ data
        inputstruct.copy('in', f)

        # Copying out/ data if exists, and if not create out group
        if 'out' in inputstruct:
            inputstruct.copy('out', f)
            out_group = f['out']
        else:
            out_group = f.create_group('out')
    
        if decomp:

            # Creating out_decomp/ group and store results
            out_group = f.create_group('out_decomp')

            for var_name in quantiles.data_vars:
                q_vals   = quantiles[var_name].values  # shape: (n_percentiles, ...)
                mean_val = means[var_name].values      # shape: (...)
                std_val  = stds[var_name].values       # shape: (...)

                # Append mean and std as two extra "quantiles"
                full_data = np.concatenate([q_vals, [mean_val], [std_val]], axis=0)
                out_group.create_dataset(var_name, data=full_data)

        else:

            for var_name in quantiles.data_vars:
                q_vals   = quantiles[var_name].values  # shape: (n_percentiles, ...)
                mean_val = means[var_name].values      # shape: (...)
                std_val  = stds[var_name].values       # shape: (...)

                # Append mean and std as two extra "quantiles"
                full_data = np.concatenate([q_vals, [mean_val], [std_val]], axis=0)
                out_group.create_dataset(var_name, data=full_data)

    inputstruct.close()

    return

#===============================================================================
def none_or_str(value):
    return None if value.lower() == "none" else value

#===============================================================================
def parse_kv_pairs(s):
    """Parse comma-separated key=value pairs into a dict."""
    if not s:  # handles None or empty string
        return {}

    d = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue

        if "=" not in pair:
            # skip malformed entries
            continue

        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()

        # Try to infer data types
        if value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        else:
            try:
                # Try numeric conversion
                value = int(value) if value.isdigit() else float(value)
            except ValueError:
                pass  # leave as string if not numeric

        d[key] = value

    return d
