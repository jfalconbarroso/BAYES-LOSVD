import os
import sys
import toml
import importlib.util
import numpy                 as np
import lib.misc_functions    as misc
import lib.cap_utils         as cap
from   tqdm.auto             import trange
from   sklearn.decomposition import PCA
#==============================================================================
def load_templates(struct, data_struct):

    # Reading relevant info from config file
    temp_name = struct['template_lib']
    velscale  = struct['velscale']
    instr     = struct['instrument']
    redshift  = struct['redshift']
    npca      = struct['npca']

    # Checking relevant scripts to load templates and info
    instr_config = toml.load("../config_files/instruments.properties")
    instr_list   = list(instr_config.keys())
    temp_config  = toml.load("../config_files/templates.properties")
    temp_list    = list(temp_config.keys())

    if instr not in instr_list:
        misc.printFAILED(f"Instrument '{instr}' not found in instruments configuration file")
        sys.exit()
    if not os.path.exists("../config_files/instruments/" + instr_config[instr]['read_file']):
        misc.printFAILED(f"Instrument read file '{instr_config[instr]['read_file']}' not found in instruments directory")
        sys.exit()
    if temp_name not in temp_list:
        misc.printFAILED(f"Template '{temp_name}' not found in instruments configuration file")
        sys.exit()
    if not os.path.exists("../config_files/templates/" + temp_config[temp_name]['read_file']):
        misc.printFAILED(f"Templates read file '{temp_config[temp_name]['read_file']}' not found in instruments directory")
        sys.exit()

    # Getting the appropiate LSF files
    lsf_data_file = "../config_files/instruments/" + instr_config[instr]['lsf_file']
    lsf_temp_file = "../config_files/templates/" + temp_config[temp_name]['lsf_file']
    if not os.path.exists(lsf_data_file):
        misc.printFAILED("Data lsf file not found in 'config_files/instruments' directory")
        sys.exit()
    if not os.path.exists(lsf_temp_file):
        misc.printFAILED("Templates lsf file not found in 'config_files/templates' directory")
        sys.exit()

    # Loading SSP models and defining some basic parameters
    print(" - Reading the templates and basic info")
    tname  = importlib.util.spec_from_file_location("", "../config_files/templates/" + temp_config[temp_name]['read_file'])
    module = importlib.util.module_from_spec(tname)
    tname.loader.exec_module(module)

    # User-provided routine returns params (generic) with shape (npar, ntemp)
    wave, temp, ntemp, npix, params = module.read_templates(temp_name)
    dwav = wave[1] - wave[0]

    # Basic sanity
    if temp.shape != (npix, ntemp):
        misc.printFAILED(f"Template matrix shape mismatch. Expected ({npix},{ntemp}), got {temp.shape}.")
        sys.exit()

    # PCA compression (if requested)
    if npca > 0:
        if ntemp > npix:
            misc.printFAILED(
                "PCA (templates as samples) requires ntemp <= npix. "
                f"Got ntemp={ntemp}, npix={npix}."
            )
            sys.exit()

        print(" - Running PCA on the templates (templates as samples)...")

        # Build PCA in wavelength space: templates are samples, pixels are features
        X   = temp.T  # (ntemp, npix)
        pca = PCA(n_components=ntemp)
        eigenspectra = pca.components_   # (ntemp, npix)
        mean_temp    = pca.mean_         # (npix,)

        cumsum_pca_variance = np.cumsum(pca.explained_variance_ratio_)
        print(
            f"    {npca} PCA components explain {cumsum_pca_variance[npca-1]*100:7.3f}% "
            "of the variance in the input library"
        )

        templates = eigenspectra[:npca, :].T      # (npix, npca)
        ntemplates = npca

    else:
        # No PCA: use original templates as-is
        mean_temp = np.zeros(npix, dtype=float)
        templates = temp
        ntemplates = ntemp

    # Convolving the templates to match the data's LSF
    if lsf_data_file != lsf_temp_file:
        print(" - Convolving the templates to match the data's LSF")
        data_lsf = misc.read_lsf(wave, lsf_data_file)
        data_lsf /= (1.0 + redshift)
        temp_lsf = misc.read_lsf(wave, lsf_temp_file)
        fwhm_diff = np.sqrt(data_lsf**2 - temp_lsf**2)  # in angstroms
        bad_pix = np.isnan(fwhm_diff)
        if np.sum(bad_pix) > 0:
            misc.printWARNING("Some values of the data LSF are below the templates values")
        fwhm_diff[bad_pix] = 1e-2
        sigma_diff = fwhm_diff / 2.355 / dwav

        mean_temp = cap.gaussian_filter1d(mean_temp, sigma_diff)
        for i in trange(ntemplates, ascii=True, leave=False):
            templates[:, i] = cap.gaussian_filter1d(templates[:, i], sigma_diff)

    # Log-rebinning the spectra using the data's velscale
    print(" - Log-rebinning the templates")
    lamRange = np.array([np.amin(wave), np.amax(wave)])
    mean_temp, lwave, _ = cap.log_rebin(lamRange, mean_temp, velscale=velscale)
    npix_temp = mean_temp.shape[0]

    tmp_temp = np.zeros((npix_temp, ntemplates))
    for i in trange(ntemplates, ascii=True, leave=False):
        tmp_temp[:, i], _, _ = cap.log_rebin(lamRange, templates[:, i], velscale=velscale)
    templates = tmp_temp

    # Match wavelength solution to observed data grid if needed
    good = (lwave >= np.log(data_struct['lmin'])) & (lwave <= np.log(data_struct['lmax']))
    check = np.array_equal(lwave[good], data_struct['wave_obs'])
    if not check:
        print(" - Resampling the templates to match the wavelength of the observed data")
        mean_temp = misc.spectres(data_struct['wave_obs'], lwave, mean_temp, fill=np.nan)
        npix_temp = len(mean_temp)
        new_temp = np.zeros((npix_temp, ntemplates))
        for i in trange(ntemplates, ascii=True, leave=False):
            new_temp[:, i] = misc.spectres(data_struct['wave_obs'], lwave, templates[:, i], fill=np.nan)
        lwave = data_struct['wave_obs']
        templates = new_temp
    else:
        mean_temp = mean_temp[good]
        templates = templates[good, :]
        lwave = lwave[good]
        npix_temp = len(lwave)

    # Normalisation convention (preserve your original behaviour)
    if npca > 0:
        mean_temp /= np.mean(mean_temp)
        for i in range(ntemplates):
            templates[:, i] -= np.mean(templates[:, i])
    else:
        mean_temp = np.zeros_like(mean_temp)
        for i in range(ntemplates):
            templates[:, i] /= np.mean(templates[:, i])

    # Store everything
    print(" - Storing everything in templates structure")
    out = {
        'lwave_temp': lwave,
        'mean_template': mean_temp,
        'templates': templates,
        'npix_temp': npix_temp,
        'ntemp': ntemp,
        'params': params
    }

    return out
