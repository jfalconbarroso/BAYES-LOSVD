import os
import sys
import glob
import h5py
import toml
import numpy                 as np
import matplotlib.pyplot     as plt
import lib.misc_functions    as misc
import lib.cap_utils         as cap
from   astropy.io            import fits
from   sklearn.decomposition import PCA
from   scipy.interpolate     import interp1d
from   tqdm.auto             import trange
#==============================================================================
def load_templates(struct,data_struct):

   # Reading relevant info from config file
   temp_name = struct['template_lib']
   velscale  = struct['velscale']
   npca      = struct['npca']
   instr     = struct['instrument']
   redshift  = struct['redshift']
   vmax      = struct['vmax']
   lmin      = data_struct['lmin']
   lmax      = data_struct['lmax']

   # Getting the appropiate LSF files
   instr_config  = toml.load("../config_files/instruments.properties")
   lsf_data_file = "../config_files/instruments/"+instr_config[instr]['lsf_file']
   lsf_temp_file = "../config_files/instruments/"+temp_name+'.lsf' 
   if not os.path.exists(lsf_data_file):
       misc.printFAILED("Data lsf file not found in 'config_files/instruments' directory")
       sys.exit()
   if not os.path.exists(lsf_temp_file):
       misc.printFAILED("Templates lsf file not found in 'config_files/instruments' directory")
       sys.exit()

   # Loading SSP models and defining some basic parameters–
   list  = glob.glob("../templates/"+temp_name+"/*")
   ntemp = len(list)
   print(" - "+str(ntemp)+" templates found in "+temp_name+" library")

   hdu  = fits.open(list[0])
   tmp  = np.ravel(hdu[0].data)
   hdr  = hdu[0].header
   wave = hdr['CRVAL1']+np.arange(len(tmp))*hdr['CDELT1']
   dwav = hdr['CDELT1']
   npix = len(wave)
   
   # Defining output arrays
   temp  = np.zeros((npix,ntemp))
   scale = np.zeros(ntemp)
    
   # Loading templates into final arrays
   # NOTE: this loops already cuts the spectra to the Lmin,Lmax limits
   print(" - Loading and preparing the templates...")
   for i in trange(ntemp, ascii=True, leave=False):
        
       # Reading, trimming and scaling the spectra 
       hdu        = fits.open(list[i])
       temp[:,i]  = np.ravel(hdu[0].data)
       scale[i]   = np.mean(temp[:,i])
       temp[:,i] /= scale[i]       
            
   # Running PCA on the input models
   if npix < ntemp:
      misc.printFAILED("The number of pixels in the spectra ("+str(npix)+") has to be larger than the number of templates ("+str(ntemp)+") to run PCA.")
      sys.exit()

   if npca > 0:
       print(" - Running PCA on the templates...")
       mean_temp = np.mean(temp,axis=1)
       pca       = PCA(n_components=ntemp)
       PC_tmp    = pca.fit_transform(temp)
      
       # Extracting the desired number of PCA components
       cumsum_pca_variance = np.cumsum(pca.explained_variance_ratio_)
       print("    "+str(npca)+" PCA components explain {:7.3f}".format(cumsum_pca_variance[npca]*100)+"% of the variance in the input library")
       templates = np.zeros((npix,npca))
       templates = PC_tmp[:,0:npca]
       ntemplates = npca

       # Continuum and Z-score Normalization to aid in the minimization
       for i in range(npca):
          coef = np.polyfit(wave,templates[:,i],1)
          pfit = np.polyval(coef,wave)
          templates[:,i] -= pfit
          templates[:,i] /= np.std(templates[:,i])
   else:
       mean_temp  = np.zeros(npix)
       templates  = temp
       ntemplates = ntemp

   # Convolving the templates to match the data's LSF
   print(" - Convolving the templates to match the data's LSF")
   data_lsf   = misc.read_lsf(wave, lsf_data_file)
   data_lsf  /= (1.0 + redshift) 
   temp_lsf   = misc.read_lsf(wave, lsf_temp_file)
   fwhm_diff  = np.sqrt(data_lsf**2 - temp_lsf**2)  # in angstroms
   bad_pix    = np.isnan(fwhm_diff)
   if np.sum(bad_pix) > 0:
       misc.printWARNING("Some values of the data LSF are below the templates values")
   fwhm_diff[bad_pix] = 1E-2  # Fixing the FWHM_diff to a tiny value if there are NaNs
   sigma_diff = fwhm_diff/2.355/dwav

   mean_temp = cap.gaussian_filter1d(mean_temp,sigma_diff)
   for i in trange(ntemplates, ascii=True, leave=False):
      templates[:,i] = cap.gaussian_filter1d(templates[:,i], sigma_diff)  # convolution with variable sigma
   
   # Log-rebinning the PCA spectra using the data's velscale
   print(" - Log-rebinning the templates")
   lamRange = np.array([np.amin(wave),np.amax(wave)])
   mean_temp, lwave, dummy = cap.log_rebin(lamRange, mean_temp, velscale=velscale)
   npix_temp = mean_temp.shape[0]
   tmp_temp  = np.zeros((npix_temp,ntemplates))
   for i in range(ntemplates):
       tmp_temp[:,i], dummy, dummy = cap.log_rebin(lamRange, templates[:,i], velscale=velscale)
   templates = tmp_temp

   # Checking the wavelength solution for the templates is identical to the data
   # If not, the templates are resampled
   # NOTE: this is important to have a centered LOSVD on xvel=0.0
   good  = (lwave >= np.log(data_struct['lmin'])) & (lwave <= np.log(data_struct['lmax']))
   check = np.array_equal(lwave[good], data_struct['wave_obs'])
   if check == False:
       print(" - Resampling the templates to match the wavelength of the observed data (if needed)")
       mean_temp = misc.spectres(data_struct['wave_obs'], lwave, mean_temp, fill=np.nan)
       npix_temp = len(mean_temp)
       new_temp  = np.zeros((npix_temp,ntemplates))
       for i in range(ntemplates):
           new_temp[:,i] = misc.spectres(data_struct['wave_obs'], lwave, templates[:,i], fill=np.nan)
       lwave     = data_struct['wave_obs']
       templates = new_temp
   else:
       mean_temp = mean_temp[good]
       templates = templates[good,:]
       lwave     = lwave[good]
       npix_temp = len(lwave)

   # Normalizing the mean template to 1.0 and adjusting the other templates so that the mean is around 0.0
   mean_temp /= np.mean(mean_temp)
   for i in range(ntemplates):
       templates[:,i] -= np.mean(templates[:,i])
 
   # Storing everything into a dictionary
   print(" - Storing everything in templates structure")
   struct = {'lwave_temp':    lwave,
             'mean_template': mean_temp,
             'templates':     templates,
             'npix_temp':     npix_temp,
             'ntemp':         ntemplates
            }
   
   return struct

