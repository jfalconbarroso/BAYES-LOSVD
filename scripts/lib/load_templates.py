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

   # Creating the LOSVD velocity vector
   print(" - Creating the LOSVD velocity vector")
   xvel = misc.mirror_vector(vmax,inc=velscale)
   if (xvel[1]-xvel[0] < velscale):
       xvel = xvel[1:-1]
   xvel = np.flip(xvel) # The flipping is necessary because of the way the convolution is done
   nvel = len(xvel)    

   # Loading SSP models and defining some basic parameters–
   list  = glob.glob("../templates/"+temp_name+"/*")
   ntemp = len(list)
   print(" - "+str(ntemp)+" templates found in "+temp_name+" library")

   hdu  = fits.open(list[0])
   tmp  = np.ravel(hdu[0].data)
   hdr  = hdu[0].header
   wave = hdr['CRVAL1']+np.arange(len(tmp))*hdr['CDELT1']
   dwav = hdr['CDELT1']
   
   # Defining pixels to cut in wavelength as the input +-50A either side
   # This avoids border effects when convolving the templates to the data LSF
   # and for the padding of the spectra to do the convolution in Stan
   idx  = (wave >= lmin-100.0) & (wave <= lmax+100.0) 
   wave = wave[idx]
   npix = np.sum(idx)
   
   # Defining output arrays
   temp  = np.zeros((npix,ntemp))
   scale = np.zeros(ntemp)
    
   # Loading templates into final arrays
   # NOTE: this loops already cuts the spectra to the Lmin,Lmax limits
   print(" - Loading and preparing the templates...")
   for i in range(ntemp):
        
       # Reading, trimming and scaling the spectra 
       hdu        = fits.open(list[i])
       tmp        = np.ravel(hdu[0].data)
       temp[:,i]  = tmp[idx]
       scale[i]   = np.mean(temp[:,i])
       temp[:,i] /= scale[i]       
       
       misc.printProgress(i+1, ntemp, suffix = 'Complete', barLength = 50) 
     
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
       print("  "+str(npca)+" PCA components explain {:7.3f}".format(cumsum_pca_variance[npca]*100)+"% of the variance in the input library")
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
   for i in range(ntemplates):
      templates[:,i] = cap.gaussian_filter1d(templates[:,i], sigma_diff)  # convolution with variable sigma
      misc.printProgress(i+1, ntemplates, prefix = '   Progress:', suffix = 'Complete', barLength = 50) 
   

   # Log-rebinning the PCA spectra using the data's velscale
   print(" - Log-rebinning the templates")
   lamRange = np.array([np.amin(wave),np.amax(wave)])
   mean_temp, lwave, dummy = cap.log_rebin(lamRange, mean_temp, velscale=velscale)
   npix    = mean_temp.shape[0]
   tmp_temp = np.zeros((npix,ntemplates))
   for i in range(ntemplates):
       tmp_temp[:,i], dummy, dummy = cap.log_rebin(lamRange, templates[:,i], velscale=velscale) 
   templates = tmp_temp

   # Cutting the templates so that only nvel/2 pixels are below the Lmin wavelength
   # This is the equivalent of zero-padding the spectra to center the LOSVD in the
   # middle of the kernel range
   pad    = int(np.floor(nvel/2))
   lwave0 = np.log(lmin)-pad*(lwave[1]-lwave[0])
   lwave1 = np.log(lmax)+(pad-1)*(lwave[1]-lwave[0])
   if (lwave[0] > lwave0):
      misc.printFAILED("Templates wavelength range is not sufficient for padding")
      sys.exit()

   idx        = (lwave >= lwave0) & (lwave <= lwave1)
   mean_temp  = mean_temp[idx]
   templates  = templates[idx,:]
   lwave      = lwave[idx]
   npix_temp  = len(lwave)
   
   diff = npix_temp-(data_struct['npix_obs']+nvel-1)
   if (diff == -1):       
       mean_temp  = np.pad(mean_temp, pad_width=(0,1),         mode='edge')
       templates = np.pad(templates,pad_width=((0,1),(0,0)), mode='edge')
       lwave      = np.pad(lwave, pad_width=(0,1), mode='constant', constant_values=lwave[-1]+(lwave[1]-lwave[0]))
       npix_temp  = len(lwave)       
   elif (diff == 1):
       mean_temp  = mean_temp[0:-2]
       templates  = templates[0:-2,:]
       lwave      = lwave[0:-2]
       npix_temp  = len(lwave)
             
   print(" - Storing everything in templates structure")
   struct = {'lwave_temp':    lwave,
             'mean_template': mean_temp,
             'templates':     templates,
             'npix_temp':     npix_temp,
             'ntemp':         ntemplates,
             'nvel':          nvel,
             'xvel':          xvel
            }
   
   return struct

