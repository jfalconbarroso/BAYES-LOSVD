import glob
import numpy                 as np
import matplotlib.pyplot     as plt
import lib.misc_functions    as misc
import lib.cap_utils         as cap
from   astropy.io            import fits
from   sklearn.decomposition import PCA
from   scipy.interpolate     import interp1d
#==============================================================================
def load_templates(struct,idx,data_struct):

   # Reading relevant info from config file
   temp_name = struct['Templates'][idx]
   velscale  = struct['Velscale'][idx]
   npca      = struct['NPCA'][idx]
   survey    = struct['Survey'][idx]
   redshift  = struct['Redshift'][idx]
   vmax      = struct['Vmax'][idx]
   lmin      = data_struct['lmin']
   lmax      = data_struct['lmax']

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
   tmp  = hdu[0].data
   hdr  = hdu[0].header
   wave = hdr['CRVAL1']+np.arange(len(tmp))*hdr['CDELT1']
   dwav = wave[1]-wave[0]
   
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
       tmp        = hdu[0].data
       temp[:,i]  = tmp[idx]
       scale[i]   = np.mean(temp[:,i])
       temp[:,i] /= scale[i]       
       
       misc.printProgress(i+1, ntemp, suffix = 'Complete', barLength = 50) 
     
   # Running PCA on the input models
   print(" - Running PCA on the templates...")
   mean_temp = np.mean(temp,axis=1)
   pca       = PCA(n_components=ntemp)
   PC_tmp    = pca.fit_transform(temp)

   # Extracting the desired number of PCA components
   cumsum_pca_variance = np.cumsum(pca.explained_variance_ratio_)
   print("  "+str(npca)+" PCA components explain {:7.3f}".format(cumsum_pca_variance[npca]*100)+"% of the variance in the input library")
   pca_models = np.zeros((npix,npca))
   pca_models = PC_tmp[:,0:npca]

   # Z-score Normalization to aid in the minimization
   for i in range(npca):
      pca_models[:,i] /= np.std(pca_models[:,i])


   #fig, ax = plt.subplots(nrows=7,ncols=1, sharex=True, sharey=False, figsize=(8,8))    
   #ax = ax.ravel()
   #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
   #col = ['blue','orange','green','red','magenta','brown']

   #ax[0].plot(wave,mean_temp/np.std(mean_temp), color='black', label='Mean spectrum')
   #ax[0].legend()
   #for i in range(6):
       #ax[i+1].plot(wave,pca_models[:,i],color=col[i], label='PC'+str(i+1))
       #ax[i+1].set_xlim([4750.0,5450.0])
       #ax[i+1].set_ylim([-3.5,5.9])
       #ax[i+1].legend()
   #ax[6].set_xlabel("Wavelength ($\mathrm{\AA}$)")    
   #plt.show()
   #exit()
   
   # Convolving the templates to match the data's LSF
   if not (survey == 'TEST'):
      print(" - Convolving the templates to match the data's LSF")
      data_lsf   = misc.read_lsf(wave, survey)
      data_lsf  /= (1.0 + redshift) 
      temp_lsf   = misc.read_lsf(wave, temp_name)
      fwhm_diff  = np.sqrt(data_lsf**2 - temp_lsf**2)  # in angstroms
      sigma_diff = fwhm_diff/2.355/dwav
  
      mean_temp = cap.gaussian_filter1d(mean_temp,sigma_diff)
      for i in range(npca):
         pca_models[:,i] = cap.gaussian_filter1d(pca_models[:,i], sigma_diff)  # convolution with variable sigma
         misc.printProgress(i+1, npca, prefix = '   Progress:', suffix = 'Complete', barLength = 50) 
   

   # Log-rebinning the PCA spectra using the data's velscale
   print(" - Log-rebinning the templates")
   lamRange = np.array([np.amin(wave),np.amax(wave)])
   mean_temp, lwave, dummy = cap.log_rebin(lamRange, mean_temp, velscale=velscale)
   npix    = mean_temp.shape[0]
   tmp_pca = np.zeros((npix,npca))
   for i in range(npca):
       tmp_pca[:,i], dummy, dummy = cap.log_rebin(lamRange, pca_models[:,i], velscale=velscale) 
   pca_models = tmp_pca

   # Cutting the templates so that only nvel/2 pixels are below the Lmin wavelength
   # This is the equivalent of zero-padding the spectra to center the LOSVD in the
   # middle of the kernel range
   pad    = int(np.floor(nvel/2))
   lwave0 = np.log(lmin)-pad*(lwave[1]-lwave[0])
   lwave1 = np.log(lmax)+(pad-1)*(lwave[1]-lwave[0])
   if (lwave[0] > lwave0):
      print("ERROR: Templates wavelength range is not sufficient for padding")
      exit()
   idx        = (lwave >= lwave0) & (lwave <= lwave1)
   mean_temp  = mean_temp[idx]
   pca_models = pca_models[idx,:]
   lwave      = lwave[idx]
   npix_temp  = len(lwave)
   
   diff = npix_temp-(data_struct['npix_obs']+nvel-1)
   if (diff == -1):       
       mean_temp  = np.pad(mean_temp, pad_width=(0,1),         mode='edge')
       pca_models = np.pad(pca_models,pad_width=((0,1),(0,0)), mode='edge')
       lwave      = np.pad(lwave, pad_width=(0,1), mode='constant', constant_values=lwave[-1]+(lwave[1]-lwave[0]))
       npix_temp  = len(lwave)       
   elif (diff == 1):
       mean_temp  = mean_temp[0:-2]
       pca_models = pca_models[0:-2,:]
       lwave      = lwave[0:-2]
       npix_temp  = len(lwave)
             
   print(" - Storing everything in templates structure")
   struct = {'lwave_temp':    lwave,
             'mean_template': mean_temp,
             'templates':     pca_models,
             'npix_temp':     npix_temp,
             'ntemp':         npca,
             'nvel':          nvel,
             'xvel':          xvel
            }
   
   return struct

