import os
import sys
import toml
import importlib.util
import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
import lib.cap_utils      as cap
from   tqdm.auto          import trange
#===============================================================================
def load_data(struct):

   # Adding the relative path to input filename and check file exists
   if not os.path.exists("../data/"+struct['filename']):
       misc.printFAILED("File '"+struct['filename']+"' not found in 'data' directory")
       sys.exit()
   struct['filename'] = "../data/"+struct['filename']
   
   # Reading instruments config file
   instr_config = toml.load("../config_files/instruments.properties")
   instr_list   = list(instr_config.keys())

   if struct['instrument'] not in instr_list:
       misc.printFAILED("Instrument '"+struct['instrument']+"' not found in instruments configuration file")
       sys.exit()
   if not os.path.exists("../config_files/instruments/"+instr_config[struct['instrument']]['read_file']):
       misc.printFAILED("Instrument read file '"+instr_config[struct['instrument']]['read_file']+"' not found in instruments directory")
       sys.exit()

   # Reading instrument specific data and info
   print(" - Reading the data and basic info")
   instr  = importlib.util.spec_from_file_location("", "../config_files/instruments/"+instr_config[struct['instrument']]['read_file'])
   module = importlib.util.module_from_spec(instr)
   instr.loader.exec_module(module)
   data   = module.read_data("../data/"+struct['filename'])

   # Creating variables for convenience
   wave  = data['wave']
   spec  = data['spec']
   espec = data['espec']
   x     = data['x']
   y     = data['y']
   npix  = data['npix']
   nspec = data['nspax']
   psize = data['psize']
   ndim  = data['ndim']
   lmin  = struct['lmin']
   lmax  = struct['lmax']

   # Correcting the data for redshift
   print(" - Correcting data for redshift")
   wave /= (1.0 + struct['redshift'])
   
   # Checking the desired wavelength range is within data wavelength limits
   if (wave[0] > lmin):
       lmin = wave[0]
   if (wave[-1] < lmax):
       lmax = wave[-1]

   # Cutting the data to the desired wavelength range
   print(" - Cutting data to desired wavelength range")
   idx   = (wave >= lmin) & (wave <= lmax)
   wave  = wave[idx]
   spec  = spec[idx,:]
   espec = espec[idx,:]
   npix  = np.sum(idx)

   # Computing the SNR in each spaxel
   print(" - Computing the SNR of each spaxel")
   signal = np.nanmedian(spec,axis=0)
   noise  = np.abs(np.nanmedian(espec,axis=0)) 

   # Filtering out those spectra with NaN estimates for SNR
   good = np.isfinite(signal/noise) & (signal/noise > 0.0)
   if np.sum(good) > 0:
       signal = signal[good]
       noise  = noise[good]
       spec   = spec[:,good]
       espec  = espec[:,good]
       x      = x[good]
       y      = y[good]
       nspec  = np.sum(good)

   # Selecting those spaxels above SNR_min
   print(" - Selecting spaxels above SNR_min")
   delta  = np.abs((signal/noise)-struct['snr_min'])
   idx    = (delta <= 3.0)
   if np.sum(idx) > 0:
       isof   = np.mean(signal[idx])
       idx    = (signal >= isof)
       spec   = spec[:,idx]
       espec  = espec[:,idx]   
       signal = signal[idx]
       noise  = noise[idx]
       x, y   = x[idx], y[idx]
       nspec  = np.sum(idx)

   # IF requested, Voronoi binning the data
   if struct['snr'] > 0.0:

       # Determining Voronoi binning to the data
       print(" - Computing the Voronoi binning")
       binNum, xbin, ybin, xBar, yBar, bin_snr, nPixels, scale = cap.voronoi_2d_binning(x, y, \
               signal, noise, struct['snr'], plot=False, quiet=True, pixelsize=psize)
          
       print("   - "+str(len(xbin))+" Voronoi bins created")   
          
       # Applying the Voronoi binning to the data
       print("   - Applying the Voronoi binning")
       ubins     = np.unique(binNum)
       nbins     = len(ubins)
       bin_spec  = np.zeros([npix,nbins])
       bin_espec = np.zeros([npix,nbins])
       bin_flux  = np.zeros(nbins)
    
       for i in trange(nbins, ascii=True, leave=False):
           k = np.where( binNum == ubins[i] )[0]
           valbin = len(k)
           if valbin == 1:
              av_spec     = spec[:,k]
              av_err_spec = espec[:,k]
           else:
              av_spec     = np.nansum(spec[:,k],axis=1)
              av_err_spec = np.sqrt(np.sum(espec[:,k]**2,axis=1))
    
           bin_flux[i]    = np.mean(av_spec,axis=0)
           bin_spec[:,i]  = np.ravel(av_spec)
           bin_espec[:,i] = np.ravel(av_err_spec)
           
   else:

       bin_snr   = signal/noise
       binNum    = np.arange(nspec)
       bin_flux  = np.mean(spec,axis=0)
       bin_spec  = spec
       bin_espec = espec
       nbins     = nspec
       xbin      = x
       ybin      = y        
       print(" - "+str(len(xbin))+" spectra in file")   
       
   # Log-rebinning the data to the input Velscale
   print(" - Log-rebinning and normalizing the spectra")
   lamRange = np.array([np.amin(wave),np.amax(wave)])
   dummy, lwave, _ = cap.log_rebin(lamRange, bin_spec[:,0], velscale=struct['velscale'])
   npix_log = len(dummy)
   lspec, lespec = np.zeros([npix_log,nbins]), np.zeros([npix_log,nbins])
   for i in trange(nbins, ascii=True, leave=False):
       
      #Log-rebinning the spectra 
      lspec[:,i],  dummy , dummy = cap.log_rebin(lamRange, bin_spec[:,i],  velscale=struct['velscale'])
      lespec[:,i], dummy , dummy = cap.log_rebin(lamRange, bin_espec[:,i], velscale=struct['velscale'])

      # Normalizing the observed and error spectra respecting the SNR of each bin
      lespec[:,i] /= np.nanmedian(lspec[:,i])
      lspec[:,i]  /= np.nanmedian(lspec[:,i]) 

   # Defining the data mask
   print(" - Defining the data mask")
   if (struct['mask_file'] == "None"):
       mn   = np.int(0.01*npix_log) # Masking edges only
       mask = np.arange(mn, npix_log-mn)
   else:
       if not os.path.exists("../config_files/"+struct['mask_file']):
           misc.printFAILED("Cannot find mask file in 'config_files' directory")
           sys.exit()
       mask = misc.spectralMasking("../config_files/"+struct['mask_file'],lwave, struct['redshift'])
      
   # Storing all the info in a data structure
   print(" - Storing everything in data structure")
   print("")
   data_struct = {'binID':      binNum,
                  'x':          x,
                  'y':          y,
                  'flux':       signal,
                  'xbin':       xbin,
                  'ybin':       ybin,
                  'bin_flux':   bin_flux,
                  'spec_obs':   lspec,
                  'sigma_obs':  lespec,
                  'wave_obs':   lwave,
                  'wave':       wave,
                  'velscale':   struct['velscale'],
                  'mask':       np.ravel(mask),
                  'nmask':      len(mask), 
                  'bin_snr':    bin_snr,
                  'npix':       npix,
                  'npix_obs':   npix_log,
                  'nspec':      nspec,
                  'porder':     struct['porder'],
                  'nbins':      nbins,
                  'snr':        struct['snr'],
                  'lmin':       lmin,
                  'lmax':       lmax,
                  'ndim':       ndim
                 }

   return data_struct

       
