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
def load_testdata(struct):

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
   lmin  = np.log(struct['lmin'])
   lmax  = np.log(struct['lmax'])

   # Correcting the data for redshift
   # NOTE: remember that testdata is already in log
#    print(" - Correcting data for redshift")
#    wave -= (1.0 + struct['redshift'])
   
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
   flux  = np.median(spec)
   npix_log = npix
   nbins = nspec

   print(" - Normalising the spectra")
   for i in trange(nbins, ascii=True, leave=False):
       
      # Normalizing the observed and error spectra respecting the SNR of each bin
      espec[:,i] /= np.nanmedian(spec[:,i])
      spec[:,i]  /= np.nanmedian(spec[:,i]) 

   # Defining the data mask
   print(" - Defining the data mask")
   if (struct['mask_file'] == "None"):
       mn   = np.int(0.01*npix_log) # Masking edges only
       mask = np.arange(mn, npix_log-mn)
   else:
       if not os.path.exists("../config_files/"+struct['mask_file']):
           misc.printFAILED("Cannot find mask file in 'config_files' directory")
           sys.exit()
       mask = misc.spectralMasking("../config_files/"+struct['mask_file'], wave, struct['redshift'])
      
   # Storing all the info in a data structure
   print(" - Storing everything in data structure")
   print("")
   data_struct = {'binID':      np.arange(nbins),
                  'x':          x,
                  'y':          y,
                  'flux':       flux,
                  'xbin':       x,
                  'ybin':       y,
                  'bin_flux':   flux,
                  'spec_obs':   spec,
                  'sigma_obs':  espec,
                  'wave_obs':   wave,
                  'wave':       wave,
                  'velscale':   struct['velscale'],
                  'mask':       np.ravel(mask),
                  'nmask':      len(mask), 
                  'bin_snr':    0,
                  'npix':       npix,
                  'npix_obs':   npix_log,
                  'nspec':      nspec,
                  'porder':     struct['porder'],
                  'nbins':      nbins,
                  'snr':        struct['snr'],
                  'lmin':       np.exp(lmin),
                  'lmax':       np.exp(lmax),
                  'ndim':       ndim
                 }

   return data_struct

       
