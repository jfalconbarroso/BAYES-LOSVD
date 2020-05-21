import h5py
import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
import lib.cap_utils      as cap
#===============================================================================
def load_testdata(struct,idx):

   # Reading relevant info from config file
   rname      = struct['Runname'][idx]
   rootname   = rname.split('-')[0]
   filename   = rootname+'.hdf5'
   lmin       = struct['Lmin'][idx]
   lmax       = struct['Lmax'][idx]
   porder     = struct['Porder'][idx]
   vmax       = struct['Vmax'][idx]
   mask_width = struct['Mask_width'][idx]

   # Opening the file with testdata
   print(" - Reading the testdata file and basic info")
   f        = h5py.File('../data/'+filename,'r')
   bin_snr  = np.array(f['SNR'])
   velscale = np.array(f['VELSCALE'])
   lspec    = np.array(f['SPEC'])
   lespec   = np.array(f['ESPEC'])
   lwave    = np.array(f['WAVE'])
   npix, nspec = lspec.shape
   nbins = nspec
   
   # Checking the desired wavelength range is within data wavelength limits
   if (np.exp(lwave[0]) > lmin):
       lmin = np.exp(lwave[0])
   if (np.exp(lwave[-1]) < lmax):
       lmax = np.exp(lwave[-1])

   # Cutting the data to the desired wavelength range
   print(" - Cutting data to desired wavelength range")
   idx    = (np.exp(lwave) >= lmin) & (np.exp(lwave) <= lmax)
   lwave  = lwave[idx]
   lspec  = lspec[idx,:]
   lespec = lespec[idx,:]
   npix   = np.sum(idx)
      
   # Defining the mask
   print(" - Defining the data mask")
   mask = cap.determine_goodpixels(lwave,[lmin,lmax],0.0)
   mask = np.arange(mask[0],mask[-1])
      
   # Storing all the info in a data structure
   print(" - Storing everything in data structure")
   print("")
   data_struct = {'binID':     np.arange(nspec),
                  'x':         np.zeros(1),
                  'y':         np.zeros(1),
                  'flux':      np.ones(1)*np.mean(lspec,axis=1),
                  'xbin':      np.zeros(1),
                  'ybin':      np.zeros(1),
                  'bin_flux':  np.ones(1)*np.mean(lspec,axis=1),
                  'spec_obs':  lspec,
                  'sigma_obs': lespec,
                  'wave_obs':  lwave,
                  'velscale':  velscale,
                  'mask':      np.ravel(mask),
                  'nmask':     len(mask),
                  'mask_width': mask_width,
                  'bin_snr':   bin_snr,
                  'npix':      npix,
                  'npix_obs':  npix,
                  'nspec':     nspec,
                  'porder':    porder,
                  'nbins':     nbins,
                  'snr':       bin_snr,
                  'lmin':      lmin,
                  'lmax':      lmax
                 }

   return data_struct

       
