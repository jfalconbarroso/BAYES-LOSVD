import h5py
import numpy as np
#==============================================================================
def load_preproc_data(filename=''):
    
    
    f = h5py.File(filename)
    
    tab = {'binID':         np.array(f['binID']),
           'x':             np.array(f['x']),
           'y':             np.array(f['y']),
           'flux':          np.array(f['flux']),
           'xbin':          np.array(f['xbin']),
           'ybin':          np.array(f['ybin']),
           'bin_flux':      np.array(f['bin_flux']),
           'spec_obs':      np.array(f['spec_obs']),
           'sigma_obs':     np.array(f['sigma_obs']),
           'wave_obs':      np.array(f['wave_obs']),
           'velscale':      np.array(f['velscale']),
           'mask':          np.ravel(np.array(f['mask'])),
           'nmask':         np.array(f['nmask']),
           'mask_width':    np.array(f['mask_width']),
           'xvel':          np.array(f['xvel']),
           'nvel':          np.array(f['nvel']),
           'npix':          np.array(f['npix']),
           'npix_obs':      np.array(f['npix_obs']),
           'porder':        np.array(f['porder']),
           'border':        np.array(f['border']),
           'lwave_temp':    np.array(f['lwave_temp']),
           'mean_template': np.array(f['mean_template']),
           'templates':     np.array(f['templates']),
           'npix_temp':     np.array(f['npix_temp']),
           'ntemp':         np.array(f['ntemp']),
           'nspec':         np.array(f['nspec']),
           'nbins':         np.array(f['nbins'])
           }
    
    return tab
