import h5py
import numpy              as np
#===============================================================================
def read_data(filename):

    hdu = h5py.File(filename,'r')
    #---------------------------
    spec  = np.array(hdu['SPEC'])
    espec = np.array(hdu['ESPEC'])
    wave  = np.array(hdu['WAVE'])
    #---------------------------
    x     = np.arange(spec.shape[1])
    y     = np.zeros(spec.shape[1])
    psize = 1.0
    npix  = spec.shape[0]
    nspax = spec.shape[1]

    struct = {'wave':wave, 'spec':spec, 'espec':espec, 'x':x, 'y':y, 'npix':npix, 'nspax':nspax, 'psize':psize, 'ndim':1}

    return struct       