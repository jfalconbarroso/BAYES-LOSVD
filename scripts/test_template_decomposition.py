import glob
import h5py
import numpy                 as np
import matplotlib.pyplot     as plt
import lib.misc_functions    as misc
import lib.cap_utils         as cap
from   astropy.io            import fits
from   sklearn.decomposition import PCA, FastICA
from   scipy.interpolate     import interp1d
#==============================================================================
if (__name__ == '__main__'):

    ntemp = 5

    f    = h5py.File("../templates/miles_test.hdf5","r")
    temp = np.array(f['spec'])

    print(" - Running PCA on the templates...")
    mean_temp = np.mean(temp,axis=1)
    pca       = PCA(n_components=ntemp)
    PC_tmp    = pca.fit_transform(temp)

    print(" - Running ICA on the templates...")
    ica = FastICA(n_components=ntemp)
    S_ = ica.fit_transform(temp)  # Reconstruct signals

    fig, ax = plt.subplots(nrows=2, ncols=1)

    ax[0].plot(PC_tmp)
    ax[1].plot(S_)
    plt.show()

