import os
import sys
import h5py
import optparse
import warnings
import numpy              as np
import lib.misc_functions as misc
#==============================================================================
def load_hdf5(filename, verbose=True):

    misc.printRUNNING("Loading "+filename+" data")

    # Checking file exists
    if not os.path.exists(filename):
        misc.printFAILED("Cannot find file "+filename)
        sys.exit()
 
    # Opening file
    if verbose:
        print("# Opening file")
        print("")
    f = h5py.File(filename,'r')

    # Defining output dictionary     
    struct = {}

    # Filling up dictionary
    if verbose:
        print("# Loading input data:")
    input_data = f['in']
    for key,values in input_data.items():
        if verbose:
            print(' - '+key)
        struct[key] = np.array(values)

    if f.get("out") != None:
        if verbose:
            print("")
            print("# Loading Stan results:")
        output_data = f['out']
        bins_list   = list(output_data.keys())
        for idx in bins_list:
            tmp = f['out/'+idx]
            struct[int(idx)] = {}
            for key,values in tmp.items():
                if verbose:
                    print(' - ['+idx+'] '+key)
                struct[int(idx)][key] = np.array(values)

    misc.printDONE()

    return struct
#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("               BAYES-LOSVD                 ")
    print("             (load results)                ")
    print("===========================================")
    print("")

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -f file")
    parser.add_option("-f", "--filename",  dest="filename", type="string", default=None, help="Filename of the HDF5 file with results")

    (options, args) = parser.parse_args()
    filename = options.filename

    tab = load_hdf5(filename)

    print(tab[0]['losvd'].shape)
