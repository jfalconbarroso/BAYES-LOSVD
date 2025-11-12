import os
import sys
import h5py
import argparse
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
            print("# Loading model results:")
        output_data = f['out']
        for key,values in output_data.items():
            if verbose:
                print(' - '+key)
            struct[key] = np.array(values)

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

    parser = argparse.ArgumentParser(
        prog="myscript.py",
        usage="%(prog)s -f file [options]",
        description="Process spectra with configurable parameters."
    )

    parser.add_argument("-f", "--filename", type=str, default=None, help="File with the results")

    tab = load_hdf5(filename)
