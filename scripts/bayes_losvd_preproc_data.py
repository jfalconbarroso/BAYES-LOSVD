import os
import h5py
import warnings
import argparse
import toml
import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
from   lib.load_data      import load_data
from   lib.load_testdata  import load_testdata
from   lib.load_templates import load_templates
from   lib.cap_utils      import display_bins
from   astropy.io         import ascii
from   matplotlib.backends.backend_pdf import PdfPages
from   astropy.stats      import sigma_clip
#==============================================================================
def compute_limits(data, sym=None, decimals=0):
   
   tmp = sigma_clip(data, sigma=4)
   if sym:
      mx = np.amax(np.fabs(tmp))
      lims = [-mx,mx]
   else:
      lims = [np.amin(tmp),np.amax(tmp)]   

   lims = np.round(lims, decimals=decimals)
   if lims[0] == lims[1]:
      lims[0] *= 0.9
      lims[1] *= 1.1
   
   return lims

#==============================================================================
def run_preproc_data(rname, struct):

    # Checking there is no missing keyword in configuration structure
    misc.check_configuration(struct)
    
    # Defining output filenames
    outhdf5 = "../preproc_data/"+rname+".hdf5"
    outpdf  = "../preproc_data/"+rname+".pdf"

    # Creating output directories if they do not exist
    if not os.path.exists("../preproc_data"):
          os.mkdir("../preproc_data")
    if os.path.exists(outhdf5):
          os.remove(outhdf5)
    if os.path.exists(outpdf):
          os.remove(outpdf)

    # Printing some basic info
    print("--------------------------------------------")
    print("- Input run name:   "+rname)
    print("- Survey:           "+struct['instrument'])
    print("- Wavelength range: "+str(struct['lmin'])+"-"+str(struct['lmax']))
    print("- Min SNR:          "+str(struct['snr_min']))
    print("- Target SNR:       "+str(struct['snr']))
    print("- Velscale:         "+str(struct['velscale']))
    print("- Vmax:             "+str(struct['vmax']))
    print("- Template library: "+str(struct['template_lib']))
    if struct['npca'] > 0:
        print("- Number of PCA:    "+str(struct['npca']))
    if 'xcen' in struct and 'ycen' in struct:
        print("- Xcenter:          "+str(struct['xcen']))
        print("- Ycenter:          "+str(struct['ycen']))
    print("--------------------------------------------")
    print("")

    # Processing data
    print("# Processing data .....") 
    data_struct = load_data(struct)

    # Processing templates 
    print("# Processing templates .....") 
    temp_struct = load_templates(struct,data_struct)

    # Creating the LOSVD velocity vector
    print("") 
    print("# Creating the LOSVD velocity vector")
    print("") 
    xvel, nvel = misc.create_xvel_vector(struct['vmax'], struct['velscale'])

    # Saving preprocessed information
    print("# Saving preproc data: "+outhdf5)
    print("")
    f    = h5py.File(outhdf5, "w")
    #------------
    f.create_dataset("in/xvel", data=xvel)
    f.create_dataset("in/nvel", data=nvel)
    f.create_dataset("in/npca", data=struct['npca'])
    #------------
    for key, val in data_struct.items():
       if (np.size(val) < 2):
          f.create_dataset("in/"+key, data=val)
       else:
          f.create_dataset("in/"+key, data=val, compression="gzip")
    #------------
    for key, val in temp_struct.items():
       if val is None:
          continue
       if (np.size(val) < 2):
          f.create_dataset("in/"+key, data=val)
       else:
          f.create_dataset("in/"+key, data=val, compression="gzip")
    #------------
    f.close()

    # Saving a simple plot with some basic figures about the pre-processed data
    print("# Plotting some basic info in "+outpdf)
    pdf_pages = PdfPages(outpdf)
 
    if data_struct['ndim'] == 2:

       # Bin map -----------
       fig = plt.figure(figsize=(10,7))
       plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.925, wspace=0.0, hspace=0.3)
       ax0 = plt.subplot2grid((1,1),(0,0))
      
       plt.sca(ax0)
       lims = compute_limits(np.log10(data_struct['bin_flux']),decimals=1)
       display_bins(data_struct['x'],data_struct['y'], data_struct['binID'], np.log10(data_struct['bin_flux']), 
                    vmin=lims[0], vmax=lims[1], pixelsize=data_struct['psize'], colorbar=True)

       ax0.set_title("BinID map")
       if 'xcen' in struct and 'ycen' in struct:
          ax0.axvline(0.0, ls=":", color='gray')
          ax0.axhline(0.0, ls=":", color='gray')

       pdf_pages.savefig(fig, dpi=300)
       plt.close()


    # Input brightest spectra and templates
    idx = np.argmax(data_struct['bin_flux'])

    fig = plt.figure(figsize=(10,7))
    plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.925, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot2grid((3,1),(0,0))
    ax1.plot(np.exp(data_struct['wave_obs']),data_struct['spec_obs'][:,idx],'k')
    ax1.set_ylabel("Brightest spec")
    ax1.set_xlim([np.exp(temp_struct['lwave_temp'])[0],np.exp(temp_struct['lwave_temp'])[-1]])

    ax2 = plt.subplot2grid((3,1),(1,0))
    ax2.plot(np.exp(temp_struct['lwave_temp']),temp_struct['mean_template'])
    ax2.set_ylabel("Mean Template")
    ax2.set_xlim([np.exp(temp_struct['lwave_temp'])[0],np.exp(temp_struct['lwave_temp'])[-1]])

    ax3 = plt.subplot2grid((3,1),(2,0))
    ax3.plot(np.exp(temp_struct['lwave_temp']),temp_struct['templates'])
    ax3.set_ylabel("Templates")          
    ax3.set_xlabel("Restframe wavelength ($\\mathrm{\\AA}$)")
    ax3.set_xlim([np.exp(temp_struct['lwave_temp'])[0],np.exp(temp_struct['lwave_temp'])[-1]])

    pdf_pages.savefig(fig)    
    pdf_pages.close()   
    plt.close()
   
    misc.printDONE(rname)

    return

#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("               BAYES-LOSVD                 ")
    print("             (preproc_data)                ")
    print("===========================================")
    print("")

    parser = argparse.ArgumentParser(
        prog="myscript.py",
        usage="%(prog)s -f file [options]",
        description="Process spectra with configurable parameters."
    )

    parser.add_argument("-c", "--config_file", type=str, default=None, help="Filename of the general config file")
    args = parser.parse_args()

    # Reading config file
    config = toml.load(args.config_file)
    cases  = list(config.keys())
    ncases = len(cases)
    
    # Procesing each case
    for i in range(ncases):
        misc.printRUNNING(cases[i]) 
        run_preproc_data(cases[i], config[cases[i]])
            
