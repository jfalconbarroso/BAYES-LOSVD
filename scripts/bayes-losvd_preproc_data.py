import os
import h5py
import warnings
import optparse
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
    print("- Target SNR:       "+str(struct['snr']))
    print("- Min SNR:          "+str(struct['snr_min']))
    print("- Redshift:         "+str(struct['redshift']))
    print("- Velscale:         "+str(struct['velscale']))
    print("- LOSVD Vmax:       "+str(struct['vmax']))
    print("- Mask file:        "+str(struct['mask_file']))
    print("- Pol. order:       "+str(struct['porder']))
    print("- Templates:        "+str(struct['template_lib']))
    print("- Number of PCA:    "+str(struct['npca']))
    print("--------------------------------------------")
    print("")

    # Processing data 
    data_struct = load_data(struct)
    exit()

    # Processing templates 
    temp_struct = load_templates(struct,data_struct)

    # Saving preprocessed information
    print("")
    print(" - Saving preproc data: "+outhdf5)
    print("")
    f    = h5py.File(outhdf5, "w")
    #------------
    for key, val in data_struct.items():
       if (np.size(val) < 2):
          f.create_dataset("in/"+key, data=val)
       else:
          f.create_dataset("in/"+key, data=val, compression="gzip")
    #------------
    for key, val in temp_struct.items():
       if (np.size(val) < 2):
          f.create_dataset("in/"+key, data=val)
       else:
          f.create_dataset("in/"+key, data=val, compression="gzip")
    #------------
    f.close()

    # Saving a simple plot with some basic figures about the pre-processed data
    print(" - Plotting some basic info in "+outpdf)
    pdf_pages = PdfPages(outpdf)
 
    if not (struct['Survey'][idx] == 'TEST'):

       # Bin map -----------
       fig = plt.figure(figsize=(10,7))
       plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.925, wspace=0.0, hspace=0.3)
       ax0 = plt.subplot2grid((1,1),(0,0))
       ax0.set_aspect('equal')
       ax0.set_title("BinID map")
       ax0.plot(data_struct['xbin'],data_struct['ybin'],'+', color='gray')
       for i in range(np.amax(data_struct['binID'])+1):
           ax0.text(data_struct['xbin'][i],data_struct['ybin'][i],i, fontsize=5, horizontalalignment='left', verticalalignment='center')
       pdf_pages.savefig(fig)

    # Input central spectra including mask and PCA templates
    fig = plt.figure(figsize=(10,7))
    plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.925, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot2grid((3,1),(0,0))
    ax1.plot(np.exp(data_struct['wave_obs']),data_struct['spec_obs'][:,0],'k')
    ax1.set_ylabel("Central spec")
    ax1.set_xlim([np.exp(temp_struct['lwave_temp'])[0],np.exp(temp_struct['lwave_temp'])[-1]])
    ax1.axvline(x=np.exp(data_struct['wave_obs'][data_struct['mask'][0]]),  color='k', linestyle=":")
    ax1.axvline(x=np.exp(data_struct['wave_obs'][data_struct['mask'][-1]]), color='k', linestyle=":")

    w = np.flatnonzero(np.diff(data_struct['mask']) > 1)
    if w.size > 0:
       for wj in w:
         l0 = np.exp(data_struct['wave_obs'][data_struct['mask'][wj]])
         l1 = np.exp(data_struct['wave_obs'][data_struct['mask'][wj+1]])
         ax1.axvspan(l0,l1, alpha=0.5, color='red')  
        
    ax2 = plt.subplot2grid((3,1),(1,0))
    ax2.plot(np.exp(temp_struct['lwave_temp']),temp_struct['mean_template'],'k')
    ax2.set_ylabel("Mean Template")
    ax2.set_xlim([np.exp(temp_struct['lwave_temp'])[0],np.exp(temp_struct['lwave_temp'])[-1]])

    ax3 = plt.subplot2grid((3,1),(2,0))
    ax3.plot(np.exp(temp_struct['lwave_temp']),temp_struct['templates'])
    ax3.set_ylabel("PCA Templates")
    ax3.set_xlabel("Restframe wavelength ($\\mathrm{\\AA}$)")
    ax3.set_xlim([np.exp(temp_struct['lwave_temp'])[0],np.exp(temp_struct['lwave_temp'])[-1]])

    pdf_pages.savefig(fig)    
    pdf_pages.close()   
   
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

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -c cfile")
    parser.add_option("-c", "--config",  dest="config_file", type="string", default="../config_files/example_preproc.properties", help="Filename of the general config file")

    (options, args) = parser.parse_args()
    config_file = options.config_file
    
    # Reading config file
    config = toml.load(config_file)
    cases  = list(config.keys())
    ncases = len(cases)
    
    # Procesing each case
    for i in range(ncases):
        misc.printRUNNING(cases[i]) 
        run_preproc_data(cases[i], config[cases[i]])
        
    exit()
    
