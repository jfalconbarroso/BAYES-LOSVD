import os
import h5py
import warnings
import optparse
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
def run_preproc_data(struct,idx):

    # Checking the file exists
    rname = struct['Runname'][idx]
    rootname = rname.split('-')[0]
    filename = "../data/"+rootname+'.fits'
    if not os.path.exists(filename):
       misc.printFAILED(filename+" does not exist")
       exit()

    outhdf5 = "../preproc_data/"+rname+".hdf5"
    outpdf  = "../preproc_data/"+rname+".pdf"

    if not os.path.exists("../preproc_data"):
          os.mkdir("../preproc_data")
    if os.path.exists(outhdf5):
          os.remove(outhdf5)
    if os.path.exists(outpdf):
          os.remove(outpdf)

    # Printing some basic info
    print("-------------------------------------------")
    print("- Input run name:   "+struct['Runname'][idx])
    print("- Survey:           "+struct['Survey'][idx])
    print("- Wavelength range: "+str(struct['Lmin'][idx])+"-"+str(struct['Lmax'][idx]))
    print("- Target SNR:       "+str(struct['SNR'][idx]))
    print("- Min SNR:          "+str(struct['SNR_min'][idx]))
    print("- Redshift:         "+str(struct['Redshift'][idx]))
    print("- Velscale:         "+str(struct['Velscale'][idx]))
    print("- LOSVD Vmax:       "+str(struct['Vmax'][idx]))
    print("- Mask flag:        "+str(struct['Mask'][idx]))
    print("- Pol. order:       "+str(struct['Porder'][idx]))
    print("- Bsplines order:   "+str(struct['Border'][idx]))
    print("- Templates:        "+str(struct['Templates'][idx]))
    print("- Number of PCA:    "+str(struct['NPCA'][idx]))
    print("-------------------------------------------")
    print("")

    # Processing data 
    if (struct['Survey'][idx] == 'TEST'):
       data_struct = load_testdata(struct,idx)
    else:   
       data_struct = load_data(struct,idx)
   
    # Processing templates 
    temp_struct = load_templates(struct,idx,data_struct)

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
    print("           (preproc_testdata)              ")
    print("===========================================")
    print("")

    # Defining cases
    _case   = ['Marie1','Marie2','Gaussian','Double_Gaussian']
    _snr    = [10,25,50,100,200]
    _ssp    = ['Old','Young','Mixed']
    _ftype  = ['free','penalised']
    _border = [0,1,2,3,4]

    case, snr, ssp, ftype, border = np.meshgrid(_case, _snr, _ssp, _ftype, _border)
    case   = case.ravel()
    snr    = snr.ravel()
    ssp    = ssp.ravel()
    ftype  = ftype.ravel()
    border = border.ravel()
    ncases = len(snr)

    config = {'Runname': list(np.repeat('',ncases)),'Survey': np.repeat('TEST',ncases), 'Lmin': np.repeat(4750.0,ncases), 'Lmax': np.repeat(5500.0,ncases), 
              'Vmax': np.repeat(700.0,ncases), 'Velscale': np.repeat(50.0,ncases), 'Redshift': np.repeat(0.0,ncases), 'SNR': np.repeat(0.0,ncases), 
              'SNR_min': np.repeat(0.0,ncases), 'Porder': np.repeat(5,ncases), 'Mask': np.repeat(0,ncases), 'NPCA':np.repeat(10,ncases), 
              'Templates': list(np.repeat('MILES_SSP',ncases)), 'Border': np.repeat(0,ncases), 'Mask_width': np.repeat(250.0,ncases)}
   
    # Procesing each case
    for i in range(ncases):
        config['Runname'][i] = 'testcase_'+case[i]+'_'+ssp[i]+'_SNR'+str(snr[i])+'-Border'+str(border[i])+'_'+ftype[i]
        if (ftype[i] == 'penalised') and (border[i] == 0):
            border[i] = -100
        config['Border'][i]  = border[i]

        misc.printRUNNING(config['Runname'][i]) 
        run_preproc_data(config,i)
        
    exit()
    
