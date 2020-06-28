import os
import optparse
import warnings
import h5py
import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
from   astropy.io         import ascii
from   lib.cap_utils      import display_bins
from   matplotlib.patches import Circle
#==============================================================================
def run_inspect_ghfit(filename,idx, losvd_file=None, norm=0, save=0):

   # Checking bin exists in dataset
   stridx = str(idx)
   f      = h5py.File(filename,"r")
   dummy  = f.get('out/'+stridx+'/offset')
   # print(dummy)
   if dummy == None:
      misc.printFAILED("Bin "+stridx+" does not exist in file")
      exit()

   # Reading input LOSVD if requested
   if not (losvd_file == None): 
      tab          = ascii.read(losvd_file)
      input_xvel   = tab['col1']
      input_losvd  = tab['col2'] / np.sum(tab['col2'])
    
   # Reading the results
   # --- Input data ----------
   xbin  = np.array(f['in/xbin'])
   ybin  = np.array(f['in/ybin'])
   xvel  = np.array(f['in/xvel'])
   nbins = len(xbin)
   # --- Output results ---------
   losvd        = np.array(f['out/'+stridx+'/losvd'])
   losvd_gh_mod = np.array(f['out/'+stridx+'/losvd_gh_mod'])
  
   # # Normalizing LOSVDs if requested ----------------------------------------------------------
   # if (norm == 1):
   #    # norm_factor = np.trapz(losvd[2,:],-xvel)
   #    norm_factor = np.sum(losvd[2,:])
   #    for i in range(5):
   #        losvd[i,:] /= norm_factor              

   # Making plot ----------------------------------------------------------

   # Bin map -----------
   if len(xbin) > 1:
      fig = plt.figure(figsize=(10,4))
      plt.subplots_adjust(left=0.07, bottom=0.15, right=0.98, top=0.925, wspace=0.0, hspace=0.3)
      fig.suptitle("BinID: "+str(idx), fontsize=14, fontweight='bold')
      ax0 = plt.subplot2grid((1,4),(0,0), colspan=1)
      ax0.set_title("BinID map")
      ax0.plot(xbin,ybin,'k+', zorder=0)
      ax0.plot(xbin[idx],ybin[idx],'r.', markersize=15.0)
      ax0.set_aspect('equal')
      for i in range(nbins):
          ax0.text(xbin[i],ybin[i],i, fontsize=5, horizontalalignment='right', verticalalignment='center',zorder=1)
            
   # LOSVD -----------
   if len(xbin) <= 1:
      fig = plt.figure(figsize=(6,4))
      plt.subplots_adjust(left=0.1, bottom=0.13, right=0.99, top=0.99)
      ax1 = plt.subplot2grid((1,1),(0,0), colspan=2)
   else:
      ax1 = plt.subplot2grid((1,4),(0,2), colspan=2)
   ax1.fill_between(xvel,losvd[0,:],losvd[4,:], color='gray', alpha=0.15, step='mid')
   ax1.fill_between(xvel,losvd[1,:],losvd[3,:], color='gray', alpha=0.50, step='mid')
   ax1.plot(xvel,losvd[2,:],'.--', color='black',ds='steps-mid', label='BAYES-LOSVD fit')

   ax1.fill_between(xvel,losvd_gh_mod[0,:],losvd_gh_mod[4,:], color='red', alpha=0.25, step='mid')
   ax1.fill_between(xvel,losvd_gh_mod[1,:],losvd_gh_mod[3,:], color='red', alpha=0.50, step='mid')
   ax1.plot(xvel,losvd_gh_mod[2,:],'r.-', ds='steps-mid', label='GH fit')

   if not (losvd_file == None):
      ax1.plot(input_xvel, input_losvd,'r.-', ds='steps-mid') 
   ax1.axhline(y=0.0,color='k', linestyle='--')
   ax1.axvline(x=0.0, color='k', linestyle=":")
   ax1.set_xlabel("Velocity (km s$^{-1}$)")
   ax1.legend()

   if (save == 1):
      dirname, inputname = os.path.split(filename) 
      basename = os.path.splitext(inputname)[0]
      outpng   = dirname+'/'+basename+'_bin'+stridx+'.png'
      plt.savefig(outpng)
   else:
      plt.show()

   return

#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("               BAYES-LOSVD                 ")
    print("            (inspect_ghfits)               ")
    print("===========================================")

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -f file")
    parser.add_option("-r", "--run",   dest="runname",  type="string", default=None,   help="Runname with the results")
    parser.add_option("-b", "--binID", dest="binID",    type="int",    default=0,      help="ID of the bin to plot")
    parser.add_option("-l", "--losvd", dest="losvd",    type="str",    default=None,   help="(Optional) Filename of the input LOSVD")
    parser.add_option("-s", "--save",  dest="save",     type="int",    default=0,      help="(Optional) Save figure")
    parser.add_option("-n", "--norm",  dest="norm",     type="int",    default=0,      help="(Optional) Normalizing LOSVD")
    parser.add_option("-d", "--dir",   dest="dir",      type="string", default='../results/', help="(Optional) The directory with results")


    (options, args) = parser.parse_args()
    runname    = options.runname
    binID      = options.binID
    losvd_file = options.losvd
    save       = options.save
    norm       = options.norm
    dir        = options.dir
    filename   = dir+runname+"/"+runname+"_gh_results.hdf5"

    if not os.path.exists(filename):
       misc.printFAILED(filename+" does not exist.")
       sys.exit()

    run_inspect_ghfit(filename,binID,losvd_file,norm=norm,save=save)

    misc.printDONE(runname+" - Bin: "+str(binID))
