import os
import sys
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
def run_inspect_fits(filename,idx, losvd_file=None, save=0):

   # Checking bin exists in dataset
   stridx = str(idx)
   f      = h5py.File(filename,"r")
   dummy  = f.get('out/'+stridx+'/bestfit')
   if dummy == None:
      misc.printFAILED("ERROR: Bin "+stridx+" does not exist in file")
      sys.exit()

   # Reading input LOSVD if requested
   if not (losvd_file == None): 
      tab          = ascii.read(losvd_file)
      input_xvel   = tab['col1']
      input_losvd  = tab['col2'] / np.sum(tab['col2'])
    
   # Reading the results
   # --- Input data ----------
   xbin     = np.array(f['in/xbin'])
   ybin     = np.array(f['in/ybin'])
   wave_obs = np.exp(np.array(f['in/wave_obs']))
   spec_obs = np.array(f['in/spec_obs'][:,idx])
   xvel     = np.array(f['in/xvel'])
   mask     = np.array(f['in/mask'])
   ndim     = np.array(f['in/ndim'])
   # --- Output results ---------
   bestfit  = np.array(f['out/'+stridx+'/bestfit'])
   losvd    = np.array(f['out/'+stridx+'/losvd'])
   poly     = np.array(f['out/'+stridx+'/poly'])+1.0
   nbins    = len(xbin)
  
   # Normalizing LOSVDs ----------------------------------------------------------
   norm_factor = np.sum(losvd[2,:])
   for i in range(5):
       losvd[i,:] /= norm_factor              

   # Making plot ----------------------------------------------------------
   fig = plt.figure(figsize=(10,7))
   fig.suptitle("BinID: "+str(idx), fontsize=14, fontweight='bold')
   plt.subplots_adjust(left=0.07, bottom=0.10, right=0.98, top=0.925, wspace=0.0, hspace=0.3)

   # Bin map -----------
   if ndim > 1:
      ax0 = plt.subplot2grid((2,4),(0,0), colspan=1)
      ax0.set_title("BinID map")
      ax0.plot(xbin,ybin,'k+', zorder=0)
      ax0.plot(xbin[idx],ybin[idx],'r.', markersize=15.0)
      ax0.set_aspect('equal')
      for i in range(nbins):
          ax0.text(xbin[i],ybin[i],i, fontsize=5, horizontalalignment='right', verticalalignment='center',zorder=1)
            
   # LOSVD -----------
   ax1 = plt.subplot2grid((2,4),(0,2), colspan=2)
   ax1.fill_between(xvel,losvd[0,:],losvd[4,:], color='blue', alpha=0.15, step='mid')
   ax1.fill_between(xvel,losvd[1,:],losvd[3,:], color='blue', alpha=0.50, step='mid')
   ax1.plot(xvel,losvd[2,:],'k.-', ds='steps-mid')
   if not (losvd_file == None):
      ax1.plot(input_xvel, input_losvd,'r.-', ds='steps-mid') 
   ax1.axhline(y=0.0,color='k', linestyle='--')
   ax1.axvline(x=0.0, color='k', linestyle=":")
   ax1.set_xlabel("Velocity (km s$^{-1}$)")

   # Spectral fit
   mx  = 1.1*np.amax(spec_obs)
   mn0 = 0.7*np.amin(spec_obs)
   ax2 = plt.subplot2grid((2,4),(1,0), colspan=4)
   ax2.fill_between(wave_obs,poly[1,:],poly[3,:], facecolor='yellow',zorder=0, alpha=0.50,label="Leg. polynomial")
   ax2.plot(wave_obs,poly[1,:], color='gray',linestyle='--', linewidth=1,zorder=0)
   ax2.plot(wave_obs,poly[3,:],color='gray',linestyle='--', linewidth=1,zorder=0)
   ax2.plot(wave_obs,spec_obs,'k', zorder=1,label="Obs. data")
   ax2.fill_between(wave_obs,bestfit[1,:],bestfit[3,:], facecolor='orange',zorder=2, alpha=0.75)
   ax2.plot(wave_obs,bestfit[2,:],color='red',zorder=3, label="Bestfit")
   res = spec_obs - bestfit[2,:] + mn0 + 0.1
   ax2.plot(wave_obs, res, color='green', label="Residuals")
   ax2.set_ylim([mn0,mx])
   ax2.axhline(y=mn0+0.1,color='k', linestyle='--')
   ax2.axvline(x=wave_obs[mask[0]],  color='k', linestyle=":")
   ax2.axvline(x=wave_obs[mask[-1]], color='k', linestyle=":")

   w = np.flatnonzero(np.diff(mask) > 1)
   if w.size > 0:
       for wj in w:
         l0 = wave_obs[mask[wj]]
         l1 = wave_obs[mask[wj+1]]
         ax2.axvspan(l0,l1, alpha=0.25, color='gray')  

   ax2.set_ylabel("Norm. flux")
   ax2.set_xlabel("Wavelength ($\\mathrm{\\AA}$)")

   print(1.0/np.std(res[mask]-mn0-0.1))
   # exit()

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
    print("             (inspect_fits)                ")
    print("===========================================")

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -f file")
    parser.add_option("-r", "--run",   dest="runname",  type="string", default=None,   help="Runname with the results")
    parser.add_option("-b", "--binID", dest="binID",    type="int",    default=0,      help="ID of the bin to plot")
    parser.add_option("-l", "--losvd", dest="losvd",    type="str",    default=None,   help="(Optional) Filename of the input LOSVD")
    parser.add_option("-s", "--save",  dest="save",     type="int",    default=0,      help="(Optional) Save figure")
    parser.add_option("-d", "--dir",   dest="dir",      type="string", default='../results/', help="(Optional) The directory with results")


    (options, args) = parser.parse_args()
    runname    = options.runname
    binID      = options.binID
    losvd_file = options.losvd
    save       = options.save
    dir        = options.dir
    filename   = dir+runname+"/"+runname+"_results.hdf5"

    if not os.path.exists(filename):
       misc.printFAILED(filename+" does not exist.")
       sys.exit()

    run_inspect_fits(filename,binID,losvd_file,save=save)

    misc.printDONE(runname+" - Bin: "+str(binID))
