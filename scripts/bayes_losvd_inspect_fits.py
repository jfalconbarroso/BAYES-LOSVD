import os
import sys
import argparse
import warnings
import h5py
import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
from   lib.cap_utils      import display_bins
from   astropy.stats      import sigma_clip
from   lib.create_params_plots import tiles_weight_and_uncertainty
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
#------------------------------------------------------------------------------
def reshape_to_2D(x,y,z):

   # Unique sorted axes
   x_unique = np.sort(np.unique(x))
   y_unique = np.sort(np.unique(y))

   # Initialize grid
   z_grid = np.full((len(x_unique), len(y_unique)), np.nan)  # fill with NaN initially

   # Fill the grid
   for a, m, v in zip(x, y, z):
       i = np.where(x_unique == a)[0][0]
       j = np.where(y_unique == m)[0][0]
       z_grid[i, j] = v

   return x_unique, y_unique, z_grid 

#==============================================================================
def run_inspect_fits(filename, idx, params_labels, figsize=None, wspace=None, hspace=None, 
                     top=None, bottom=None, left=None, right=None,
                     cbar_width=None, cbar_pad=None, save=False):
    
   # Reading the results
   f = h5py.File(filename,'r')

   # --- Input data ----------
   x        = np.array(f['in/x'])
   y        = np.array(f['in/y'])
   xbin     = np.array(f['in/xbin'])
   ybin     = np.array(f['in/ybin'])
   binID    = np.array(f['in/binID'])
   flux     = np.array(f['in/bin_flux'])
   psize    = np.array(f['in/psize'])
   xvel     = np.array(f['in/xvel'])
   wave_obs = np.exp(np.array(f['in/wave_obs']))
   spec_obs = np.array(f['in/spec_obs'][:,idx])
   mask     = np.array(f['in/mask'])
   ndim     = np.array(f['in/ndim'])
   npca     = np.array(f['in/npca'])
   params   = np.array(f['in/params']).T
   nparams  = params.shape[1]
   # --- Output results ---------
   poly    = np.array(f['out/continuum'][idx,:])
   bestfit = np.array(f['out/model_spec'][idx,:])
   losvd   = np.array(f['out/losvd'][idx,:])
   weights = np.array(f['out/weights'][idx,:])

   if np.isnan(losvd).all():
      misc.printFAILED("The selected bin has no results")
      return
   
   # Making LOSVD and spectral fit plot ----------------------------------------------------------
   fig = plt.figure(figsize=(11,9))
   plt.subplots_adjust(left=0.05, bottom=0.065, right=0.99, top=0.99, wspace=0.6, hspace=0.3)

   # Bin map -----------
   if ndim > 1:
      ax = plt.subplot2grid((2,4),(0,0), colspan=2)
      plt.sca(ax)
      lims = compute_limits(np.log10(flux),decimals=1)

      display_bins(x, y, binID, np.log10(flux), vmin=lims[0], vmax=lims[1], pixelsize=psize, colorbar=False, cmap="Blues")
      ax.plot(xbin,ybin,'k+', zorder=0)
      ax.plot(xbin[idx],ybin[idx],'y.', mec='black', markersize=20.0)

   # LOSVD -----------
   ax = plt.subplot2grid((2,4),(0,2), colspan=2)
   ax.fill_between(xvel,losvd[0,:],losvd[4,:], color='blue', alpha=0.15, step='mid')
   ax.fill_between(xvel,losvd[1,:],losvd[3,:], color='blue', alpha=0.50, step='mid')
   ax.plot(xvel,losvd[2,:],'k.-', ds='steps-mid')
   ax.axhline(y=0.0,color='k', linestyle='--')
   ax.axvline(x=0.0, color='k', linestyle=":")
   ax.set_xlabel("Velocity (km s$^{-1}$)")

   xlims = ax.get_xlim()
   ylims = ax.get_ylim()
   ax.text(xlims[0]*0.9, ylims[1]*0.9,"BinID: "+str(idx), ha="left", fontsize=14, fontweight='bold' )

   # Spectral fit
   mx  = 1.1*np.amax(spec_obs)
   mn0 = 0.7*np.amin(spec_obs)
   ax = plt.subplot2grid((2,4),(1,0), colspan=4)
   ax.fill_between(wave_obs,poly[1,:],poly[3,:], facecolor='yellow',zorder=0, alpha=0.50,label="Leg. polynomial")
   ax.plot(wave_obs,poly[1,:], color='gray',linestyle='--', linewidth=1,zorder=0)
   ax.plot(wave_obs,poly[3,:],color='gray',linestyle='--', linewidth=1,zorder=0)
   ax.plot(wave_obs,spec_obs,'k', zorder=1,label="Obs. data")
   ax.fill_between(wave_obs,bestfit[1,:],bestfit[3,:], facecolor='orange',zorder=2, alpha=0.75)
   ax.plot(wave_obs,bestfit[2,:],color='red',zorder=3, label="Bestfit")
   res = spec_obs - bestfit[2,:] + mn0 + 0.1
   ax.plot(wave_obs, res, color='green', label="Residuals")
   ax.set_ylim([mn0,mx])
   ax.axhline(y=mn0+0.1,color='k', linestyle='--')
   ax.axvline(x=wave_obs[mask[0]],  color='k', linestyle=":")
   ax.axvline(x=wave_obs[mask[-1]], color='k', linestyle=":")

   w = np.flatnonzero(np.diff(mask) > 1)
   if w.size > 0:
       for wj in w:
         l0 = wave_obs[mask[wj]]
         l1 = wave_obs[mask[wj+1]]
         ax.axvspan(l0,l1, alpha=0.25, color='gray')

   ax.set_ylabel("Norm. flux")
   ax.set_xlabel("Wavelength ($\\mathrm{\\AA}$)")

   if save:
      dirname, inputname = os.path.split(filename)
      basename = os.path.splitext(inputname)[0]
      outpng   = dirname+'/'+basename+'_bin'+str(idx)+'_losvdfit.png'
      print(" Saving plot at: "+outpng)
      plt.savefig(outpng)

   # Making parameters corner plots ----------------------------------------------------------
   if npca == 0:

      # params: (ncomb, npars)  (often these are bin CENTERS along each dimension)
      # weights: (ncomb,)
      if params_labels is None:
         labels = ['P'+str(i+1) for i in range(nparams)]
         # labels = ["Log(Age) [Gyr]", "[M/H] (dex)"]#, "[Mg/Fe] (dex)"]
      else:  
         labels = params_labels.split(",")

      tiles_weight_and_uncertainty(
          params, weights[5,:], weights[6,:], 
          labels=labels, figsize=figsize, cbar_width=cbar_width, cbar_pad=cbar_pad,
          left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace,
          reducer = "sum", uncertainty_reducer="quadrature" 
      )

   if save:
      dirname, inputname = os.path.split(filename)
      basename = os.path.splitext(inputname)[0]
      outpng   = dirname+'/'+basename+'_bin'+str(idx)+'_params.png'
      print(" Saving plot at: "+outpng)
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

   parser = argparse.ArgumentParser(
       prog="myscript.py",
       usage="%(prog)s -f file [options]",
       description="Process spectra with configurable parameters."
   )

   parser.add_argument("-f", "--filename", type=str,   default=None,     help="File with the results")
   parser.add_argument("-l", "--bin",      type=int,   default=None,     help="Bin ID for spectrum display")
   parser.add_argument("--figsize",        type=float, nargs=2, metavar=("W", "H"),  default=(12, 6))
   parser.add_argument("--wspace",         type=float, default=0.45)
   parser.add_argument("--hspace",         type=float, default=0.25)
   parser.add_argument("--left",           type=float, default=0.06)
   parser.add_argument("--right",          type=float, default=0.92)
   parser.add_argument("--top",            type=float, default=0.98)
   parser.add_argument("--bottom",         type=float, default=0.10)
   parser.add_argument("--cbar_width",     type=float, default=0.05)
   parser.add_argument("--cbar_pad",       type=float, default=0.0)
   parser.add_argument("--labels",         type=str,   default=None, help="Labels for the input Params")
   parser.add_argument("-s", "--save",    action="store_true",  help="(Optional) Save figure")   
   args = parser.parse_args()

   if not os.path.exists(args.filename):
      misc.printFAILED(args.filename+" does not exist.")
      sys.exit()

   if args.bin is None:

      f = h5py.File(args.filename,'r')
      nbins = f['in/nbins'][...]
      f.close()

      for idx in range(nbins):
         run_inspect_fits(args.filename, idx, args.labels,
                          figsize=args.figsize, 
                          wspace=args.wspace,
                          hspace=args.hspace,
                          left=args.left,
                          right=args.right,
                          top=args.top,
                          bottom=args.bottom,
                          cbar_width=args.cbar_width,
                          cbar_pad=args.cbar_pad,
                          save=args.save
                        )

   else:

      run_inspect_fits(args.filename, args.bin, args.labels,
                       figsize=args.figsize, 
                       wspace=args.wspace,
                       hspace=args.hspace,
                       left=args.left,
                       right=args.right,
                       top=args.top,
                       bottom=args.bottom,
                       cbar_width=args.cbar_width,
                       cbar_pad=args.cbar_pad,
                       save=args.save
                     )

   misc.printDONE()
