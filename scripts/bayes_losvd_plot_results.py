import os
import sys
import optparse
import warnings
import h5py
import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
from   lib.cap_utils      import display_bins
from   astropy.stats      import sigma_clip
#==============================================================================
def save_plot(save, filename, label):

   if save == 1:
      dirname, inputname = os.path.split(filename)
      basename = os.path.splitext(inputname)[0]
      outpng   = dirname+'/'+basename+'_'+label+'.png'
      print(" - Saving file: "+outpng)
      plt.savefig(outpng)

   return
#------------------------------------------------------------------------------
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
def put_label(label, fontsize=10, fontweight='bold', loc=(0.95,0.95)):

   plt.text(
    loc[0], loc[1],                          # x, y (relative to axes)
    label, # label text
    transform=plt.gca().transAxes,      # use axes-relative coords
    fontsize=fontsize,
    fontweight=fontweight,
    va='top', ha='right',               # align text to top-right corner
    bbox=dict(
        facecolor='white',              # opaque white background
        alpha=0.8,                      # transparency
        edgecolor='black',              # box border
        boxstyle='round,pad=0.3'        # rounded box
    )
   )
   
   return

# 'normal'	Standard text weight
# 'bold'	Bold
# 'heavy'	Extra bold
# 'light'	Lighter than normal
# 'ultrabold'	Heavier than bold
# 'ultralight'	Lighter than light
# 'medium'	Between light and bold
# 'semibold'	Slightly bolder than medium
# 'black'	Very heavy (usually the heaviest)

#==============================================================================
if (__name__ == '__main__'):

   warnings.filterwarnings("ignore")

   print("===========================================")
   print("                BAYES-LOSVD                ")
   print("              (plot_results)               ")
   print("===========================================")

   # Capturing the command line arguments
   parser = optparse.OptionParser(usage="%prog -f file")
   parser.add_option("-f", "--filename", dest="filename", type="string", default=None,   help="Filen with the results")
   parser.add_option("-s", "--save",     dest="save",     type="int",    default=0,      help="(Optional) Save figure")

   (options, args) = parser.parse_args()
   filename = options.filename
   save     = options.save

   if not os.path.exists(filename):
      misc.printFAILED(filename+" does not exist.")
      sys.exit()

   # Reading the results
   f = h5py.File(filename,'r')

   # --- Input data ----------
   x     = np.array(f['in/x'])
   y     = np.array(f['in/y'])
   xbin  = np.array(f['in/xbin'])
   ybin  = np.array(f['in/ybin'])
   binID = np.array(f['in/binID'])
   flux  = np.array(f['in/bin_flux'])
   psize = np.array(f['in/psize'])
   xvel  = np.array(f['in/xvel'])
   # --- Output data ----------
   params_obs = np.array(f['out/mean_params'][:,5,:])
   params_std = np.array(f['out/mean_params'][:,6,:])
   vel_star   = np.array(f['out/vel_star'])
   sigma_star = np.array(f['out/sigma_star'])
   h3_star    = np.array(f['out/h3_star'])
   h4_star    = np.array(f['out/h4_star'])
   npar       = params_obs.shape[-1]
   
   # KINEMATIC MAPS -----------
   fig,ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(12,5))
   plt.subplots_adjust(left=0.05, bottom=0.065, right=0.95, top=0.95, wspace=0.2, hspace=0.25)   
   ax = ax.ravel()

   lims_vel   = compute_limits(vel_star[:,5],decimals=3, sym=True)
   lims_sigma = compute_limits(sigma_star[:,5],decimals=3)

   plt.sca(ax[0])
   display_bins(x, y, binID, vel_star[:,5], vmin=lims_vel[0], vmax=lims_vel[1], pixelsize=psize, colorbar=True, cmap='jet')
   ax[0].set_title("Velocity fit")

   plt.sca(ax[1])
   display_bins(x, y, binID, sigma_star[:,5], vmin=lims_sigma[0], vmax=lims_sigma[1], pixelsize=psize, colorbar=True, cmap='rainbow')
   ax[1].set_title("Vel. Dispersion fit")

   plt.sca(ax[2])
   display_bins(x, y, binID, h3_star[:,5], vmin=-0.15, vmax=0.15, pixelsize=psize, colorbar=True, cmap='jet')
   ax[2].set_title("h$_3$ fit")

   plt.sca(ax[3])
   display_bins(x, y, binID, h4_star[:,5], vmin=-0.15, vmax=0.15, pixelsize=psize, colorbar=True, cmap='jet')
   ax[3].set_title("h$_4$ fit")

   plt.sca(ax[4])
   display_bins(x, y, binID, vel_star[:,6], vmin=0.0, vmax=np.nanmax(vel_star[:,6]), pixelsize=psize, colorbar=True, cmap='rainbow')
   ax[4].set_title("Velocity unc.")

   plt.sca(ax[5])
   display_bins(x, y, binID, sigma_star[:,6], vmin=0.0, vmax=np.nanmax(sigma_star[:,6]), pixelsize=psize, colorbar=True, cmap='rainbow')
   ax[5].set_title("Vel. Dispersion unc.")

   plt.sca(ax[6])
   display_bins(x, y, binID, h3_star[:,6], vmin=0.0, vmax=0.15, pixelsize=psize, colorbar=True, cmap='rainbow')
   ax[6].set_title("h$_3$ unc.")

   plt.sca(ax[7])
   display_bins(x, y, binID, h4_star[:,6], vmin=0.0, vmax=0.15, pixelsize=psize, colorbar=True, cmap='rainbow')
   ax[7].set_title("h$_3$ unc.")

   save_plot(save, filename,'GH_kinematics')


   # PARAMS MAPS -----------
   fig,ax = plt.subplots(nrows=npar, ncols=2, sharex=True, sharey=True, figsize=(10,10))
   plt.subplots_adjust(left=0.05, bottom=0.065, right=0.92, top=0.95, wspace=0.2, hspace=0.25)   

   for i in range(npar): 
      
       lims_par   = compute_limits(params_obs[:,i],decimals=3)
       std_par_mx = np.nanmax(params_std[:,i])

       plt.sca(ax[i,0])
       display_bins(x, y, binID, params_obs[:,i], vmin=lims_par[0], vmax=lims_par[1], pixelsize=psize, colorbar=True, cmap='rainbow')
       ax[i,0].set_title("Param "+str(i)+" fit")

       plt.sca(ax[i,1])
       display_bins(x, y, binID, params_std[:,i], vmin=0.0, vmax=std_par_mx, pixelsize=psize, colorbar=True, cmap='rainbow')
       ax[i,1].set_title("Param "+str(i)+" unc.")

   save_plot(save, filename,'params')


   if save == 0:
      plt.show() 
