import h5py
from astropy.io          import ascii
import numpy             as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#==============================================================================
def figure2_plot(snr, suffix=''):

   if suffix == "_11":
      kk = "_bin0"
   elif suffix == "_penalised_11":
      kk = "_bin0"   
   else:
      kk = ''   

   # Loading files with results
   outdir = "../results_denso/"
   files  = ["testcase_Gaussian_Young_SNR"+snr+"-NPCA2"+suffix+"_results"+kk+".hdf5",
             "testcase_Gaussian_Young_SNR"+snr+"-NPCA5"+suffix+"_results"+kk+".hdf5",
             "testcase_Gaussian_Young_SNR"+snr+"-NPCA10"+suffix+"_results"+kk+".hdf5",
             "testcase_Gaussian_Old_SNR"+snr+"-NPCA2"+suffix+"_results"+kk+".hdf5",
             "testcase_Gaussian_Old_SNR"+snr+"-NPCA5"+suffix+"_results"+kk+".hdf5",
             "testcase_Gaussian_Old_SNR"+snr+"-NPCA10"+suffix+"_results"+kk+".hdf5",
             "testcase_Gaussian_Mixed_SNR"+snr+"-NPCA2"+suffix+"_results"+kk+".hdf5",
             "testcase_Gaussian_Mixed_SNR"+snr+"-NPCA5"+suffix+"_results"+kk+".hdf5",
             "testcase_Gaussian_Mixed_SNR"+snr+"-NPCA10"+suffix+"_results"+kk+".hdf5"]


   nfiles = len(files)
   labels = ['Young','Old','Mixed']

   # Loading input LOSVD
   tab = ascii.read("../losvd/losvd_Gaussian_velscale50.dat")
   tab['col2'] /= np.trapz(tab['col2'],tab['col1'])

   # Plotting figure
   fig, ax = plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True,figsize=(10,6))
   ax = ax.ravel()
   plt.subplots_adjust(left=0.025, bottom=0.08, right=0.99, top=0.99, wspace=0.05, hspace=0.0)

   for i in range(nfiles):

       rname     = str.split(files[i],'_results'+kk+'.hdf5')[0]
       npca      = str.split(rname,'NPCA')[1]
       f         = h5py.File(outdir+rname+"/"+files[i],'r')
       xvel      = np.array(f['in/xvel'])
       wave_obs  = np.exp(np.array(f['in/wave_obs']))
       spec_obs  = np.array(f['in/spec_obs'])
       mask      = np.array(f['in/mask'])
       bestfit   = np.array(f['out/0/bestfit'])
       poly      = np.array(f['out/0/poly'])+1.0
       losvd     = np.array(f['out/0/losvd'])
       f.close()

       losvd_3s_lo  = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
       losvd_3s_hi  = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
       losvd_1s_lo  = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
       losvd_1s_hi  = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
       losvd_med    = losvd[2,:]/np.trapz(losvd[2,:],-xvel)


       mx  = 1.1*np.amax(spec_obs[:,0])
       mn0 = 0.7*np.amin(spec_obs[:,0])
       mx  = 3.0
       mn0 = 0.5

       res = spec_obs[:,0] - bestfit[2,:] + mn0 + 0.1

       # Spectral fits plots ---------
       ax[i].text(4750,2.7,'NPCA: '+npca,horizontalalignment='left', fontweight='bold',fontsize=10)

      #  ax[i] = plt.subplot2grid((3,3),(i,0), colspan=2)
      #  ax[i].fill_between(wave_obs[mask],poly[0,mask],poly[4,mask], facecolor='yellow',zorder=0, alpha=0.15)
       ax[i].fill_between(wave_obs[mask],poly[1,mask],poly[3,mask], facecolor='yellow',zorder=0, alpha=0.50,label="Leg. polynomial")
       ax[i].plot(wave_obs[mask],poly[1,mask], color='gray',linestyle='--', linewidth=1,zorder=0)
       ax[i].plot(wave_obs[mask],poly[3,mask],color='gray',linestyle='--', linewidth=1,zorder=0)
       ax[i].plot(wave_obs[mask],spec_obs[mask,0],'k', zorder=1,label="Obs. data", ds='steps-mid')
      #  ax[i].fill_between(wave_obs[mask],bestfit[0,mask],bestfit[4,mask], facecolor='orange',zorder=2, alpha=0.5, step='mid')
       ax[i].fill_between(wave_obs[mask],bestfit[1,mask],bestfit[3,mask], facecolor='orange',zorder=2, alpha=0.75, step='mid')
       ax[i].plot(wave_obs[mask],bestfit[2,mask],color='red',zorder=3, label="Bestfit", ds='steps-mid')
       ax[i].plot(wave_obs[mask], res[mask], color='green', label="Residuals",ds='steps-mid')
       ax[i].set_ylim([mn0,mx])
       ax[i].axhline(y=mn0+0.1,color='k', linestyle='--')
    #    ax.axvline(x=wave_obs[mask[0]],  color='k', linestyle=":")
    #    ax.axvline(x=wave_obs[mask[-1]], color='k', linestyle=":")
       ax[i].set_yticks([])
       ax[i].set_xlabel("Wavelength ($\\mathrm{\\AA}$)")

       if i == 0:
          ax[i].set_ylabel(labels[0],fontweight='bold')
       elif i == 3:
          ax[i].set_ylabel(labels[1],fontweight='bold')
       elif i == 6:
          ax[i].set_ylabel(labels[2],fontweight='bold')


       #  # LOSVD plots ---------
       axins = inset_axes(ax[i], width="50%", height="50%")
       axins.set_yticks([])

       axins.fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
       axins.fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
       axins.plot(xvel,losvd_med,'k.-', ds='steps-mid')
       axins.plot(tab['col1'], tab['col2'],'r.-', ds='steps-mid') 
       axins.axhline(y=0.0,color='k', linestyle='--')
       axins.axvline(x=0.0, color='k', linestyle="--")

       axins.plot(xvel, -0.001 + losvd_med - tab['col2'],color='darkgray', ds='steps-mid')
       axins.axhline(y=-0.001,color='k', linestyle=':')
       axins.set_ylim([-0.00175,0.004])

#    plt.show()   
   fig.savefig("bayes-losvd_fig2_SNR"+snr+suffix+".pdf",dpi=600)

   return

#==============================================================================
if (__name__ == '__main__'):

   figure2_plot("10")
   figure2_plot("25")
   figure2_plot("50")
   figure2_plot("100")
   figure2_plot("200")

   figure2_plot("10", suffix="_11")
   figure2_plot("25", suffix="_11")
   figure2_plot("50", suffix="_11")
   figure2_plot("100", suffix="_11")
   figure2_plot("200", suffix="_11")   

   figure2_plot("10",suffix="_penalised")
   figure2_plot("25",suffix="_penalised")
   figure2_plot("50",suffix="_penalised")
   figure2_plot("100",suffix="_penalised")
   figure2_plot("200",suffix="_penalised")

   figure2_plot("10",suffix="_penalised_11")
   figure2_plot("25",suffix="_penalised_11")
   figure2_plot("50",suffix="_penalised_11")
   figure2_plot("100",suffix="_penalised_11")
   figure2_plot("200",suffix="_penalised_11")
