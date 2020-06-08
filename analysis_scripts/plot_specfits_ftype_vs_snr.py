import h5py
from astropy.io          import ascii
import numpy             as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#==============================================================================
def plot_figure(case, id=[10,30,60,80,110]):

   # Loading files with results
   snr   = [10,25,50,100,200]
   ftype = ['S0','S1','A1','A2','A3','A4','B3','B4']
   outdir = "../results_deimos_v4/"

   # Loading input LOSVD
   tab = ascii.read("../losvd/"+case+"_velscale50.dat")
   tab['col2'] /= np.trapz(tab['col2'],tab['col1'])

   # Plotting figure
   nsnr    = len(snr)
   nftype  = len(ftype)
   fig, ax = plt.subplots(nrows=nsnr,ncols=nftype,sharex=True,sharey=True,figsize=(15,8))
   plt.subplots_adjust(left=0.015, bottom=0.06, right=0.995, top=0.97, wspace=0.0, hspace=0.0)
   
   for i in range(nftype):

        rootname = "testcases_"+case+"-"+ftype[i]
        filename = outdir+rootname+"/"+rootname+"_results.hdf5"
        f        = h5py.File(filename,'r')
        xvel     = np.array(f['in/xvel'])
        wave_obs = np.exp(np.array(f['in/wave_obs']))
        mask     = np.array(f['in/mask'])

        for j in range(nsnr):


            spec_obs = np.array(f['in/spec_obs'])[:,id[j]]
            bestfit  = np.array(f['out/'+str(id[j])+'/bestfit'])
            poly     = np.array(f['out/'+str(id[j])+'/poly'])+1.0
            losvd    = np.array(f['out/'+str(id[j])+'/losvd'])

            mx  = 1.1*np.amax(spec_obs)
            mn0 = 0.7*np.amin(spec_obs)
            mx  = 3.0
            mn0 = 0.45


            losvd_3s_lo  = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
            losvd_3s_hi  = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
            losvd_1s_lo  = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
            losvd_1s_hi  = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
            losvd_med    = losvd[2,:]/np.trapz(losvd[2,:],-xvel)

            res = spec_obs - bestfit[2,:] + mn0 + 0.1

            # Spectral fits plots ---------
            # ax[j,i].fill_between(wave_obs[mask],poly[1,mask],poly[3,mask], facecolor='yellow',zorder=0, alpha=0.50,label="Leg. polynomial")
            # ax[j,i].plot(wave_obs[mask],poly[1,mask], color='gray',linestyle='--', linewidth=1,zorder=0)
            # ax[j,i].plot(wave_obs[mask],poly[3,mask],color='gray',linestyle='--', linewidth=1,zorder=0)
            ax[j,i].plot(wave_obs[mask],spec_obs[mask],'k', zorder=1,label="Obs. data", ds='steps-mid')
            ax[j,i].fill_between(wave_obs[mask],bestfit[1,mask],bestfit[3,mask], facecolor='orange',zorder=2, alpha=0.75, step='mid')
            ax[j,i].plot(wave_obs[mask],bestfit[2,mask],color='red',zorder=3, label="Bestfit", ds='steps-mid')
            ax[j,i].plot(wave_obs[mask], res[mask], color='green', label="Residuals",ds='steps-mid')
            ax[j,i].set_ylim([mn0,mx])
            ax[j,i].axhline(y=mn0+0.1,color='k', linestyle='--')
            ax[j,i].set_yticks([])

            if j == nsnr-1:
               ax[j,i].set_xlabel("Wavelength ($\\mathrm{\\AA}$)")
            if j == 0:
               ax[j,i].set_title(ftype[i],fontweight='bold')   
            if i == 0:
               ax[j,i].set_ylabel("S/N="+str(snr[j]),fontweight='bold')    

            # LOSVD plots ---------
            axins = inset_axes(ax[j,i], width="50%", height="40%")
            axins.set_yticks([])
            axins.tick_params(axis='x', labelsize=8)

            axins.fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
            axins.fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
            axins.plot(xvel,losvd_med,'k.-', ds='steps-mid')
            axins.plot(tab['col1'], tab['col2'],'r.-', ds='steps-mid') 
            axins.axvline(x=0.0, color='k', linestyle="--", lw=1)
            axins.set_ylim([0.0,0.004])

   f.close()
  
#    plt.show()   
   fig.savefig("Figures/bayes-losvd_specfits_"+case+"_test.pdf",dpi=600)

   return

#==============================================================================
if (__name__ == '__main__'):

   idx = 10
   idx = 8
   # plot_figure("Gaussian",id=idx + np.arange(5)*25)
   # plot_figure("Double_Gaussian",id=idx + np.arange(5)*25)
   plot_figure("Marie1",id=idx + np.arange(5)*25)
   # plot_figure("Marie2",id=idx + np.arange(5)*25)
   # plot_figure("Wings",id=idx + np.arange(5)*25)

