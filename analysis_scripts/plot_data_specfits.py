import h5py
from astropy.io          import ascii
import numpy             as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#==============================================================================
def plot_figure(case, dir="../results/", binID=[0], ftype=['S0','S1','A1','A2','A3','B3','B4','G0']):

   # Plotting figure
   nbin    = len(binID)
   nftype  = len(ftype)
   fig, ax = plt.subplots(nrows=nbin,ncols=nftype,sharex=True,sharey=True,figsize=(16,8))
   if nftype == 1:
       ax = np.broadcast_to(ax,(nftype,nbin)).T
   plt.subplots_adjust(left=0.015, bottom=0.06, right=0.995, top=0.97, wspace=0.0, hspace=0.0)
   
   for i in range(nftype):

        rootname = case+"-"+ftype[i]
        filename = dir+rootname+"/"+rootname+"_results.hdf5"
        f        = h5py.File(filename,'r')
        xvel     = np.array(f['in/xvel'])+60.0
        wave_obs = np.exp(np.array(f['in/wave_obs']))
        mask     = np.array(f['in/mask'])
      
        for j in range(nbin):

            spec_obs = np.array(f['in/spec_obs'])[:,binID[j]]
            bestfit  = np.array(f['out/'+str(binID[j])+'/bestfit'])
            # poly     = np.array(f['out/'+str(binID[j])+'/poly'])+1.0
            losvd    = np.array(f['out/'+str(binID[j])+'/losvd'])

            mx  = 1.1*np.amax(spec_obs)
            mn0 = 0.7*np.amin(spec_obs)
            mx  = 2.0
            mn0 = 0.45

            losvd_3s_lo  = losvd[0,:]/np.trapz(losvd[2,:],-xvel)
            losvd_3s_hi  = losvd[4,:]/np.trapz(losvd[2,:],-xvel)
            losvd_1s_lo  = losvd[1,:]/np.trapz(losvd[2,:],-xvel)
            losvd_1s_hi  = losvd[3,:]/np.trapz(losvd[2,:],-xvel)
            losvd_med    = losvd[2,:]/np.trapz(losvd[2,:],-xvel)

            res = spec_obs - bestfit[2,:] + mn0 + 0.1
            

            # PLOT FULL SPECTRA AND BESTFIT AND INDICATE MASK WITH VERTICAL SHADED AREAS

            # Spectral fits plots ---------
            # ax[j,i].fill_between(wave_obs[mask],poly[1,mask],poly[3,mask], facecolor='yellow',zorder=0, alpha=0.50,label="Leg. polynomial")
            # ax[j,i].plot(wave_obs[mask],poly[1,mask], color='gray',linestyle='--', linewidth=1,zorder=0)
            # ax[j,i].plot(wave_obs[mask],poly[3,mask],color='gray',linestyle='--', linewidth=1,zorder=0)
            ax[j,i].plot(wave_obs,spec_obs,'k', zorder=1,label="Obs. data")#, ds='steps-mid')
            ax[j,i].fill_between(wave_obs,bestfit[1,:],bestfit[3,:], facecolor='orange',zorder=2, alpha=0.75)#, step='mid')
            ax[j,i].plot(wave_obs,bestfit[2,:],color='red',zorder=3, label="Bestfit")#, ds='steps-mid')
            ax[j,i].plot(wave_obs, res, color='green', label="Residuals")#,ds='steps-mid')
            ax[j,i].set_ylim([mn0,mx])
            ax[j,i].axhline(y=mn0+0.1,color='k', linestyle='--')
            ax[j,i].set_yticks([])

            if j == nbin-1:
               ax[j,i].set_xlabel("Wavelength ($\\mathrm{\\AA}$)")
            if j == 0:
               ax[j,i].set_title(ftype[i],fontweight='bold')   
            if i == 0:
               ax[j,i].set_ylabel("BinID="+str(binID[j]),fontweight='bold')    

            w = np.flatnonzero(np.diff(mask) > 1)
            if w.size > 0:
                for wj in w:
                  l0 = wave_obs[mask[wj]]
                  l1 = wave_obs[mask[wj+1]]
                  ax[j,i].axvspan(l0,l1, alpha=0.25, color='gray')  

            
            # LOSVD plots ---------
            axins = inset_axes(ax[j,i], width="50%", height="40%")
            axins.set_yticks([])
            axins.tick_params(axis='x', labelsize=8)

            axins.fill_between(xvel,losvd_3s_lo,losvd_3s_hi, color='blue', alpha=0.15, step='mid')
            axins.fill_between(xvel,losvd_1s_lo,losvd_1s_hi, color='blue', alpha=0.50, step='mid')
            axins.plot(xvel,losvd_med,'k.-', ds='steps-mid')
            axins.axvline(x=0.0, color='k', linestyle="--", lw=1)
            axins.set_ylim([0.0,0.005])

   f.close()
  
   fig.savefig("Figures/"+case+"_specfits.pdf",dpi=300)

   return

#==============================================================================
if (__name__ == '__main__'):

   plot_figure("NGC4550_SAURON", dir="../results/", binID=[0,169,189], ftype=['S0','S1','A1','A2','A3','B3','B4','G0'])

