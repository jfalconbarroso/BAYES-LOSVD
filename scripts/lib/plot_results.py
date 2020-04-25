import os
import h5py
# import corner
import numpy             as np
import matplotlib.pyplot as plt
import lib.misc_functions    as misc
#==============================================================================
def plot_bestfit(pdf_pages,i,data_struct,bestfit,losvd,poly):

   # Reading the results
   f = h5py.File(filename,"r")
   wave_obs  = np.exp(np.array(f['wave_obs']))
   spec_obs  = np.array(f['spec_obs'])
   sigma_obs = np.array(f['sigma_obs'])
   wave_pca  = np.exp(np.array(f['wave_pca']))
   xvel      = np.array(f['xvel'])
   bestfit   = np.array(f['bestfit'])
   losvd     = np.array(f['losvd'])
   mask      = np.array(f['mask'])-1
   velscale  = np.array(f['velscale'])
   poly      = np.array(f['poly'])+1.0
      
   # Computing some statistics
   bfit_low,  bfit_med,  bfit_high  = misc.get_percentiles(bestfit.T)
   losvd_low, losvd_med, losvd_high = misc.get_percentiles(losvd.T)
   poly_low,  poly_med,  poly_high  = misc.get_percentiles(poly.T)
   
   # Making plot ----------------------------------------------------------
   fig = plt.figure(figsize=(10,7))
   fig.suptitle(filename, fontsize=14, fontweight='bold')

   plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.925, wspace=0.0, hspace=0.3)
   ax0  = plt.subplot2grid((2,4),(0,0), colspan=1)
   ax1  = plt.subplot2grid((2,4),(0,2), colspan=2)
   ax2  = plt.subplot2grid((2,4),(1,0), colspan=4)

   # Deleting ax0 axis
   ax0.axis('off')
   ax0.set_xlim([0,1])
   ax0.set_ylim([0,1])


   # LOSVD -----------
   ax1.fill_between(xvel,losvd_low,losvd_high, color='blue', alpha=0.25)
   ax1.plot(xvel,losvd_low,color='blue')
   ax1.plot(xvel,losvd_high,color='blue')
   ax1.plot(xvel,np.mean(losvd,axis=0),'k.-')
   ax1.axhline(y=0.0,color='k', linestyle='--')
   ax1.set_xlabel("Velocity (km s$^{-1}$)")

   # Spectral fit
   mx  = 1.1*np.amax(spec_obs[:,0])
   mn0 = 0.7*np.amin(spec_obs[:,0])
   ax2.fill_between(wave_pca,poly_low,poly_high, facecolor='yellow',zorder=0, alpha=0.5)
   ax2.plot(wave_pca,poly_low, color='gray',linestyle='--', linewidth=1,zorder=0)
   ax2.plot(wave_pca,poly_high,color='gray',linestyle='--', linewidth=1,zorder=0)
   ax2.plot(wave_obs,spec_obs[:,0],'k', zorder=1)
   ax2.fill_between(wave_pca,bfit_low,bfit_high, facecolor='orange',zorder=2, alpha=0.5)
   ax2.plot(wave_pca,bfit_med,color='red',zorder=3)
   res = spec_obs[mask,0] - bfit_med[mask] + mn0 + 0.1
   ax2.plot(wave_obs[mask], res, color='green')
   mn = 0.8*np.amin(res)*0.0
   ax2.set_ylim([mn,mx])
   ax2.axhline(y=mn0+0.1,color='k', linestyle='--')
   ax2.axvline(x=wave_obs[mask[0]], color='k', linestyle=":")
   ax2.axvline(x=wave_obs[mask[-1]], color='k', linestyle=":")
   
   ax2.plot(wave_obs,sigma_obs[:,0],'gray')
   ax2.set_ylabel("Norm. flux")
   ax2.set_xlabel("Wavelength ($\mathrm{\AA}$)")
   
   # Saving the figure on disk
   tmpname  = os.path.basename(filename)
   rootname = os.path.splitext(tmpname)[0]
   outpdf   = "../figures/"+rootname+"_results.pdf"
   if os.path.exists(outpdf):
      os.remove(outpdf)
   fig.savefig(outpdf)

   #plt.show()

   return

#==============================================================================
def plot_chains(pdf,idx,losvd_samples,weights_samples,sigma_los,coefs):

   # Plotting the MCMC chains
   fig = plt.figure(figsize=(8,7))
   plt.subplots_adjust(left=0.10, bottom=0.07, right=0.98, top=0.95, wspace=0.0, hspace=0.3)
   ax1 = plt.subplot2grid((4,1),(0,0))
   ax2 = plt.subplot2grid((4,1),(1,0))
   ax3 = plt.subplot2grid((4,1),(2,0))
   ax4 = plt.subplot2grid((4,1),(3,0))
   
   fig.suptitle("BinID: "+str(idx), fontsize=14, fontweight='bold')

   ax1.plot(losvd_samples)
   ax1.set_ylabel("LOSVD")
   ax2.plot(sigma_los)
   ax2.set_ylabel("$\sigma_{LOSVD}$")
   ax3.plot(weights_samples)
   ax3.set_ylabel("Template Weights")
   ax4.plot(coefs)
   ax4.set_ylabel("Coefs.")
   ax4.set_xlabel("Iteration")
      
   pdf.savefig(fig)
   
   return

#==============================================================================
def plot_sampler_params(pdf,idx,accept_stat,stepsize,treedepth,n_leapfrog,divergent,energy):

   niter       = energy.shape[0]
   
   # Plotting diagnostics
   fig, ax = plt.subplots(nrows=6,ncols=1, sharex=True, figsize=(8,7))
   ax = ax.ravel()
   fig.subplots_adjust(left=0.10, bottom=0.07, right=0.98, top=0.95, wspace=0.0, hspace=0.3)

   alpha = 1.0

   fig.suptitle("BinID: "+str(idx), fontsize=14, fontweight='bold')

   ax[0].plot(accept_stat)
   ax[0].set_ylabel("Accept_stat")
   ax[1].semilogy(stepsize)
   ax[1].set_ylabel("Stepsize")
   ax[2].plot(treedepth)
   ax[2].set_ylabel("Treedepth")
   ax[3].semilogy(n_leapfrog)
   ax[3].set_ylabel("N. leapfrog")
   ax[4].plot(divergent)
   ax[4].set_ylabel("Divergent")
   ax[5].semilogy(energy)
   ax[5].set_ylabel("Energy")
   ax[5].set_xlabel("Iteration")
   ax[5].set_xlim([0,niter])
 
   for i in range(6):
      ax[i].axvspan(0, niter*0.5,       alpha=0.05, color='red')
      ax[i].axvspan(0.5*niter+1, niter, alpha=0.05, color='green')
      ax[i].axvline(x=0.5*niter, color='k',linestyle=':')

   pdf.savefig(fig)
   
   return
#==============================================================================
#def plot_corner(pdf,idx,samples, nvel, ncoefs, nweights):
def plot_corner(idx=3):
    
    
   f = h5py.File("kk.hdf5",'r')
   losvd = f['losvd']
   sigma = f['sigma_los']
   coefs = f['coefs']
   wei   = f['weights']
    
   nvel = losvd.shape[1]
   ncoefs = coefs.shape[1]
   nweights = wei.shape[1]
    
   # samples_corner = np.hstack((losvd,sigma,coefs,wei))
   
   samples_corner = np.hstack((sigma,coefs,wei))

   labels = []   
   #for i in range(nvel):
       #labels = np.append(labels, 'LOS$_{'+str(i)+'}$')
   labels = np.append(labels,r'$\sigma_{LOS}$')
   for i in range(ncoefs):    
       labels = np.append(labels, 'C$_{'+str(i)+'}$')
   #for i in range(nweights):    
       #labels = np.append(labels, 'W$_{'+str(i)+'}$')
 
   npar = len(labels)
   fig, ax = plt.subplots(nrows=npar,ncols=npar, figsize=(10,10))
   fig.suptitle("BinID: "+str(idx), fontsize=14, fontweight='bold')
   fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.925)

   
   corner.corner(samples_corner, labels=labels, show_titles=True, fig=fig)
   
   plt.show() 
   #pdf.savefig(fig)
   
   return
#==============================================================================
if (__name__ == '__main__'):

   plot_corner()
