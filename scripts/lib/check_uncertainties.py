import os
import glob
import h5py
import numpy             as np
import matplotlib.pyplot as plt
import misc_functions    as misc
#==============================================================================
def compute_intervals(ref,data):
    
    u  = np.unique(ref)
    nu = len(u)
    par_lo, par_med, par_hi = np.zeros(nu),np.zeros(nu),np.zeros(nu)
    for i in range(nu):
       idx = (ref == u[i])
       par_lo[i],par_med[i],par_hi[i]=np.percentile(res[idx],q=[34.0,50.0,86.0])
    
    return u, par_lo, par_med, par_hi
#==============================================================================
if (__name__ == '__main__'):

  list  = glob.glob("../results2/*.hdf5")
  nlist = len(list)
  
  snr, res, vel, sigma, h3, h4 = np.zeros(nlist), np.zeros(nlist), np.zeros(nlist),np.zeros(nlist),np.zeros(nlist),np.zeros(nlist)
  
  for i in range(nlist):
      
      #print(list[i])
      
      f        = h5py.File(list[i],"r")
      losvd    = np.array(f['losvd'])
      velscale = np.array(f['velscale'])
      
      # Extracting input parameters
      tmpname  = os.path.basename(list[i])
      rootname = os.path.splitext(tmpname)[0]      
      dummy    = str.split(rootname,'_')
      snr_str  = str.split(dummy[2],'SNR')
      pars     = np.array([dummy[4],dummy[5],dummy[6],dummy[7]], dtype=float)
      
      snr[i]   = float(snr_str[1])
      vel[i]   = pars[0]
      sigma[i] = pars[1]
      h3[i]    = pars[2]
      h4[i]    = pars[3]
      
      # Re-creating input LOSVD
      input_losvd, input_xvel, dummy = misc.create_gh_losvd(np.array(pars),velscale)
 
      # Extracting result LOSVD
      losvd_low, losvd_med, losvd_high = misc.get_percentiles(losvd.T)
   
      # Getting the STD of the residuals
      metric = np.abs(input_losvd-losvd_med)/(np.abs(input_losvd)+1)
      res[i] = np.sum(metric)
      
  
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8), sharey=True)
  ax = ax.ravel()
  plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.95, wspace=0.05, hspace=0.2)

  ylim0 = 0.0
  ylim1 = 0.4
  
  for i in range(4):
      ax[i].set_ylim([ylim0,ylim1])

  
  ax[0].plot(snr,res,'k.',alpha=0.15)
  u, par_lo, par_med, par_hi = compute_intervals(snr,res)
  ax[0].fill_between(u, par_lo, par_hi, color='blue',alpha=0.4)
  ax[0].plot(u,par_med,'k.-')
  ax[0].set_xlabel("SNR")
  ax[0].set_ylabel(r'$\sum_{j=0}^{nvel}$ $\vert$LOSVD$_{in,j}$ - LOSVD$_{out,j}$$\vert$ / (1 + $\vert$LOSVD$_{in,j}$$\vert$)')
  
  
  ax[1].plot(sigma,res,'k.',alpha=0.15)
  u, par_lo, par_med, par_hi = compute_intervals(sigma,res)
  ax[1].fill_between(u, par_lo, par_hi, color='blue',alpha=0.4)
  ax[1].plot(u,par_med,'k.-')
  ax[1].set_xlabel("$\sigma$")

  ax[2].plot(h3,res,'k.',alpha=0.15)
  u, par_lo, par_med, par_hi = compute_intervals(h3,res)
  ax[2].fill_between(u, par_lo, par_hi, color='blue',alpha=0.4)
  ax[2].plot(u,par_med,'k.-')
  ax[2].set_xlabel("h$_3$")
  ax[2].set_ylabel(r'$\sum_{j=0}^{nvel}$ $\vert$LOSVD$_{in,j}$ - LOSVD$_{out,j}$$\vert$ / (1 + $\vert$LOSVD$_{in,j}$$\vert$)')

  ax[3].plot(h4,res,'k.',alpha=0.15)
  u, par_lo, par_med, par_hi = compute_intervals(h4,res)
  ax[3].fill_between(u, par_lo, par_hi, color='blue',alpha=0.4)
  ax[3].plot(u,par_med,'k.-')
  ax[3].set_xlabel("h$_4$")

  plt.show()
