import arviz                 as az
import numpy                 as np
import matplotlib.pyplot     as plt
from   matplotlib.backends.backend_pdf import PdfPages
#==============================================================================
def plot_sampler_params(idx,accept_stat,stepsize,treedepth,n_leapfrog,divergent,energy):

   niter = energy.shape[0]
   
   # Plotting diagnostics
   fig, ax = plt.subplots(nrows=6,ncols=1, sharex=True, figsize=(8,7))
   ax = ax.ravel()
   fig.subplots_adjust(left=0.10, bottom=0.07, right=0.98, top=0.95, wspace=0.0, hspace=0.3)

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
   
   return

#===============================================================================
def create_diagnostic_plots(pdf_filename,samples):

    # Converting the Stan FIT object to Arviz InfereceData
    data      = az.from_pystan(samples)
    tmp       = data.posterior
    var_names = list(tmp.data_vars)

    # Filtering the list of parameters to plot
    unwanted  = {'losvd','spec','conv_spec','poly','bestfit','losvd_','losvd_mod','spec_pred','log_likelihood'}
    vars_main = [e for e in var_names if e not in unwanted]

    pdf_pages = PdfPages(pdf_filename)

    print(" - Chains")
    az.plot_trace(data, var_names=vars_main, divergences=True)
    pdf_pages.savefig()

    print(" - Pair plot")
    az.plot_pair(data,var_names=vars_main,kind='kde',divergences=True)
    pdf_pages.savefig()

    print(" - Autocorr plot")
    az.plot_autocorr(data, var_names=vars_main)
    pdf_pages.savefig()

    print(" - Energy plot")
    az.plot_energy(data)
    pdf_pages.savefig()

    pdf_pages.close()   
    return
