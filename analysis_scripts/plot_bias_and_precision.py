import os
import sys
import glob
import h5py
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
from   compute_bias_precision  import compute_bias_precision
from   astropy.io        import ascii
from   astropy.table     import Table
from chainconsumer import ChainConsumer
#==============================================================================
if (__name__ == '__main__'):

   # Loading file list
#    dir = "../results_denso/"
   dir = "../results_deimos/"
   filelist = glob.glob(dir+"testcase_*/testcase_*_results.hdf5")

   # Loading files with results
   print("# Loading results")
   tab = compute_bias_precision(filelist)

   idx = ~np.isnan(tab['S/N'])
   tab = tab[idx]

#    idx = (tab['B-spline order'] == 2.0)
#    tab = tab[idx]

#    idx = (tab['case'] == 'Marie2')
#    tab = tab[idx]
      
 
#    pars   = ['S/N','B-spline order','Model','Fit type']
#    plt.tricontourf(np.log10(tab['S/N']),tab['B-spline order'],tab['Bias'], alpha=0.5)

#    sns.pairplot(tab.to_pandas())
#    plt.show()
#    exit()    

#    # Plotting results
   print("# Plotting results")
   colors  = ['red','blue','yellow','green','brown','cyan']
   cmaps = ['Reds','Blues','Oranges','Greens','Purples']
   pars   = ['S/N','B-spline order','Model','Fit type']
   bias_lims = [-0.039,0.0025]
   prec_lims = [0.0,3.99]

   # ------------
   fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(9,6))
   ax = ax.ravel()
   plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
 
   for i in range(len(pars)):
       upar = np.unique(tab[pars[i]])
       for j in range(len(upar)):
          idx = (tab[pars[i]] == upar[j])
          ax[i].tricontourf(tab['Bias'][idx],tab['Precision'][idx],i*np.ones(np.sum(idx)),colors=colors[j], alpha=0.5, linestyles='solid')
          ax[i].scatter(tab['Bias'][idx],tab['Precision'][idx], c=colors[j], alpha=0.5, edgecolors='k', label=str(upar[j]), s=16.0)
        #   sns.kdeplot(tab['Bias'][idx],tab['Precision'][idx],cmap=cmaps[j], shade=False, shade_lowest=False, n_levels=5, ax=ax[i], legend=True,label=str(upar[j]))
       ax[i].legend(title=pars[i], loc=2)
       if i >= 2:
          ax[i].set_xlabel('Bias')
       if (i == 0) or (i == 2):
          ax[i].set_ylabel('Precision')
       else:
           ax[i].set_ylabel('')   
       ax[i].set_xlim(bias_lims)
       ax[i].set_ylim(prec_lims)
       ax[i].axhline(y=1.0, color='k',linestyle=':')
       ax[i].axvline(x=0.0, color='k',linestyle=':')
   
#    plt.show()
   fig.savefig("Figures/bias_vs_precision.png", dpi=300)

   # ------------
   fig, ax = plt.subplots(nrows=2, ncols=len(pars), sharex=False, sharey=False, figsize=(12,6))
   ax = ax.ravel()
   plt.subplots_adjust(left=0.07, bottom=0.08, right=0.97, top=0.99, wspace=0.0, hspace=0.0)

   for i in range(len(pars)):
       values = []
       upar = np.unique(tab[pars[i]])
       for j in range(len(upar)):
          idx = (tab[pars[i]] == upar[j]) 
          values.append(list(tab['Bias'][idx]))  
       box = ax[i].boxplot(values,patch_artist=True, labels=upar)
       ax[i].set_ylim(bias_lims)
       ax[i].axhline(y=0.0, color='gray', linestyle=':')
       if i != 0:
           ax[i].set_yticks([])
       else:
           ax[0].set_ylabel('Bias')    
       for patch, color in zip(box['boxes'], colors):
           patch.set_facecolor('papayawhip')


   for i in range(len(pars)):
       values = []
       upar = np.unique(tab[pars[i]])
       for j in range(len(upar)):
          idx = (tab[pars[i]] == upar[j]) 
          values.append(list(tab['Precision'][idx]))  
       box = ax[i+len(pars)].boxplot(values,patch_artist=True, labels=upar)
       ax[i+len(pars)].set_xlabel(pars[i])
       ax[i+len(pars)].set_ylim(prec_lims)
       ax[i+len(pars)].axhline(y=1.0, color='gray', linestyle=':')
       for patch, color in zip(box['boxes'], colors):
           patch.set_facecolor('papayawhip')

       if i != 0:
           ax[i+len(pars)].set_yticks([])
       else:
           ax[len(pars)].set_ylabel('Precision')    



#    plt.show()
   fig.savefig("Figures/bias_and_precision_trends.png", dpi=300)
 
   