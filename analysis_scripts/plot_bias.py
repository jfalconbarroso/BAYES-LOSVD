import os
import sys
import glob
import h5py
import numpy             as np
import matplotlib.pyplot as plt
from   astropy.io        import ascii
from   astropy.table     import Table
from chainconsumer import ChainConsumer
#==============================================================================
def load_results(filelist):

    xvel_lims = [-500.0,500.0]

    nfiles = len(filelist)
    print(" - "+str(nfiles)+" files found")

    struct = {'case':list(np.repeat('',nfiles)), 'B-spline order':np.repeat(np.nan,nfiles), 
              'S/N':np.repeat(np.nan,nfiles),'Fit type':list(np.repeat('',nfiles)), 'Model':list(np.repeat('',nfiles)), 
              'Bias':np.repeat(np.nan,nfiles),'Precision':np.repeat(np.nan,nfiles)}

    for i in range(nfiles):

        bname  = os.path.basename(filelist[i])
        case   = bname.split('_')[1]
        snr    = bname.split('SNR')[1].split('-')[0]
        border = bname.split('Border')[1].split('_')[0]
        ftype  = bname.split('_')[-2]
        ssp    = bname.split('_')[2]

        if case == 'Double':
            case = 'Double_Gaussian'
            ssp = bname.split('_')[3]

        if ftype == 'penalised':
            ftype = 'RW'
        else:
            ftype = 'Free'        

        tab = ascii.read("../losvd/"+case+"_velscale50.dat")
        vel = tab['col1']
        tmp_losvd = tab['col2'] / np.sum(tab['col2'])   


        f         = h5py.File(filelist[i],'r')
        xvel      = np.array(f['in/xvel'])
        true_losvd = np.interp(xvel, vel, tmp_losvd)
        good      = (xvel >= xvel_lims[0]) & (xvel <= xvel_lims[1])
        losvd_med = np.array(f['out/0/losvd'][2,:])
        losvd_med /= np.sum(losvd_med)
        diff      = losvd_med - true_losvd
        error     = 0.5*(f['out/0/losvd'][3,:]-f['out/0/losvd'][1,:])

        struct['B-spline order'][i] = border
        struct['case'][i]      = case
        struct['S/N'][i]       = snr
        struct['Fit type'][i]  = ftype
        struct['Model'][i]     = ssp
        struct['Bias'][i]      = np.mean(diff[good])/np.amax(true_losvd) 
        # struct['Bias'][i]      = np.mean(diff[good])/np.mean(true_losvd[good]) 
        # struct['Bias'][i]      = np.mean(diff[good]/true_losvd[good]) 
        struct['Precision'][i] = np.std(diff[good])/np.mean(error[good]) 

        f.close()       

        # plt.fill_between(xvel,f['out/0/losvd'][1,:]/np.sum(f['out/0/losvd'][2,:]),f['out/0/losvd'][3,:]/np.sum(f['out/0/losvd'][2,:]),alpha=0.5, step='mid')
        # plt.plot(xvel, losvd_med, 'k', ds='steps-mid')
        # plt.plot(xvel, true_losvd,'r', ds='steps-mid')
        # plt.title(case+' '+str(snr)+' '+str(border)+' '+ftype+' '+ssp)

        # plt.plot(xvel, diff/true_losvd, 'gray', ds='steps-mid', linestyle='--')
        # plt.axhline(y=np.mean(np.fabs(residual)/error), color='black', linestyle=':')
        # plt.ylim([-1,5])
 
        # plt.show()
        # exit()

    return Table(struct)

#==============================================================================
if (__name__ == '__main__'):

   # Loading file list
   dir = "../results_denso/"
   filelist = glob.glob(dir+"testcase_*/testcase_*_results.hdf5")

   # Loading files with results
   tab = load_results(filelist)

#    idx = (tab['B-spline order'] == 2.0)
#    tab = tab[idx]

#    idx = (tab['case'] == 'Marie2')
#    tab = tab[idx]
      

#    # Plotting results
   colors = ['red','blue','brown','green','cyan','yellow']
   pars   = ['S/N','B-spline order','Model','Fit type']
   bias_lims = [-0.11,0.015]
   prec_lims = [0.0,3.99]

   # ------------
   fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(9,6))
   ax = ax.ravel()
   plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
 
   for i in range(len(pars)):
       upar = np.unique(tab[pars[i]])
       for j in range(len(upar)):
          idx = (tab[pars[i]] == upar[j])
          ax[i].tricontourf(tab['Bias'][idx],tab['Precision'][idx],i*np.ones(np.sum(idx)),colors=colors[j], alpha=0.5)
          ax[i].scatter(tab['Bias'][idx],tab['Precision'][idx], c=colors[j], alpha=0.5, edgecolors='k', label=str(upar[j]))
       ax[i].legend(title=pars[i], loc=2)
       if i >= 2:
          ax[i].set_xlabel('Bias')
       if (i == 0) or (i == 2):
          ax[i].set_ylabel('Precision')
       ax[i].set_xlim(bias_lims)
       ax[i].set_ylim(prec_lims)
       ax[i].axhline(y=1.0, color='k',linestyle=':')
       ax[i].axvline(x=0.0, color='k',linestyle=':')
   
#    plt.show()
   fig.savefig("bias_vs_precision.png", dpi=300)

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
   fig.savefig("bias_and_precision_trends.png", dpi=300)
 
   