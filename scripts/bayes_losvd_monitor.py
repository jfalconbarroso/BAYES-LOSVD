import os
import glob
import time
import optparse
import numpy              as np
import matplotlib.pyplot  as plt
import lib.misc_functions as misc
from  astropy.io          import ascii
#==============================================================================
def monitor_progress(runname, idx):

   list = glob.glob("../results/"+runname+"/"+runname+"_progress_bin"+str(idx)+"*.csv")
   nlist = len(list)
   if (nlist == 0):
      misc.printFAILED("Cannot find any progress files for "+runname+" - Bin: "+str(idx))
      exit()

   print("- Checking...")

   fig, ax = plt.subplots(nrows=7,ncols=1, sharex=True, figsize=(12,15))
   ax = ax.ravel()
   fig.subplots_adjust(hspace=0.0,left=0.1,right=0.95,bottom=0.05,top=0.95)   
   alpha = 0.5

   params = ['lp',
             'accept',
             'stepsize',
             'treedepth',
             'n_leap',
             'diver.',
             'energy']
   
   npar = len(params)
   ax[6].set_xlabel("Iterations")    
   for i in range(npar):
       ax[i].set_ylabel(params[i])
   
   for infile in list:
       
       tab = ascii.read(infile)

       ax[0].semilogy(-1.0*tab['lp__'],alpha=alpha)
       ax[1].plot(tab['accept_stat__'],alpha=alpha)
       ax[2].semilogy(tab['stepsize__'],alpha=alpha)
       ax[3].plot(tab['treedepth__'],alpha=alpha)       
       ax[4].semilogy(tab['n_leapfrog__'],alpha=alpha)
       ax[5].plot(tab['divergent__'],alpha=alpha)
       ax[6].semilogy(tab['energy__'],alpha=alpha)
          
   fig.savefig("../results/"+runname+"/"+runname+"_progress_bin"+str(idx)+".png")
   plt.close('all')

   return
#==============================================================================
if (__name__ == '__main__'):

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -r <runname> -b <bin number>")
    parser.add_option("-r", "--runname", dest="runname",   type="string", default=None,  help="Run name of the preproc file")    
    parser.add_option("-b", "--bin",     dest="bin",       type="string", default="all", help="Bin ID for spectrum run [all,odd,even,bin_list]")
    parser.add_option("-i", "--inter",   dest="interval",  type="int",    default=5,     help="Time interval (in seconds) for checking")    

    (options, args) = parser.parse_args()
    runname  = options.runname
    interval = options.interval
    bin      = options.bin

    # First quick run     
    monitor_progress(runname,bin)

    # Now running on intervals     
    t0 = time.time()
    while True:
        dif = time.sleep(interval - ((time.time() - t0) % interval))
        if (dif == None):
           monitor_progress(runname,bin)
           
           
