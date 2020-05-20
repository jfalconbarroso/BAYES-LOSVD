import os
import sys
import glob
import h5py
import pickle
import pystan
import optparse
import threading
import warnings
import traceback
import arviz                        as az
import numpy                        as np
import matplotlib.pyplot            as plt
import lib.misc_functions           as misc
from   lib.create_diagnostic_plots  import create_diagnostic_plots 
from   hashlib                      import md5
from   multiprocessing              import Queue, Process, cpu_count
#==============================================================================
def worker(inQueue, outQueue):

    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for i, bin_list, runname, niter, nchain, adapt_delta, max_treedepth, verbose, save_chains, save_plots in iter(inQueue.get, 'STOP'):

        status = run(i, bin_list, runname, niter, nchain, adapt_delta, max_treedepth, verbose, save_chains, save_plots)

        outQueue.put(( status ))

#==============================================================================
def stan_cache(model_code, model_name=None, codefile=None, **kwargs):

    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'stan_model/cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'stan_model/cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel for "+codefile)

    return sm

#==============================================================================
def run(i, bin_list, runname, niter, nchain, adapt_delta, max_treedepth, verbose, save_chains, save_plots):

    idx    = bin_list[i]
    stridx = str(idx)
    misc.printRUNNING(runname+" - Bin: "+stridx) 

    try:
    
        # Checking the desired bin exists
        input_file = "../results/"+runname+"/"+runname+"_results.hdf5"

        struct    = h5py.File(input_file,'r')
        check_bin = struct.get('out/'+stridx)
        if check_bin == None:
           misc.printFAILED("Bin "+stridx+" does not exist in "+input_file)
           return 'ERROR'

        # Defining the version of the code to use
        codefile = 'stan_model/bayes-losvd_ghfit.stan'
        if not os.path.exists(codefile):
           misc.printFAILED(codefile+" does not exist.")
           exit()

        # Defining output names and directories
        outdir           = "../results/"+runname
        pdf_filename     = outdir+"/"+runname+"_gh_diagnostics_bin"+stridx+".pdf"
        summary_filename = outdir+"/"+runname+"_gh_Stan_summary_bin"+stridx+".txt"
        arviz_filename   = outdir+"/"+rootname+"_gh_chains_bin"+str(idx)+".netcdf"
        # chains_filename  = outdir+"/"+runname+"_gh_chains_bin"+stridx+".hdf5"
        sample_filename  = outdir+"/"+runname+"_gh_progress_bin"+stridx+".csv"
        outhdf5          = outdir+"/"+runname+"_gh_results_bin"+stridx+".hdf5"

        # Creating the structure with the data for Stan
        losvd       = struct['out/'+stridx+'/losvd'][2,:]
        sigma       = np.zeros((len(losvd),2))
        sigma[:,0]  = np.fabs(struct['out/'+stridx+'/losvd'][1,:]-losvd)
        sigma[:,1]  = np.fabs(struct['out/'+stridx+'/losvd'][3,:]-losvd)
        sigma_losvd = np.mean(sigma,axis=1)

        data = {'nvel':        struct['in/nvel'], 
                'xvel':        struct['in/xvel'],
                'losvd_obs':   losvd,
                'sigma_losvd': sigma_losvd
                }

        # Running the model
        with open(codefile, 'r') as myfile:
           code   = myfile.read()
        model     = stan_cache(model_code=code, codefile=codefile) 
        fit       = model.sampling(data=data, iter=niter, chains=nchain, 
                                   control={'adapt_delta':adapt_delta, 'max_treedepth':max_treedepth},
                                   sample_file=sample_filename, check_hmc_diagnostics=True)
        samples   = fit.extract(permuted=True)
        diag_pars = fit.get_sampler_params()
    
        # If requested, saving sample chains
        if (save_chains == True):
           print("")
           print("# Saving chains in Arviz (NETCDF) format: "+arviz_filename) 
           arviz_data = az.from_pystan(posterior=fit)
  
        # Saving Stan's summary of main parameters on disk
        print("")
        print("# Saving Stan summary: "+summary_filename)         
        unwanted  = {'losvd_mod'}
        misc.save_stan_summary(fit, unwanted=unwanted, verbose=verbose,summary_filename=summary_filename)

        # Processing output and saving results
        print("")
        print("# Processing and saving results: "+outhdf5)
        misc.process_stan_output_hdp(struct,samples,outhdf5,stridx)

        # Creating diagnostic plots
        if (save_plots == True):
            if os.path.exists(pdf_filename):
              os.remove(pdf_filename)    
            print("")
            print("# Saving diagnostic plots: "+pdf_filename) 
            create_diagnostic_plots(idx, pdf_filename, fit, diag_pars, niter, nchain)

         # Removing progess files
        print("")
        print("# Deleting progress files")
        misc.delete_files(sample_filename,'csv')
        misc.delete_files(sample_filename,'png')

        # If we are here, we are DONE!
        struct.close()
        misc.printDONE(runname+" - Bin: "+stridx)

        return 'OK'

    except:
    
        misc.printFAILED()
        traceback.print_exc()            
          
        return 'ERROR'

#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("               BAYES-LOSVD                 ")
    print("                (gh_fit)                   ")
    print("===========================================")
    print("")

    # Capturing the command line arguments
    parser = optparse.OptionParser(usage="%prog -f file")
    parser.add_option("-r", "--runname",       dest="runname",       type="string", default="",   help="Runname of the preprocessed file")
    parser.add_option("-i", "--niter",         dest="niter",         type="int",    default=1000, help="Number of iterations in stan")
    parser.add_option("-c", "--nchain",        dest="nchain",        type="int",    default=1,    help="Number of simultaneous chains to run")
    parser.add_option("-a", "--adapt_delta",   dest="adapt_delta",   type="float",  default=0.99, help="Stan Adapt_delta")
    parser.add_option("-m", "--max_treedepth", dest="max_treedepth", type="int",    default=15,   help="Stan maximum tree depth")
    parser.add_option("-b", "--bin",           dest="bin",           type="string", default="-1", help="Bin ID for single spectrum run")
    parser.add_option("-n", "--njobs",         dest="njobs",         type="int",    default=1,    help="Number of jobs to lauch in parallel")
    parser.add_option("-v", "--verbose",       dest="verbose",       type="int",    default=0,    help="Printing Stan summary for each fit")
    parser.add_option("-s", "--save_chains",   dest="save_chains",   type="int",    default=0,     help="Saving chains for each fit (Default: 0/False)")
    parser.add_option("-p", "--save_plots",    dest="save_plots",    type="int",    default=0,     help="Saving diagnistic plots (Default: 0/False)")

    (options, args) = parser.parse_args()
    runname         = options.runname
    niter           = options.niter
    nchain          = options.nchain
    adapt_delta     = options.adapt_delta
    max_treedepth   = options.max_treedepth
    bin             = options.bin
    njobs           = options.njobs
    verbose         = options.verbose
    save_chains     = options.save_chains
    save_plots      = options.save_plots

    if (verbose == 0):
        verbose = False
    else:
        verbose = True    

    if (save_chains == 0):
        save_chains = False
    else:
        save_chains = True    

    if (save_plots == 0):
        save_plots = False
    else:
        save_plots = True    

    # Checking the file exists
    results_file = "../results/"+runname+"/"+runname+"_results.hdf5"
    if not os.path.exists(results_file):
       misc.printFAILED(results_file+" does not exist.")
       exit()

    # Loading input information from the results file
    f     = h5py.File(results_file,'r')
    nbins = np.array(f['in/nbins'])
    f.close()
                        
     # Defining the list of bins to be analysed
    if (bin == "all"):
       bin_list = list(np.arange(nbins))
       print("# ENTIRE list of bins selected")
    elif (bin == "odd"):
       bin_list = list(np.arange(0,nbins,2)) 
       print("# ODD bins selected")
    elif (bin == "even"):
       bin_list = list(np.arange(1,nbins,2)) 
       print("# EVEN bins selected")
    else:
       dummy    = bin.split(",") 
       bin_list = list(np.array(dummy,dtype=int))
       nbins    = len(bin_list)
       print("# Selected bins: "+bin)
    
    # Managing the work PARALLEL or SERIAL accordingly
    if njobs*nchain > cpu_count():
        misc.printFAILED("ERROR: The chosen number of NJOBS and NCHAIN seems to be larger than the number of cores in the system!")
        exit()

    # Create Queues
    inQueue  = Queue()
    outQueue = Queue()

    # Create worker processes
    ps = [Process(target=worker, args=(inQueue, outQueue)) for _ in range(njobs)]

    # Start worker processes
    for p in ps: p.start()

    # Fill the queue
    for i in range(nbins):
        inQueue.put( ( i, bin_list, runname, niter, nchain, adapt_delta, 
                       max_treedepth, verbose, save_chains, save_plots) )

    # Now running the processes
    run_tmp = [outQueue.get() for _ in range(nbins)]

    # Send stop signal to stop iteration
    for _ in range(njobs): inQueue.put('STOP')

    # Stop processes
    for p in ps: p.join()

    # Pack all results into a single file if everything went OK
    if 'ERROR' not in run_tmp:
       print("")
       print("# Packing all results into a single HDF5 file.")
       misc.pack_results(runname, suffix='_gh')

    exit()
  
