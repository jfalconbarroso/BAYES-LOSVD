import os
import sys
import jax
import h5py
import argparse
import warnings
import traceback
import importlib
import jax.numpy          as jnp
import arviz              as az
import numpy              as np
import lib.misc_functions as misc
import multiprocessing    as mp
from   numpyro.infer         import MCMC, NUTS
from   lib.gauss_hermite_fit import gauss_hermite_fit
# from   lib.lmoments_fit      import lmoments_fit
mp.set_start_method("spawn", force=True)
#==============================================================================
def run_fit(runname, preproc_file, bin_list, mask, porder, outdir, nsamples, nchain, njobs, save_chains, fit_type, extra_params):

    with mp.Pool(processes=njobs) as pool:
           results = pool.starmap(
               run_model,
               [(runname, preproc_file, i, mask, porder, outdir, nsamples, nchain, save_chains, fit_type, extra_params) for i in bin_list]
           )

    return 'OK'

#------------------------------------------------------------------------------
def run_model(runname, preproc_file, idx, mask, porder, outdir, nsamples, nchain, save_chains, fit_type, extra_params):

    misc.printRUNNING(runname+" - Fit type: "+fit_type+" - Bin: "+str(idx)) 

    try:

        # Creating the basic structure with the data for Numpyro model
        if extra_params == None:
            extra_params = {}
        with h5py.File(preproc_file, "r") as struct:
            data = {
                    'xvel':          np.array(struct['in/xvel']),
                    'spec_obs':      np.array(struct['in/spec_obs'][:,idx]),
                    'sigma_obs':     np.array(struct['in/sigma_obs'][:,idx]),
                    'mean_template': np.array(struct['in/mean_template']),
                    'templates':     np.array(struct['in/templates']),
                    'snr':           np.array(struct['in/bin_snr'][idx]),
                    'nbins':         np.array(struct['in/nbins']),
                    'npca':          np.array(struct['in/npca']),
                    'porder':        porder,
                    'mask':          mask,
                    'params':        extra_params
            }
        struct.close()
        
        # Loading the model
        print("# Loading the model")
        module_name = misc.read_code(fit_type)
        full_module_path = f"models.{module_name}"
        module = importlib.import_module(full_module_path)
        importlib.reload(module)
        model = getattr(module, module_name)  # function must match filename

        # Running the model
        print("")
        print("# Running the model")
        rng_key = jax.random.PRNGKey(0)
        nuts_kernel = NUTS(model, target_accept_prob=0.90, dense_mass=False, max_tree_depth=10)
        mcmc = MCMC(nuts_kernel, num_warmup=nsamples, num_samples=nsamples, num_chains=nchain, chain_method="sequential")
        mcmc.run(rng_key, data)

        # Saving results into arrays
        idata = az.from_numpyro(posterior=mcmc)
        samples = mcmc.get_samples(group_by_chain=True)
        losvd_samples = samples['losvd'].reshape(-1, samples['losvd'].shape[-1])

        # Gauss-Hermite fits
        idata = gauss_hermite_fit(data['xvel'], losvd_samples, idata, n_chains=nchain)

        # L-moments fits
        # idata = lmoments_fit(data['xvel'], losvd_samples, idata, n_chains=nchain)

        # Processing outputs
        print("")
        print("# Processing outputs")
        process_outputs(preproc_file, idata, idx, runname, fit_type, outdir, save_chains)

        return 'OK'
    
    except Exception:

        misc.printFAILED()
        traceback.print_exc()            
          
        return 'ERROR'

#------------------------------------------------------------------------------
def process_outputs(preproc_file, numpyro_data, idx, runname, fit_type, outdir, save_chains):

    # Defining output names and directories
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    rootname = runname+'_'+fit_type
    rundir   = outdir+rootname
    if not os.path.exists(rundir):
        os.mkdir(rundir)

    summary_filename = rundir+"/"+rootname+"_summary_bin"+str(idx)+".txt"
    hdf5_filename    = rundir+"/"+rootname+"_results_bin"+str(idx)+".hdf5"
    netcdf_filename  = rundir+"/"+rootname+"_chains_bin"+str(idx)+".netcdf"

    # Extracting the posteriors
    print("")
    print(" - Checking for divergencies")
    divergences      = numpyro_data.sample_stats.diverging.values.sum()
    total_samples    = numpyro_data.sample_stats.diverging.values.size
    frac_divergences = divergences / total_samples
    print(f"   Divergences: {divergences}/{total_samples} ({100*frac_divergences:.2f}%)\n")

    if frac_divergences > 0.01:
       print("WARNING: Too many divergences! Consider reparametrizing or increasing target_accept_prob.")
    else:
       print("   Divergences are under control.")

    # Saving summary of main parameters on disk
    print("")
    print(" - Saving summary: "+summary_filename)
    summary = az.summary(numpyro_data, var_names=["^(?!.*(offset|continuum|model_spec)).*"], filter_vars="regex", skipna=True)     
    summary.to_string(buf=open(summary_filename, "w"))

    # Processing output and saving results
    print("")
    print(" - Processing and saving results: "+hdf5_filename)
    misc.save_percentiles_to_hdf5(preproc_file, numpyro_data, hdf5_filename)

    # Saving sample chains
    if save_chains == 1:
        print("")
        print(" - Saving chains in Arviz (NETCDF) format: "+netcdf_filename) 
        az.to_netcdf(numpyro_data,netcdf_filename)

    # If we are here, we are DONE!
    misc.printDONE(runname+" - Fit type: "+fit_type+" - Bin: "+str(idx))

    return 'OK'

#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("               BAYES-LOSVD                 ")
    print("                  (run)                    ")
    print("===========================================")
    print("")

    parser = argparse.ArgumentParser(
        prog="myscript.py",
        usage="%(prog)s -f file [options]",
        description="Process spectra with configurable parameters."
    )

    parser.add_argument("-f", "--preproc_file", type=str, default=None,          help="Filename of the preprocessed file")
    parser.add_argument("-l", "--bin_option",   type=str, default="all",         help="BinID for spectrum run [all,odd,even,bin_list]")
    parser.add_argument("-m", "--maskfile",     type=str, default=None,          help="Filename with lines to mask [<filename> or None]")
    parser.add_argument("-p", "--porder",       type=int, default=5,             help="Legendre polynomial order")
    parser.add_argument("-n", "--nsamples",     type=int, default=500,           help="Number of samples")
    parser.add_argument("-c", "--nchain",       type=int, default=2,             help="Number of chains")
    parser.add_argument("-j", "--njobs",        type=int, default=1,             help="Number of jobs to run in parallel")
    parser.add_argument("-o", "--outdir",       type=str, default="../results/", help="Output directory for results")
    parser.add_argument("-t", "--fit_type",     type=str, default="GP",          help="type of fit to be performed")
    parser.add_argument("-s", "--save_chains",  action="store_true",             help="If set, save MCMC chains")
    parser.add_argument("--extra_params", type=misc.parse_kv_pairs,              help="Extra parameters as key=value pairs, e.g. alpha=0.1,beta=0.9,flag=True")

    # Parse arguments
    args = parser.parse_args()
    
    # --- Preparing to run the code  --------------------------------------

    # Checking the file exists
    if not os.path.exists(args.preproc_file):
       misc.printFAILED(args.preproc_file+" does not exist.")
       sys.exit()

    # Defining the runname for output files
    tmpname = os.path.basename(args.preproc_file)
    runname = os.path.splitext(tmpname)[0]
               
    # Defining the list of bins to be analysed
    bin_list, nbins = misc.create_bins_list(args.preproc_file, args.bin_option)

    # Defining the mask from mask_file
    print("")
    print("# Defining the data mask")
    mask = misc.spectral_masking(args.preproc_file, args.maskfile)

    # --- Running PARAMETRIC FIT ------------------------------------------
    print("")
    print("# Getting ready ...")
    run_tmp = run_fit(runname, args.preproc_file, bin_list, mask, args.porder, \
                      args.outdir, args.nsamples, args.nchain, args.njobs, \
                      args.save_chains, args.fit_type, args.extra_params)
    
    if 'ERROR' in run_tmp:
       misc.printFAILED("ERROR: Something went wrong during fit")
    
    # --- Collecting results  ---------------------------------------------
    print("# Packing all available results")
    print("")
    misc.pack_results(runname, suffix=args.fit_type, mask=mask, porder=args.porder)
    
    # --- END -------------------------------------------------------------
    misc.printDONE("FINISHED!")
    sys.exit()



