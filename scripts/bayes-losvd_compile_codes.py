import sys
import glob
import pickle
import pystan
import warnings
import lib.misc_functions as     misc
from   hashlib            import md5
#==============================================================================
def stan_cache(model_code, model_name=None, **kwargs):

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
        print("Using cached StanModel")

    return sm
#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("               BAYES-LOSVD                 ")
    print("             (compile codes)               ")
    print("===========================================")
    print("")


    list = glob.glob("stan_model/*.stan")

    for codefile in list:
       misc.printRUNNING(codefile) 
       with open(codefile, 'r') as myfile:
            code = myfile.read()
       model = stan_cache(model_code=code) 
       misc.printDONE(codefile+" compiled.")

    exit()