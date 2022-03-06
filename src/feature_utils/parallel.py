from joblib import Parallel, delayed, parallel_backend
from helpers.helpers import grouper
from copy import deepcopy
import numpy as np
import time
from rascal.representations import SphericalInvariants as SOAP

def get_features(frames,calculator,hypers):
    calculatorinstance = calculator(**hypers)
    #print("worker spawned")
    return calculatorinstance.transform(frames).get_features(calculatorinstance)

def get_features_in_parallel(frames,calculator,hypers,blocksize=25,n_cores=-1):
    """helper function that returns the features of a calculator (from calculator.transform())
       in parallel
    """
    
    #block is necessary to ensure that shape of the chunks is equal
    #replace by get_atomic_species functions
    

    with parallel_backend(backend="threading"):
        results = Parallel(n_jobs=n_cores)(delayed(get_features)(frame, calculator, hypers) for frame in grouper(blocksize,frames))
    
    return np.concatenate(results)
