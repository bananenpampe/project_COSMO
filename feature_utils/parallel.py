from joblib import Parallel, delayed, parallel_backend
from helpers.helpers import grouper
from scipy.special import legendre, gamma
from copy import deepcopy
import numpy as np
import time
from rascal.representations.spherical_expansion import SphericalExpansion
from rascal.representations import SphericalInvariants as SOAP
from rascal.utils import get_radial_basis_covariance, get_radial_basis_pca, get_radial_basis_projections

def get_features(frames,calculator,hypers):
    calculatorinstance = calculator(**hypers)
    #print("worker spawned")
    return calculatorinstance.transform(frames).get_features(calculatorinstance)

def get_features_by_species(frames,calculator,hypers):
    calculatorinstance = calculator(**hypers)
    #print("worker spawned")
    return calculatorinstance.transform(frames).get_features_by_species(calculatorinstance)

def get_features_in_parallel(frames,calculator,hypers,blocksize=25,n_cores=-1):
    """helper function that returns the features of a calculator (from calculator.transform())
       in parallel
    """
    
    #block is necessary to ensure that shape of the chunks is equal
    #replace by get_atomic_species functions
    

    with parallel_backend(backend="threading"):
        results = Parallel(n_jobs=n_cores)(delayed(get_features)(frame, calculator, hypers) for frame in grouper(blocksize,frames))
    
    return np.concatenate(results)

def get_features_in_parallel_by_species(frames,calculator,hypers,blocksize=25,n_cores=-1):
    """helper function that returns the features of a calculator (from calculator.transform())
       in parallel
    """
    
    #block is necessary to ensure that shape of the chunks is equal
    #replace by get_atomic_species functions
    

    with parallel_backend(backend="threading"):
        intermediate_results = Parallel(n_jobs=n_cores)(delayed(get_features_by_species)(frame, calculator, hypers) for frame in grouper(blocksize,frames))
    
    results = {}
    
    #print(intermediate_results)
    for key in intermediate_results[0].keys():
        results[key] = np.concatenate([chunk[key] for chunk in intermediate_results])
    
    return results


class BufferedSOAPFeatures:
    def __init__(self, structures, calculator_params, calculator=SOAP, optimize=True, optimize_radial=20, n_cores=4):
        self.X = None
        self.structures = structures
        self.calculator = calculator
        self.calculator_params = calculator_params
        self.optimize = optimize
        self.optimize_radial = optimize_radial
        self.n_cores = n_cores

    def get_features(self, update_params):
        
        updated_params = self.calculator_params.copy()
        
        for key, value in update_params.items():
            
            if isinstance(value, np.integer):
                value = int(value)
            if isinstance(value, np.floating):
                value = float(value)
            if isinstance(value, np.ndarray):
                value = value.tolist()
                
            updated_params[key] = value

        
        if self.X is None:
            
            #print("Initial calculation")
            #find optimal params here
            #start_time = time.time()
            if self.optimize is True:
                updated_params = get_optimal_radial_basis_hypers_parallel(updated_params,self.structures,expanded_max_radial=self.optimize_radial,num_cores=self.n_cores)
            #print("--- opt time %s seconds ---" % (time.time() - start_time))
            #start_time = time.time()
            self.X = get_features_in_parallel(self.structures,self.calculator,updated_params,n_cores=self.n_cores)
            #print("--- feature time %s seconds ---" % (time.time() - start_time))
        else:
            
            if updated_params == self.calculator_params:
                #print("Stored")
                pass
            else:
                #print("Recalculate")
                #find optimal params here
                if self.optimize is True:
                    updated_params = get_optimal_radial_basis_hypers_parallel(updated_params,self.structures,expanded_max_radial=self.optimize_radial,num_cores=self.n_cores)
                self.X = get_features_in_parallel(self.structures,self.calculator,updated_params, n_cores=self.n_cores)
        
        self.calculator_params = updated_params
        
        return self.X
    
def get_optimal_radial_basis_hypers_parallel(hypers, frames, blocksize=25,expanded_max_radial=-1,num_cores=4):
    """
    Helper function to compute an optimal radial basis following
    Goscinski et al, arxiv:2105.08717.
    hypers: dictionary
        hyperparameters for the desired representation. "max_radial" indicates
        the desired size of the optimal basis
    frames: ase.Atoms
        a list of structures used to estimate the optimal radial basis. can also
        be given as a list of frames blocks, in which case it computes the covariance
        incrementally (useful for large expanded_max_radial and/or large framesets)
    expanded_max_radial: int
        number of intermediate basis to be used to estimate the optimal basis.
        defaults to -1, in which case it is taken to be 2*max_radial
    Returns:
    -------
    optimal_hypers: dictionary
        hyperparameters including the optimal basis projectors
    """

    spherical_expansion_hypers = deepcopy(hypers)

    # removes parameters that don't make sense for a spherical expansion
    spherical_expansion_hypers.pop("normalize", None)
    spherical_expansion_hypers.pop("soap_type", None)
    spherical_expansion_hypers.pop("compute_gradients", None)
    spherical_expansion_hypers.pop("inversion_symmetry", None)

    if "optimization" in spherical_expansion_hypers:
        spherical_expansion_hypers["optimization"].pop("RadialDimReduction", None)

    if expanded_max_radial == -1:
        expanded_max_radial = 2 * hypers["max_radial"]
    spherical_expansion_hypers["max_radial"] = expanded_max_radial

    spex = SphericalExpansion(**spherical_expansion_hypers)

    # computes density expansion coefficients and covariance (incrementally if needed)
    if not type(frames[0]) is list:
        frames = [frames]
    
    feats = get_features_in_parallel_by_species(frames[0],SphericalExpansion,spherical_expansion_hypers,blocksize=blocksize,n_cores=num_cores)
    
    #get_features_in_parallel_by_species(frames[0], calculator=SphericalExpansion, \hypers=spherical_expansion_hypers)
    #compute_spex(frames[0],spherical_expansion_hypers)
    """get_features_in_parallel(frames[0], calculator=SphericalExpansion, \
                         hypers=spherical_expansion_hypers)"""
    
    
    
    cov = get_radial_basis_covariance(spex, feats)
    nframes = len(frames[0])
    
    for fr in frames[1:]:
        feats = spex.transform(fr).get_features_by_species(spex)
        icov = get_radial_basis_covariance(spex, feats)
        # bit perverse: both cov and icov are normalized, so we need to
        # un-normalize before accumulating
        for s in cov.keys():
            cov[s] = (cov[s] * nframes + icov[s] * len(fr)) / (nframes + len(fr))
        nframes += len(fr)

    # principal components from the covariance
    p_val, p_vec = get_radial_basis_pca(cov)

    # converts to the format suitable for hypers
    p_mat = get_radial_basis_projections(p_vec, hypers["max_radial"])

    # assemble the updated hypers
    optimal_hypers = deepcopy(hypers)
    if not "optimization" in optimal_hypers:
        optimal_hypers["optimization"] = {}
    optimal_hypers["optimization"] = {
        "RadialDimReduction": {"projection_matrices": p_mat},
    }

    if not "Spline" in optimal_hypers["optimization"]:
        optimal_hypers["optimization"]["Spline"] = {"accuracy": 1e-8}

    return optimal_hypers    
