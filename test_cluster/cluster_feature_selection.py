#load dependencies

from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
from ase.io import read
from rascal.representations import SphericalInvariants
from rascal.utils import get_optimal_radial_basis_hypers
import numpy as np
from helpers import *
from skcosmo.sample_selection import PCovFPS

#define paths
#TODO:make relative
PATH_TRAIN = "CSD-3k+S546_shift_tensors.xyz"

#define hypers
hypers = dict(soap_type="PowerSpectrum",
              interaction_cutoff=4.5,
              max_radial=12,
              max_angular=9,
              gaussian_sigma_constant=0.3,
              gaussian_sigma_type="Constant",
              cutoff_function_type="RadialScaling",
              cutoff_smooth_width=0.5,
              cutoff_function_parameters=
                    dict(
                            rate=1,
                            scale=3.0,
                            exponent=6
                        ),
              radial_basis="GTO",
              normalize=True,
              optimization=
                    dict(
                            Spline=dict(
                               accuracy=1.0e-05
                            )
                        ),
              compute_gradients=False
              )



for atomic_species in [ 1,  6,  7,  8, 16]:
    
    #load structs
    structures_train = read(PATH_TRAIN,format="extxyz",index=":")
    
    #mask and wrap
    for structure in structures_train: mask_center_atoms_by_species(structure,species_select=[atomic_species])
    for structure in structures_train: structure.wrap(eps=1e-12)
    
    #filter by status
    structures_train = filter_by_status(structures_train,status="PASSING")
    structures_train = structures_train[-100:]
    
    #determine optimal hypers after masking?
    hypers_opt = get_optimal_radial_basis_hypers(hypers, structures_train, expanded_max_radial=20)
    
    #get corresponding shifts, maybe do the same on tensors?
    shifts = np.array([tensor for structure in structures_train for tensor in structure.arrays["cs_tensor"][structure.arrays["center_atoms_mask"]]])
    
    #building train features
    calculator = SphericalInvariants(**hypers)
    X_train = calculator.transform(structures_train).get_features(calculator)
    
    #selecting samples
    selector = PCovFPS(
                    n_to_select=50,
                    progress_bar=False,
                    score_threshold=1e-12,
                    full=False,

                    # float, default=0.5
                    # The PCovR mixing parameter, as described in PCovR as alpha
                    mixing = 0.5,

                    # int or 'random', default=0
                    # Index of the first selection.
                    # If ‘random’, picks a random value when fit starts.
                    initialize = 0,
                    )
    
    selector.fit(X_train, shifts)
    selected_ids = selector.selected_idx_
    relative_ids = return_relative_inds(structures_train, selected_ids, atomic_species)
    
    #saving ids
    np.save("selected_sample_ids_{}_n12_l9_PASSING".format(atomic_species),selected_ids)
    np.save("selected_sample_ids_relative_{}_n12_l9_PASSING".format(atomic_species),relative_ids)
    
