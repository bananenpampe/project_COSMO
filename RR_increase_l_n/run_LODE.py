from projection_coeffs import Density_Projection_Calculator


import numpy as np
import time
from ase.io import read
from helpers.helpers import filter_by_status
from rascal.representations import SphericalExpansion, SphericalInvariants
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species,mask_center_atoms_by_id
from rascal.utils import (spherical_expansion_reshape, lm_slice)


hypers_lode = {'smearing':1.0, # WARNING: comp. cost scales cubically with 1/smearing
        'max_angular':6,
        'max_radial': 1,       
        'cutoff_radius':4.5,
        'potential_exponent':1,
        'compute_gradients': False
              }

frames = read('../make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz',':')
frames = filter_by_status(frames, status="PASSING")
for frame in frames: frame.wrap(eps=1e-12)

lode = Density_Projection_Calculator(**hypers_lode)
lode.transform(frames[:20], species_dict)
feat3 = lode.get_features()

np.save("lode_all_PASSING_smearing1_max_ang6.npy", feat3)