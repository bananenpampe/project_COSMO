import numpy as np
from ase.io import read
from rascal.representations import SphericalInvariants
from rascal.models.kernels import Kernel
from sklearn.kernel_ridge import KernelRidge
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
import time
import sklearn

def load_CSD_data(PATH, random_subsample=None):
    """Helper function that loads the CSD-2K and CSD-500 dataset
       The CSD-X dataset are .txt files of joined extended-xyz files. Where unit cell parameters are given in the comment line
       And atom wise calculated GIPAW shifts, are stored in additional atom-wise collums. first additional colum is GIPAW.
       In CSD-500 another column is given with the 
    
    Parameters
    ----------
    PATH             : string
                       absolut path of the dataset .txt file
    
    random_subsample : int < dataset_size
                       returns a random subsample of the dataset with N(random_subsample) entries
                       
    
    Returns
    -------
    structures : list of ase.atoms ojects
                 wrapped structures of the dataset
    
    shifts     : numpy array of size (N_environments,) or (N_environments,2)
                 shifts of the individual nuclei
    """
    
    
    
    structures = read(PATH,format="extxyz",index=":")
    

        
    for atom in structures:
        atom.wrap()
        
    if random_subsample is not None:
        ids = list(range(len(structures)))
        np.random.shuffle(ids)
        train_ids = ids[:random_subsample]
        structures_subsample = [structures[ii] for ii in ids[:random_subsample]]
        shifts_subsample = np.concatenate([atoms.arrays["CS"] for atoms in structures_subsample])
        return structures_subsample, shifts_subsample
        
        
    else:
        shifts = np.concatenate([atoms.arrays["CS"] for atoms in structures])
        return structures, shifts
    
hypers = {"soap_type": "PowerSpectrum",
          "interaction_cutoff": 3,
          "radial_basis": "GTO",
          "max_radial": 9,
          "max_angular": 9,
          "gaussian_sigma_constant": 0.3,
          "gaussian_sigma_type":"Constant",
          "cutoff_function_type":"ShiftedCosine",
          "cutoff_smooth_width": 0.5,
          "normalize": True,
          "compute_gradients":False,
          "cutoff_function_parameters":dict(rate=1,scale=3.5,exponent=4)
          #"optimization": dict(Spline=dict(accuracy=1.0e-05))
          #"expansion_by_species_method":'structure wise'
         }
    
PATH_TRAIN = "CSD-2k_relaxed_shifts.txt"

structures_train, shifts_train = load_CSD_data(PATH_TRAIN,random_subsample=100)

for structure in structures_train: 
    mask_center_atoms_by_species(structure,species_select=[1])

    

calculator = SphericalInvariants(**hypers)
atoms_list_train = calculator.transform(structures_train)

kernel = Kernel(calculator,target_type="Atom",zeta=2)

"""
t_0 = time.time()
kernel_computed = kernel(atoms_list_train)
t_1 = time.time()
print("t librascal: {:.2f}".format(t_1-t_0))
"""

t_0 = time.time()
X_train = calculator.transform(structures_train).get_features(calculator)

Kernel_sklearn = sklearn.metrics.pairwise.polynomial_kernel(X_train, degree=2, gamma=1., coef0=0)
t_1 = time.time()
print("t sklearn: {:.2f}".format(t_1-t_0))

#print(np.allclose(kernel_computed,Kernel_sklearn))
