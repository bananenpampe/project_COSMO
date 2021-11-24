from ase.io import read, write
import numpy as np
from rascal.representations import SphericalCovariants

train_structures = read("./train_tensor/CSD-3k+S546_shift_tensors.xyz",format="extxyz",index=":")

train_structures = train_structures[0:3]
for atoms in train_structures:
    atoms.wrap() 

    
hypers = {"soap_type": "LambdaSpectrum",
          "interaction_cutoff": 3,
          "radial_basis": "GTO",
          "max_radial": 9,
          "max_angular": 9,
          "gaussian_sigma_constant": 0.3,
          "gaussian_sigma_type":"Constant",
          "cutoff_function_type":"ShiftedCosine",
          "cutoff_smooth_width": 0.5,
          "normalize": True,
          "cutoff_function_parameters":dict(rate=1,scale=3.5,exponent=4),
          #"optimization": dict(Spline=dict(accuracy=1.0e-05))
          #"global_species":[1,6,7,8]
          "covariant_lambda":0
         }

calculator = SphericalCovariants(**hypers)
atoms_list_train = calculator.transform(train_structures)

X_train = calculator.transform(train_structures).get_features(calculator)

print("yo")