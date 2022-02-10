from feature_utils.parallel import BufferedSOAPFeatures, get_features_in_parallel, get_optimal_radial_basis_hypers_parallel
from loader import load_data
from copy import deepcopy
import numpy as np
from sklearn.linear_model import RidgeCV
from rascal.representations import SphericalInvariants as SOAP
from rascal.utils import get_optimal_radial_basis_hypers
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_id
from skcosmo.model_selection import atom_groups_by_frame
from sklearn.linear_model import LinearRegression, Ridge
from copy import deepcopy
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from skopt import dump
import time



train_structures, test_structures, train_properties, test_properties = load_data("./make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                    "./make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",selected_species=1,random_subsample_test=200)

hypers = dict(soap_type="PowerSpectrum",
              interaction_cutoff=4.5,
              max_radial=8,
              max_angular=8,
              gaussian_sigma_constant=0.3,
              gaussian_sigma_type="Constant",
              radial_basis="GTO",
              normalize=True,
              cutoff_smooth_width=0.3,
              cutoff_function_type="RadialScaling",
              optimization=
                    dict(
                            Spline=dict(
                               accuracy=1.0e-05
                            )
                        ),
              cutoff_function_parameters= dict(rate=1,
                    scale=2.0,
                    exponent=3
                        ),
              compute_gradients=False,
              expansion_by_species_method="user defined",
              global_species=[1, 6, 7, 8, 16]
              )

nested_param = dict(rate=1,
                    scale=3.0,
                    exponent=6
                        )

"""space = [Real(10**-5, 10**2, "log-uniform", name='alpha'),
        Real(0.05,1.5, "uniform", name="gaussian_sigma_constant"),
        Real(2.,4.5, "uniform", name="interaction_cutoff")]"""

space = [Real(10**-5, 10**3, "log-uniform", name='alpha'),
        Real(1.,3.5, "uniform", name="scale"),
        Real(3.5,8.0, "uniform", name="interaction_cutoff"),
        Real(0.01,0.5, "uniform", name="gaussian_sigma_constant"),
        Real(1.,5., "uniform", name="exponent"),
        Real(0.5,3., "uniform", name="rate")
        ]

reg = Ridge(solver="lsqr")
y = train_properties
atom_groups = atom_groups_by_frame(train_structures)
Feature_gen = BufferedSOAPFeatures(train_structures, hypers,n_cores=-1)


@use_named_args(space)
def soap_objective(**params):
    update_dict = {}
    
    new_params = params.copy()
    nested_dict_param = deepcopy(Feature_gen.calculator_params)
    nested_dict_param = nested_dict_param["cutoff_function_parameters"]
    
    for key, value in new_params.items():
        if key in Feature_gen.calculator_params:
            #hypers[key] = value
            update_dict[key] = params.pop(key, None)
        if key in nested_param:
            nested_dict_param[key] = params.pop(key, None)
            #print("I update my {} to {}".format(key,value))
        
        update_dict["cutoff_function_parameters"] = nested_dict_param
            
            
    #print(update_dict)
    start_time = time.time()
    reg.set_params(**params)
    
    
    X = Feature_gen.get_features(update_dict)
    print("--- feature_gen time %s seconds ---" % (time.time() - start_time))
    #print(X.shape)
    
    #print(Feature_gen.hypers["max_angular"])
    start_time = time.time()
    splits = list(GroupKFold(n_splits=5).split(X,y,groups=atom_groups))
    
    start_time = time.time()
    score = -np.mean(cross_val_score(reg, X, y, cv=splits, n_jobs=1,
                                    scoring="neg_mean_squared_error"))
    
    print("--- cross_val time %s seconds ---" % (time.time() - start_time))
        
    return score


start_time = time.time()
res_gp = gp_minimize(soap_objective, space, n_calls=10, random_state=0, n_jobs=1)
print("--- 10 steps took %s seconds ---" % (time.time() - start_time))

dump(res_gp, "1H_opt_RR")