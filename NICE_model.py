import sys

sys.path.append("/home/kellner/packages/project_COSMO/")
sys.path.append("/home/kellner/packages/project_COSMO/loader")

from loader import load_data
from sklearn.linear_model import Ridge, RidgeCV
from feature_utils.parallel import get_features_in_parallel, get_optimal_radial_basis_hypers_parallel
from rascal.representations import SphericalInvariants as SOAP
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skcosmo.preprocessing import StandardFlexibleScaler
from sklearn.compose import TransformedTargetRegressor
from skcosmo.model_selection import atom_groups_by_frame
from sklearn.model_selection import GroupKFold
import numpy as np
import ase.io
from nice.blocks import *
from nice.utilities import *
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from joblib import dump
from rascal.utils import get_optimal_radial_basis_hypers

train_structures, test_structures, train_properties, test_properties = load_data("../make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                    "../make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz")


HYPERS = {
'interaction_cutoff': 6.3,
'max_radial': 5,
'max_angular': 5,
'gaussian_sigma_type': 'Constant',
'gaussian_sigma_constant': 0.05,
'cutoff_smooth_width': 0.3,
"cutoff_function_type":"RadialScaling",
 'cutoff_function_parameters': dict(rate=1.,
                    scale= 2.0,
                    exponent=3.
                        ),
'radial_basis': 'GTO'
}


train_structures, test_structures, train_properties, test_properties = load_data("/home/kellner/packages/project_COSMO/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                    "/home/kellner/packages/project_COSMO/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz")


HYPERS = {
'interaction_cutoff': 6.3,
'max_radial': 5,
'max_angular': 5,
'gaussian_sigma_type': 'Constant',
'gaussian_sigma_constant': 0.05,
'cutoff_smooth_width': 0.3,
"cutoff_function_type":"RadialScaling",
 'cutoff_function_parameters': dict(rate=1.,
                    scale= 2.0,
                    exponent=3.
                        ),
'radial_basis': 'GTO'
}

def get_transformer():
    return StandardSequence([
        StandardBlock(ThresholdExpansioner(num_expand=150),
                      CovariantsPurifierBoth(max_take=10),
                      IndividualLambdaPCAsBoth(n_components=50),
                      ThresholdExpansioner(num_expand=300, mode='invariants'),
                      InvariantsPurifier(max_take=50),
                      InvariantsPCA(n_components=200)),
        StandardBlock(ThresholdExpansioner(num_expand=150),
                      CovariantsPurifierBoth(max_take=10),
                      IndividualLambdaPCAsBoth(n_components=50),
                      ThresholdExpansioner(num_expand=300, mode='invariants'),
                      InvariantsPurifier(max_take=50),
                      InvariantsPCA(n_components=200)),
        StandardBlock(None, None, None,
                      ThresholdExpansioner(num_expand=300, mode='invariants'),
                      InvariantsPurifier(max_take=50),
                      InvariantsPCA(n_components=200))
    ],
                            initial_scaler=InitialScaler(
                                mode='signal integral', individually=True))



all_species = get_all_species(train_structures + test_structures)

train_coefficients = get_spherical_expansion(train_structures, HYPERS, all_species)
test_coefficients = get_spherical_expansion(test_structures, HYPERS, all_species)
print(test_coefficients.keys())
transformers = {}
for key in train_coefficients.keys():
    transformers[key] = get_transformer()

#TODO: Pass here train_coefficients:

fit_species = [1,6,7,8]

for key in fit_species:
        fit_species = np.load("PCOV_FPS_selected_sample_ids_{}_selected_20000_n8_l8_PASSING.npy".format(key))
        print(train_coefficients[key][fit_species].shape)
        transformers[key].fit(train_coefficients[key][fit_species])

train_features = {}
for specie in [1,6,7,8]:
    train_features[specie] = transformers[specie].transform(train_coefficients[specie], return_only_invariants=True)

test_features = {}
for specie in [1,6,7,8]:
    test_features[specie] = transformers[specie].transform(test_coefficients[specie],
                                                   return_only_invariants=True)

for specie in [1,6,7,8]:
    Strain, Stest, Ytrain, Ytest = load_data("/home/kellner/packages/project_COSMO/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                    "/home/kellner/packages/project_COSMO/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz", selected_species=int(specie))
    
    
    Xtrain = np.concatenate([value for key, value in train_features[specie].items()],axis=1)
    Xtest = np.concatenate([value for key, value in test_features[specie].items()],axis=1)
    
    
    
    groups = atom_groups_by_frame(Strain)
    splits = list(GroupKFold(n_splits=5).split(Xtrain,Ytrain,groups=groups))
    model = RidgeCV(alphas=np.logspace(-6,3,10),cv=splits,scoring="neg_mean_squared_error")
    model.fit(Xtrain,Ytrain)
    print("{} species ridges' alpha is: {}".format(specie,model.alpha_))
    
    Ypred_test = model.predict(Xtest)
    Ypred_train = model.predict(Xtrain)
    
    rmse_test = mean_squared_error(Ytest,Ypred_test,squared=False)
    
    rmse_train = mean_squared_error(Ytrain,Ypred_train,squared=False)
    
    mae_test = mean_absolute_error(Ypred_test,Ytest)
    mae_train = mean_absolute_error(Ypred_train,Ytrain)
    
    print("test-RMSE: {} \n   train-RMSE: {}:\n  test-MAE:{}\n  train-MAE:{}".format(rmse_test,rmse_train,mae_test,mae_train))
    
    
    dump(transformers[specie],str(specie) + "_NICE_transformer.pkl")