from feature_utils.parallel import get_optimal_radial_basis_hypers_parallel, get_features_in_parallel
from sklearn.metrics.pairwise import pairwise_kernels
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
from rascal.representations import SphericalInvariants as SOAP
from skcosmo.model_selection import atom_groups_by_frame
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from loader.loader import load_data
import joblib
from copy import deepcopy
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import os

num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))

feature_dict = {1:[
 2.513807665262468,
 6.0162081478510965,
 0.2687521416964727,
 4.563666005760138,
 1.587405944267303]
,6:[
 2.6009421646538335,
 4.643475126818923,
 0.1793445885211041,
 4.6984309434873435,
 1.967643707392294]
                ,7:[
 1.7676477782307636,
 6.914698486615254,
 0.12265607366287667,
 5.996856926677453,
 4.8639728841745455],8:[
 2.415028605215485,
 5.631249765195175,
 0.1759552323571449,
 6.829001301667477,
 3.33033564862975]}

param_dict = {1:[0.020386178205326178,0.00011377310376903718],6:[1.3520804386962695,1.071695207955416e-06],7:[5.8642046072305245,0.0055854091930014746],8:[0.024484026246436308,4.539770180278873e-05]}

def update_hypers(hypers,param_list,params):
        cutoff_dict = hypers["cutoff_function_parameters"]
        for param,value in zip(param_list, params):
                if param in cutoff_dict:
                        hypers["cutoff_function_parameters"][param] = value
                else:
                        hypers[param] = value
        return hypers


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
              cutoff_function_parameters= dict(rate=1.,
                    scale= 2.0,
                    exponent=3.
                        ),
              compute_gradients=False,
              expansion_by_species_method="user defined",
              global_species=[1, 6, 7, 8, 16]
              )

errors = {1:None,6:None,7:None,8:None}

train_structures, test_structures, train_properties, test_properties = load_data("/home/kellner/packages/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                    "/home/kellner/packages/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",selected_species=1)

N_train = len(train_structures)
N_subsample = int(3 * N_train//4)


for specie in [1,6,7,8]:
    #make full hypers
    train_structures_full, test_structures, train_properties_full, test_properties = load_data("/home/kellner/packages/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                        "/home/kellner/packages/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",selected_species=specie)
    
    int_hyp = hypers.copy()
    int_hyp = update_hypers(int_hyp, [
                    'scale',
                    'interaction_cutoff',
                    'gaussian_sigma_constant',
                    'exponent',
                    'rate'],feature_dict[specie])

    int_hyp = get_optimal_radial_basis_hypers_parallel(int_hyp,train_structures_full,expanded_max_radial=20)
    Xtrain_full = get_features_in_parallel(train_structures_full,calculator=SOAP,hypers=int_hyp)
    Xtest = get_features_in_parallel(test_structures,calculator=SOAP,hypers=int_hyp)
    # get element specific hypers from dict
    
    K_precompute = pairwise_kernels(Xtrain_full,Xtrain_full, metric='rbf', gamma=param_dict[specie][0], n_jobs=num_cores)
    #K = self._get_kernel(X, self.X_fit_)
    K_test_precompute = pairwise_kernels(Xtest, Xtrain_full, metric='rbf', gamma=param_dict[specie][0], n_jobs=num_cores)
    

    species_alphas = []
    specie_train_rmse = []
    specie_test_rmse = []
    specie_train_mae = []
    specie_test_mae = []
    
    numbers_of_feats = []
    for model_no in range(16):
        train_structures, test_structures, train_properties, test_properties = load_data("/home/kellner/packages/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                        "/home/kellner/packages/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",selected_species=specie,random_subsample_train=N_subsample)
        
        #get the one_hot vec here
        hotencode_array = []
        #stupid O(N^2)
        structure_list = [structure.info["NAME"] for structure in train_structures]
        
        for structure in train_structures_full: 
            mask_array = structure.arrays["center_atoms_mask"]
            
            if structure.info["NAME"] in structure_list:
                hotencode_array.append(np.full(len(structure.arrays["cs_iso"][mask_array]),True))
            else:    
                hotencode_array.append(np.full(len(structure.arrays["cs_iso"][mask_array]),False))
        
        hotencode_array = np.hstack(hotencode_array)
        
        
        clf = Ridge(alpha=param_dict[specie][1])
        #.split(Xtrain,train_properties,groups=atom_groups)
        
        clf.fit(K_precompute[np.ix_(hotencode_array,hotencode_array)],train_properties)
        
        Y_test_pred = clf.predict(K_test_precompute[:,hotencode_array])
        Y_full_pred = clf.predict(K_precompute[:,hotencode_array])
        #saving truth and model prediction



        num_feat = Xtrain_full.shape[1]
        
        #save errors
        rmse_test = mean_squared_error(test_properties,Y_test_pred,squared=False)
        rmse_train = mean_squared_error(train_properties,Y_full_pred[hotencode_array],squared=False)
        mae_test = mean_absolute_error(test_properties,Y_test_pred)
        mae_train = mean_absolute_error(train_properties,Y_full_pred[hotencode_array])
        specie_train_rmse.append(rmse_train)
        specie_test_rmse.append(rmse_test)
        specie_train_mae.append(mae_train)
        specie_test_mae.append(mae_test)
        numbers_of_feats.append(num_feat)
        
        #model name
        model_name = "specie_" + str(specie) + "_model_no_" + str(model_no) 
        
        #save structure_list
        np.save(model_name + ".npy", np.array(structure_list))
        np.save(model_name + "_full_test_pred.npy", Y_full_pred)
        np.save(model_name + "_test_pred.npy", Y_test_pred)
        np.save(model_name + "_one_hot.npy", hotencode_array)
        
        #save hypers
        with open(model_name + "_hypers.json", 'w') as fileob:
            json.dump(int_hyp, fileob) 
        
        #save_model
        joblib.dump(clf, model_name + "_name_clf") 
        
    error_dict = {"rmse_test":specie_test_rmse,"rmse_train":specie_train_rmse,"mae_test":specie_test_mae,"mae_train":specie_train_mae,"alphas":species_alphas,"num_feats":numbers_of_feats}
    errors[specie] = error_dict

with open("errors_RR_mine.json", 'w') as fileob:
    json.dump(errors, fileob) 
