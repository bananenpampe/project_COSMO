from feature_utils.parallel import get_optimal_radial_basis_hypers_parallel, get_features_in_parallel
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
    
    
    
    species_alphas = []
    specie_train_rmse = []
    specie_test_rmse = []
    specie_train_mae = []
    specie_test_mae = []
    
    numbers_of_feats = []
    
    for model_no in range(16):
        train_structures, test_structures, train_properties, test_properties = load_data("/home/kellner/packages/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                        "/home/kellner/packages/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",selected_species=specie,random_subsample_train=N_subsample)
        
        atom_groups = atom_groups_by_frame(train_structures)
        structure_list = [structure.info["NAME"] for structure in train_structures]
        
        with open("/home/kellner/git_pack/ShiftML-Light/data/RR_this_work_models/hypers/{}_hypers.json".format(specie),"r") as fg:
            int_hyp = json.load(fg)
        #int_hyp = get_optimal_radial_basis_hypers_parallel(int_hyp,train_structures,expanded_max_radial=20)
        Xtrain = get_features_in_parallel(train_structures,calculator=SOAP,hypers=int_hyp)
        Xtest = get_features_in_parallel(test_structures,calculator=SOAP,hypers=int_hyp)
        
        clf = GridSearchCV(Ridge(),param_grid={"alpha":np.logspace(-6,3,10)},cv=GroupKFold(n_splits=5),scoring="neg_mean_squared_error")
        #.split(Xtrain,train_properties,groups=atom_groups)
        
        clf.fit(Xtrain,train_properties,groups=atom_groups) 
        species_alphas.append(clf.cv_results_['params'][clf.best_index_]["alpha"])
        
        Y_test_pred = clf.predict(Xtest)
        Y_train_pred = clf.predict(Xtrain)
        
        num_feat = Xtrain.shape[1]
        
        #save errors
        rmse_test = mean_squared_error(test_properties,Y_test_pred,squared=False)
        rmse_train = mean_squared_error(train_properties,Y_train_pred,squared=False)
        mae_test = mean_absolute_error(test_properties,Y_test_pred)
        mae_train = mean_absolute_error(train_properties,Y_train_pred)
        specie_train_rmse.append(rmse_train)
        specie_test_rmse.append(rmse_test)
        specie_train_mae.append(mae_train)
        specie_test_mae.append(mae_test)
        numbers_of_feats.append(num_feat)
        
        #model name
        model_name = str(specie) + "_" + str(model_no) 
        
        #save structure_list
        np.save(model_name + ".npy", np.array(structure_list))
        
        #save hypers
        with open(model_name + "_hypers.json", 'w') as fileob:
            json.dump(int_hyp, fileob) 
        
        #save_model
        joblib.dump(clf.best_estimator_, model_name + "_name")
        joblib.dump(clf, model_name + "_name_clf") 
        
    error_dict = {"rmse_test":specie_test_rmse,"rmse_train":specie_train_rmse,"mae_test":specie_test_mae,"mae_train":specie_train_mae,"alphas":species_alphas,"num_feats":numbers_of_feats}
    errors[specie] = error_dict

with open("errors_RR_mine.json", 'w') as fileob:
    json.dump(errors, fileob) 
