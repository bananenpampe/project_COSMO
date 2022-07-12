import sys

sys.path.append("/home/kellner/packages/project_COSMO/")
sys.path.append("/home/kellner/packages/project_COSMO/loader")

from feature_utils.parallel import get_optimal_radial_basis_hypers_parallel, get_features_in_parallel
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
from rascal.representations import SphericalInvariants as SOAP
from skcosmo.model_selection import atom_groups_by_frame
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
from copy import deepcopy
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import pickle


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from loader import load_data
from feature_utils.parallel import get_features_in_parallel, get_optimal_radial_basis_hypers_parallel     
from rascal.utils import (spherical_expansion_reshape, lm_slice)
from rascal.representations import SphericalExpansion
from rascal.representations import SphericalInvariants as SOAP
from skcosmo.preprocessing import StandardFlexibleScaler
import os 
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_default_tensor_type(torch.DoubleTensor)
num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
device = "cuda" if torch.cuda.is_available() else "cpu"
#define NN class


class NeuralNetwork(nn.Module):
    def __init__(self,l1):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Dropout(p=0.2),
            nn.Linear(num_input, l1),
            nn.LayerNorm(l1),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            #nn.Linear(248),
            #nn.BatchNorm1d(64),
            #nn.Softplus(),
            #nn.Dropout(p=0.5),
            nn.Linear(l1,1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



#define training function

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        def closure():
            #Clear gradients, not be accumulated
            optimizer.zero_grad()

            #Forward pass to get output
            outputs = model(X)

            #Calculate Loss: softmax + cross entropy loss
            loss = loss_fn(outputs, y)

            #Get gradients 
            loss.backward()
            return loss
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()#closure)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


#define testfunction
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = scaler_y.inverse_transform(torch.Tensor.cpu(pred))
            y = scaler_y.inverse_transform(torch.Tensor.cpu(y))
            test_loss += loss_fn(pred, y).item()
            
    test_loss /= num_batches
    test_loss = np.sqrt(test_loss)
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

#define final testfunction
def test_final(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    predictions = []
    truth = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = scaler_y.inverse_transform(torch.Tensor.cpu(pred))
            y = scaler_y.inverse_transform(torch.Tensor.cpu(y))
            test_loss += loss_fn(pred, y).item()
            predictions.append(pred.numpy().flatten())
            truth.append(y.numpy().flatten())
            
    print(type(truth))
    print(type(truth[0]))
    #print(truth)
    
    
    test_loss /= num_batches
    test_loss = np.sqrt(test_loss)
    return np.hstack(truth), np.hstack(predictions), test_loss

#feature dict containing optimal hypers
feature_dict = {1:[
 1.5911125359265827,
 4.224852999573411,
 0.1928389330844191,
 13.039179939787841,
 6.922003432838549]
,6:[
 1.1454045306428244,
 2.9626689747037545,
 0.22491904212207098,
 6.07675166733982,
 7.238722410194945]
                ,7:[1.9992383823420647,
 2.640510291833663,
 0.0883949351512584,
 4.826107517404777,
 5.534167043298729]
,8:[
 1.5550706694708154,
 3.009381799356438,
 0.28317247058592393,
 7.470765944368672,
 5.678773836257633]}

#N_layer, lr, decay
NN_dict={1:[256,0.030678201628307138,1e-06],
         6:[512,0.007018159678494718,0.01],
         7:[1024,0.033119254316911935,0.01053693234188016],
         8:[512,0.03022462385453728,0.01]}

#update function to update hypers
def update_hypers(hypers,param_list,params):
        cutoff_dict = hypers["cutoff_function_parameters"]
        for param,value in zip(param_list, params):
                if param in cutoff_dict:
                        hypers["cutoff_function_parameters"][param] = value
                else:
                        hypers[param] = value
        return hypers

#radial spectrum hypers -> no normalization
hypers = dict(soap_type="RadialSpectrum",
              interaction_cutoff=4.5,
              max_radial=12,
              max_angular=0,
              gaussian_sigma_constant=0.3,
              gaussian_sigma_type="Constant",
              radial_basis="GTO",
              normalize=False,
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
TRAINPATH = "/home/kellner/packages/project_COSMO/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz"
TESTPATH = "/home/kellner/packages/project_COSMO/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz"

train_structures, test_structures, train_properties, test_properties = load_data(TRAINPATH,TESTPATH,selected_species=1)

N_train = len(train_structures)
N_subsample = int(3 * N_train//4)


for specie in [1,6,7,8]:
    #make full hypers
    train_structures_complete, test_structures, train_properties_complete, test_properties = load_data(TRAINPATH,TESTPATH, selected_species=specie)
    
    #copy hypers and update
    int_hyp = hypers.copy()
    int_hyp = update_hypers(int_hyp, [
                    'scale',
                    'interaction_cutoff',
                    'gaussian_sigma_constant',
                    'exponent',
                    'rate'],feature_dict[specie])
    
    #get optimal hypers
    int_hyp = get_optimal_radial_basis_hypers_parallel(int_hyp,train_structures_complete,expanded_max_radial=20)
    Xtrain_complete = get_features_in_parallel(train_structures_complete,SOAP,int_hyp,n_cores=num_cores)
    #get scaler once THIS DOES NOT WORK !!! ONE SCALER FOR EACH MODEL (makes sense for bias etc)
    """Xtrain = get_features_in_parallel(train_structures,SOAP,hypers,n_cores=num_cores)
    scaler_X = StandardFlexibleScaler().fit(Xtrain)
    train_properties = train_properties.reshape(-1, 1)
    scaler_y = StandardFlexibleScaler().fit(train_properties)"""
    
    learning_rate_opt = NN_dict[specie][1]
    decay_opt = NN_dict[specie][2]
    
    final_results = []
    final_predictions = []
    
    for model_no in range(16):
        train_structures, test_structures, train_properties, test_properties = load_data("/home/kellner/packages/project_COSMO/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                            "/home/kellner/packages/project_COSMO/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",selected_species=specie,random_subsample_train=N_subsample)
        
        
        structure_list = [structure.info["NAME"] for structure in train_structures]
        
        hotencode_array = []
        
        #stupid O(N^2)
        for structure in train_structures_complete:
            mask_array = structure.arrays["center_atoms_mask"]
            if structure.info["NAME"] in structure_list:
                hotencode_array.append(np.full(len(structure.arrays["cs_iso"][mask_array]),True))
            else:    
                hotencode_array.append(np.full(len(structure.arrays["cs_iso"][mask_array]),False))
        
        Xtrain = get_features_in_parallel(train_structures,SOAP,int_hyp,n_cores=num_cores)
        Xtest = get_features_in_parallel(test_structures,SOAP,int_hyp,n_cores=num_cores)
        
        #scale features using scalers
        scaler_X = StandardFlexibleScaler().fit(Xtrain)
        Xtrain = scaler_X.transform(Xtrain)
        Xtest = scaler_X.transform(Xtest)
        Xtrain_complete_rescaled = scaler_X.transform(np.copy(Xtrain_complete))
        train_properties = train_properties.reshape(-1, 1)
        test_properties = test_properties.reshape(-1, 1)
        train_properties_complete_rescaled = np.copy(train_properties_complete).reshape(-1, 1)
        
        scaler_y = StandardFlexibleScaler().fit(train_properties)
        train_properties = scaler_y.transform(train_properties)
        train_properties_complete_rescaled = scaler_y.transform(train_properties_complete_rescaled)
        test_properties = scaler_y.transform(test_properties)

        print(Xtrain.shape)
        
        #generate tensors from numpy arrays
        Xtrain = torch.from_numpy(Xtrain).to(dtype=torch.float64)
        Ytrain = torch.from_numpy(train_properties).to(dtype=torch.float64)
        Xtest = torch.from_numpy(Xtest).to(dtype=torch.float64)
        Ytest = torch.from_numpy(test_properties).to(dtype=torch.float64)
        Xeval = torch.from_numpy(Xtrain_complete_rescaled).to(dtype=torch.float64)
        Yeval = torch.from_numpy(train_properties_complete_rescaled).to(dtype=torch.float64)
        
        #generate datasets from tensors
        dataset_train = TensorDataset( Xtrain, Ytrain )
        dataset_test = TensorDataset( Xtest, Ytest )
        dataset_eval = TensorDataset( Xeval, Yeval )
        batch_size = 512

        num_input = len(Xtrain[0])

        # Create data loaders.
        train_dataloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True)
        test_dataloader = DataLoader(dataset_test, batch_size=batch_size,shuffle=False)
        eval_dataloader = DataLoader(dataset_eval, batch_size=batch_size,shuffle=False)
        
        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break



        model = NeuralNetwork(l1=NN_dict[specie][0]).to(device)
        
        #print model archtiecture
        print(model)

        swa_model = AveragedModel(model)
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate_opt, weight_decay=decay_opt,amsgrad=True)
        
        
        #------SWA steps-----
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        swa_start = 50
        swa_scheduler = SWALR(optimizer, swa_lr=2*learning_rate_opt)

        swa_start = 50
        for epoch in range(100):
               print(f"Epoch {epoch+1}\n-------------------------------")
               train(train_dataloader, model, loss_fn, optimizer)
               test(test_dataloader, model, loss_fn)     

               if epoch > swa_start:
                   swa_model.update_parameters(model)
                   swa_scheduler.step()
               else:
                   scheduler.step()

        swa_model.to("cpu")
        
        # Update bn statistics for the swa_model at the end
        torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
        
        # Use swa_model to make predictions on test data
        swa_model.to("cuda")
        
        test(test_dataloader,swa_model,loss_fn)
        truth, predictions, final_result = test_final(test_dataloader,swa_model,loss_fn)
        truth_eval, predictions_eval, final_result_eval = test_final(eval_dataloader,swa_model,loss_fn)
        
        np.save("specie_{}_model_no_{}_predictions_train_complete.npy".format(specie, model_no),  predictions_eval)
        np.save("specie_{}_true_train_complete.npy".format(specie), truth_eval)
        
        np.save("specie_{}_model_no_{}_predictions_test.npy".format(specie, model_no),  predictions)
        np.save("specie_{}_true_test.npy".format(specie), truth)
        
        np.save("membership_specie_{}_model_{}_one_hot_train.npy".format(specie, model_no), np.hstack(hotencode_array))
        np.save("structure_names_specie_{}_model_{}".format(specie,model_no),np.array(structure_list))
        
        #----- save model ----
        torch.save(swa_model.state_dict(), "large_model_{}_{}_float64_opt.pth".format(specie,model_no))
        print("Saved PyTorch Model State to large_model_{}_{}_float64_opt.pth".format(specie,model_no))
        
        #----- save scalers -----
        with open("scaler_features_{}_model_no_{}_float64_opt.pkl".format(specie,model_no),"wb") as fg:
            pickle.dump(scaler_X, fg)

        with open("scaler_labels_{}_model_no_{}_float64_opt.pkl".format(specie,model_no),"wb") as fg:    
            pickle.dump(scaler_y, fg)
            

        
        #----- append to 
        final_results.append(final_result)
        final_predictions.append(predictions)
        

        
    #------ save hypers -------
    with open(str(specie) + "_hypers.json", 'w') as fileob:
        
        json.dump(int_hyp, fileob)    
    #-----add to error dict-----
    errors[specie] = final_results
    

    
#---save final results-----
with open("errors_RR_mine.json", 'w') as fileob:
    json.dump(errors, fileob) 
