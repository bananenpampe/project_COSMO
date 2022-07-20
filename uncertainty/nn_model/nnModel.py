from torch import nn
import torch
import pickle 
import json
from torch.optim.swa_utils import AveragedModel, SWALR
from lshiftml.feature_utils.parallel import get_features_in_parallel
from rascal.representations import SphericalInvariants as SOAP
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
from skcosmo.preprocessing import StandardFlexibleScaler
import numpy as np
import ase


class NeuralNetwork(nn.Module):
    def __init__(self,n_hidden,activation,num_input):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        #This is the pretrained_RELU stack
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_input,n_hidden),
            nn.LayerNorm(n_hidden),
            activation(),
            nn.Linear(n_hidden,1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_bak(nn.Module):
    def __init__(self,l1):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Dropout(p=0.2),
            nn.Linear(num_input, l1),
            nn.LayerNorm(l1),
            nn.ReLU(),
            nn.Linear(l1,1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def NN_factory(**kwargs):
    #this is necessary because pytorch changes the layer state dict keys when using averaged models
    torch.set_default_tensor_type(torch.DoubleTensor)
    model = NeuralNetwork(**kwargs)
    return AveragedModel(model)
    
    

class ShiftMLNN:
    
    DEFAULT_MODEL_PATH = {"v1":"/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/NN_model_data/NN_v1/large_model_{}_{}_float64_opt.pth",
                         "v2":"/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/NN_model_data/NN_v2/large_model_{}_{}_float64_opt.pth"}
    DEFAULT_FEATURE_SCALER_PATH = {"v1":"/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/NN_model_data/NN_v1/scaler_features_{}_model_no_{}_float64_opt.pkl",
                                  "v2":"/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/NN_model_data/NN_v2/scaler_features_{}_model_no_{}_float64_opt.pkl"}
    DEFAULT_LABEL_SCALER_PATH = {"v1":"/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/NN_model_data/NN_v1/scaler_labels_{}_model_no_{}_float64_opt.pkl",
                                  "v2":"/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/NN_model_data/NN_v2/scaler_labels_{}_model_no_{}_float64_opt.pkl"
                                }
    DEFAULT_HYPERS_PATH = {"v1":"/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/NN_model_data/NN_v1/{}_hypers.json","v2":"/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/NN_model_data/NN_v2/{}_hypers.json"}
    
    default_architecture_dict = {"v2":{1:{"n_hidden":512,"activation":nn.Softplus,"num_input":3780},
                                 6:{"n_hidden":256, "activation":nn.Tanh,"num_input":3780},
                                 7:{"n_hidden":16,"activation":nn.Softplus,"num_input":3780},
                                 8:{"n_hidden":512,"activation":nn.Tanh,"num_input":3780}},
                                 "v1":{1:{"n_hidden":256,"activation":nn.ReLU,"num_input":60},
                                      6:{"n_hidden":512,"activation":nn.ReLU,"num_input":60},
                                      7:{"n_hidden":1024,"activation":nn.ReLU,"num_input":60},
                                      8:{"n_hidden":512,"activation":nn.ReLU,"num_input":60}
                                }}
    
    def __init__(self,trained_for=[1,6,7,8],defined_for=[1,6,7,8,16],bodyorder="v1",model_architecture=NN_factory,architecture_dict=None,n_models=16,MODEL_PATH=None \
                 ,FEATURE_SCALER_PATH=None, LABEL_SCALER_PATH=None,HYPERS_PATH=None):
        
        """initializes model by loading state dicts,
           initializing pytorch model objects 
           and loading skcosmo flexible scalers
           
           bodyorder hyperparameter is ignored when choosing own architecture
        """
        
        self.species = trained_for
        self.defined = defined_for
        self.label_scalers = {k:[] for k in trained_for}
        self.feature_scalers = {k:[] for k in trained_for}
        #self.feature_means
        #self.feature_vars
        self.models = {k:[] for k in trained_for}
        self.hypers = {k:None for k in trained_for}
        
        self._MODEL_PATH = self.DEFAULT_MODEL_PATH[bodyorder] if MODEL_PATH is None else MODEL_PATH
        self._FEATURE_SCALER_PATH = self.DEFAULT_FEATURE_SCALER_PATH[bodyorder] if FEATURE_SCALER_PATH is None else FEATURE_SCALER_PATH
        self._LABEL_SCALER_PATH = self.DEFAULT_LABEL_SCALER_PATH[bodyorder] if LABEL_SCALER_PATH is None else LABEL_SCALER_PATH
        self._HYPERS_PATH = self.DEFAULT_HYPERS_PATH[bodyorder] if HYPERS_PATH is None else HYPERS_PATH
        
        #this is incredibly stupid and I should have never done that
        self._architecture_dict = self.default_architecture_dict[bodyorder] if architecture_dict is None else architecture_dict
        
        for specie in trained_for:
            
            with open(self._HYPERS_PATH.format(specie),"r") as fg:
                hypers = json.load(fg)
            
            self.hypers[specie] = hypers
            
            for n in range(n_models):
                
                model = model_architecture(**self._architecture_dict[specie])
                state_dict = torch.load(self._MODEL_PATH.format(specie,n),map_location ='cpu')
                model.load_state_dict(state_dict)
                self.models[specie].append(model)
                
                #this is crap. replace by json?
                with open(self._FEATURE_SCALER_PATH.format(specie,n),"rb") as fg:
                    feature_scaler = pickle.load(fg)
                with open(self._LABEL_SCALER_PATH.format(specie,n),"rb") as fg:    
                    label_scaler = pickle.load(fg)
                    
                self.feature_scalers[specie].append(feature_scaler)
                self.label_scalers[specie].append(label_scaler)
        
    
    def predict(self,frames,predict_for=None,output="average"):
        
        #assuming wrapped frames
        results = {}
        atomic_numbers = []
        
        is_single_frame = isinstance(frames,ase.atoms.Atoms)
        
        if is_single_frame:
            atomic_numbers = frames.numbers
        else:
            for frame in frames:
                atomic_numbers.append(frame.numbers)
            
        
        atomic_numbers = np.hstack(atomic_numbers)
        atomic_species = np.unique(atomic_numbers)
        
        for specie in atomic_species:
            if specie not in self.defined:
                raise NotImplementedError("Model not defined for specie {}".format(specie))            
        
        if predict_for is None:
            predict_for = self.species
            
        for specie in predict_for:
            if specie not in self.species:
                raise NotImplementedError("Model not trained for specie {}".format(specie))
        
        
        
        #avoids completely masked frames for rascal

            
        predict_for = np.intersect1d(atomic_species,predict_for)
        predict_for = [int(specie) for specie in predict_for]
                    
        for specie in predict_for:
            if is_single_frame:
                frames.arrays.pop("center_atoms_mask",None)
                mask_center_atoms_by_species(frames,species_select=[specie])
            else:
                for frame in frames: 
                    frame.arrays.pop("center_atoms_mask",None)
                    mask_center_atoms_by_species(frame,species_select=[specie])
            

            if is_single_frame: 
                soap = SOAP(**self.hypers[specie])
                Xpredict = soap.transform(frames).get_features(soap)
            else:
                Xpredict = get_features_in_parallel(frames,SOAP,self.hypers[specie])
            predictions = []
            
            for model,feature_scaler,label_scaler in \
                zip(self.models[specie],self.feature_scalers[specie],self.label_scalers[specie]):
                
                #TODO: does the scaler also work on tensors?

                Xpredict_scaled = feature_scaler.transform(Xpredict)
                
                #costs O(N): https://stsievert.com/blog/2017/09/07/pytorch/
                Xpredict_scaled = torch.from_numpy(Xpredict_scaled)

                with torch.no_grad(): #13 sec
                    Y_predict = model(Xpredict_scaled)

                Y_predict_inverse_rescaled = label_scaler.inverse_transform(Y_predict.numpy().reshape(-1,1))

                
                predictions.append(Y_predict_inverse_rescaled)
            
            results[specie] = np.hstack(predictions)
            
            if output == "average":
                average = np.mean(results[specie],axis=1)
                variance = np.var(results[specie],axis=1,ddof=1)
                results[specie] = np.vstack([average,variance]).T
            elif output == "raw":
                continue
            else:
                raise NotImplementedError
                
            #quick test to check whether copying in scaler worked
            #print(np.allclose(Xpredict,get_features_in_parallel(frames,SOAP,self.hypers[specie])))
        
        return results
        #gen feat
        
        #scale feat
        #if scaling is False: skip
        #
        #inverse_scale predictions
        #if inverse_scale is false: skip
        #
        #check option, return full predictions or average and variance
        #return framewise, or full ?