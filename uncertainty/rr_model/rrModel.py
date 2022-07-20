import json
from torch.optim.swa_utils import AveragedModel, SWALR
from lshiftml.feature_utils.parallel import get_features_in_parallel
from rascal.representations import SphericalInvariants as SOAP
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
import numpy as np
import ase
import scipy.stats as stats
from tqdm.auto import tqdm

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import joblib


class ShiftMLRR:
    
    DEFAULT_MODEL_PATH = {"v1":"/ssd/scratch/kellner/COSMO_project/uncertainty/multi_ridge_v1/specie_{}_model_no_{}_name_clf",
                         "v2":"/ssd/scratch/kellner/COSMO_project/uncertainty/multi_ridge_v2/{}_{}_name_clf"}
    DEFAULT_HYPERS_PATH = {"v1":"/ssd/scratch/kellner/COSMO_project/uncertainty/multi_ridge_v1/specie_{}_model_no_0_hypers.json","v2":"/ssd/scratch/kellner/COSMO_project/uncertainty/multi_ridge_v2/{}_0_hypers.json"}
    
    
    def __init__(self,trained_for=[1,6,7,8],defined_for=[1,6,7,8,16],bodyorder="v1",n_models=16,MODEL_PATH=None \
                 ,HYPERS_PATH=None):
        
        """initializes model by loading sklearn models,
           bodyorder hyperparameter is ignored when choosing own architecture
        """
        
        self.species = trained_for
        self.defined = defined_for

        self.models = {k:[] for k in trained_for}
        self.hypers = {k:None for k in trained_for}
        
        self._MODEL_PATH = self.DEFAULT_MODEL_PATH[bodyorder] if MODEL_PATH is None else MODEL_PATH
        self._HYPERS_PATH = self.DEFAULT_HYPERS_PATH[bodyorder] if HYPERS_PATH is None else HYPERS_PATH
        
        
        for specie in trained_for:
            
            with open(self._HYPERS_PATH.format(specie),"r") as fg:
                hypers = json.load(fg)
            
            self.hypers[specie] = hypers
            
            for n in range(n_models):
                model = joblib.load(self._MODEL_PATH.format(specie,n))
                self.models[specie].append(model)
        
    
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
            
            for model in self.models[specie]:
                
                Y_predict = model.predict(Xpredict)
                predictions.append(Y_predict)
            
            results[specie] = np.vstack(predictions).T
            
            print(results[specie].shape)
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


class ShiftMLRR_dropout(ShiftMLRR):
    
    def __init__(self,*args,**kwargs):
        
        self.eps = 0.01
        self.dropout = kwargs.pop('dropout')
        self.noise = kwargs.pop('noise')
        self.renorm = kwargs.pop('renorm')
        self.dropout_probability = kwargs.pop('dropout_probability') 
        self.dropout_runs = kwargs.pop('dropout_runs')
        self.scale = 1/(1-self.dropout_probability) #rescale feature values
        
        super().__init__(*args,**kwargs)
        """initializes model by loading sklearn models,
           bodyorder hyperparameter is ignored when choosing own architecture
        """
        if kwargs["bodyorder"] == "v1":
            raise NotImplementedError
        
        for specie in self.species:
            with open("/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/RR_model_data/RR_this_work_models/hypers/{}_hypers.json".format(specie),"r") as fg:
                int_hyp = json.load(fg)
            self.hypers[specie] = int_hyp
            
        for specie in self.species:
            self.models[specie] = [joblib.load("/ssd/scratch/kellner/ShiftML-Light/src/lshiftml/models/RR_model_data/RR_this_work_models/{}_RR.joblib".format(specie))]
    
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
            
            per_model_result_avg = []
            per_model_result_var = []            
            
            #Xpredict_ensure_equal = np.copy(Xpredict)
            for model in self.models[specie]:
                #nice implementation https://nico-curti.github.io/NumPyNet/NumPyNet/layers/dropout_layer.html
                
                per_run_result = []
                
                for run in tqdm(range(self.dropout_runs)):
                    #print("gen rng ....")
                    
                    
                    
                    if self.noise is True:
                        noise = np.random.uniform(low=0.95, high=1.05, size=Xpredict.shape)
                        X_masked_predict = Xpredict * noise
                    else:
                        X_masked_predict = np.copy(Xpredict)
                        
                    if self.dropout is True:
                        rnd_mask = np.random.uniform(low=0., high=1., size=Xpredict.shape) > self.dropout_probability
                        X_masked_predict = X_masked_predict * rnd_mask

                        if self.renorm is True:
                            
                            
                            X_masked_predict_norm = np.linalg.norm(X_masked_predict,axis=1)
                            X_masked_predict = X_masked_predict / X_masked_predict_norm[:,np.newaxis]
                        #print(np.allclose(np.linalg.norm(X_masked_predict,axis=1),np.ones(X_masked_predict.shape[0])))
                    
                    #print("predicting ....")
                    Y_predict = model.predict(X_masked_predict)
                    per_run_result.append(Y_predict)
                    
                    """
                    print(X_masked_predict[100,:200:8])
                    print(Xpredict[100,:200:8])
                    print(np.allclose(Xpredict_ensure_equal,Xpredict))
                    print(np.allclose(X_masked_predict,Xpredict))"""
                per_run_result = np.vstack(per_run_result).T
                
                run_average = np.mean(per_run_result,axis=1)
                run_variance = np.var(per_run_result,axis=1,ddof=1)
                
                per_model_result_avg.append(run_average)
                per_model_result_var.append(run_variance)
            
            if len(self.models[specie]) == 1:
                per_model_result_avg = per_run_result 
                per_model_result_var = None
            else:
                per_model_result_avg = np.vstack(per_model_result_avg).T
                per_model_result_var = np.vstack(per_model_result_var).T
            
            results[specie] = {"avg":per_model_result_avg,"var":per_model_result_var}
            
            if output == "average":
                continue
            
            elif output == "raw":
                continue
            else:
                raise NotImplementedError
                
            #quick test to check whether copying in scaler worked
            #print(np.allclose(Xpredict,get_features_in_parallel(frames,SOAP,self.hypers[specie])))
        
        return results