from ase.io import read
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
from rascal.representations import SphericalInvariants as SOAP
from feature_utils.parallel import get_features_in_parallel
import json 
from joblib import load
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from copy import deepcopy

model_fitted_for_species = set([1,6,7,8,16])

def small_model(READPATH,WRITEPATH,fit_species=None,nthreads=-1):
    
    trajframes = read(READPATH,format="extxyz",index=":")    
    for frame in trajframes: frame.wrap(eps=1e-12)
    
    if len(trajframes) == 0:
        print("empty frames")
        return
    
    species_indices = [frame.numbers for frame in trajframes]
    
    if not np.all(species_indices == trajframes[0].numbers):
        print("ordering or length has changed of frames")
        return
    
    species_indices = np.concatenate([frame.numbers for frame in trajframes])
    shifts = np.full(species_indices.shape, np.NaN)
    species = set(np.unique(species_indices))
    
    
    
    if not species.issubset(model_fitted_for_species):
        print("sorry, model has only been fitted for H,C,N,O and S containing structures")
        return
    
    if fit_species is None:
        fit_species = list(species)
        
    if not set(fit_species).issubset(species):
        print("species not contained in structures")
        return
    
    
    for specie in fit_species:
        fit_frames = deepcopy(trajframes)
    
        for frame in fit_frames: mask_center_atoms_by_species(frame,[int(specie)]) 
        
        f = open("./RR_this_work_models/hypers/{}_hypers.json".format(specie))
        hypers = json.load(f)

        model = load("./RR_this_work_models/{}_RR.joblib".format(specie))
        Xpredict = get_features_in_parallel(fit_frames,SOAP,hypers,n_cores=nthreads)
        Ypred = model.predict(Xpredict)
        shifts[species_indices==specie] = Ypred
    
    return species_indices, shifts