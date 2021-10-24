import numpy as np
from ase.io import read

def load_CSD_data(PATH, random_subsample=None):
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


def make_element_wise_environments(calculator,frames,y=None,select=False):
    
    #get unique elements 
    y_element_wise = {}
    X_element_wise = {}
    
    atoms_list = calculator.transform(frames)
    X_repr = atoms_list.get_features(calculator)
    
    elements = np.unique(atoms_list.get_representation_info()[:,2])
    

    for element in elements:
        
        ind = atoms_list.get_representation_info()[:,2] == element
        
        if y is not None:
            y_element_wise[element] = y[ind]
        X_element_wise[element] = X_repr[ind]
    
    #TODO: Change this not to loop over array
    if select is not None:
        return X_element_wise[select], y_element_wise[select] 
    else:
        return X_element_wise, y_element_wise