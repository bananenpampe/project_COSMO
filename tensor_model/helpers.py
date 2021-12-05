import numpy as np
from ase.io import read

def load_CSD_data(PATH, prop_string, random_subsample=None):
    """Helper function that loads the CSD-2K and CSD-500 dataset
       The CSD-X dataset are .txt files of joined extended-xyz files. Where unit cell parameters are given in the comment line
       And atom wise calculated GIPAW shifts, are stored in additional atom-wise collums. first additional colum is GIPAW.
       In CSD-500 another column is given with the 
    
    Parameters
    ----------
    PATH             : string
                       absolut path of the dataset .txt file
    
    prop_string      : string
                       Key string of property array that is stored in atoms.arrays 
                       
    random_subsample : int < dataset_size
                       returns a random subsample of the dataset with N(random_subsample) entries
                       
    
    Returns
    -------
    structures : list of ase.atoms ojects
                 wrapped structures of the dataset
    
    shifts     : numpy array of size (N_environments,) or (N_environments,2)
                 shifts of the individual nuclei
    """
    
    
    
    structures = read(PATH,format="extxyz",index=":")
    

        
    for atom in structures:
        atom.wrap()
        
    if random_subsample is not None:
        ids = list(range(len(structures)))
        np.random.shuffle(ids)
        train_ids = ids[:random_subsample]
        structures_subsample = [structures[ii] for ii in ids[:random_subsample]]
        shifts_subsample = np.concatenate([atoms.arrays[prop_string] for atoms in structures_subsample])
        return structures_subsample, shifts_subsample
        
    else:
        shifts = np.concatenate([atoms.arrays[prop_string] for atoms in structures])
        return structures, shifts


def make_element_wise_environments(calculator,frames,y=None,select=False):
    """Returns shifts and environments of only one atomtype from the atoms in frames. 
       Or returns a dictionary of atomic-type-wise 
    
    Parameters
    ----------
    calculator : rascal.representations calculator object
                 calculator object with hyperparameters 
    
    frames     : list of ase.atoms objects
                 wrapped structures of the dataset
    
    y          : numpy array of shape (N_environments,X)
                 array of atomic properties
                 
    select     : int
                 atomic number to select atomic species
    Returns
    -------
    
    X_element_wise: dict or numpy.array
                    either dict with atomic numbers keys containing the representations in numpy array, 
                    or numpy array with representations of the selected atomic species
    y_element_wise: dict or numpy.array
                    either dict with atomic numbers keys containing the shifts in numpy arrays, 
                    or numpy array with representations of the selected atomic species
    
    """
    
    
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
    

def return_element_wise_environments(descriptors,calculator,frames,y=None,select=False):
    """Returns shifts and environments of only one atomtype from the atoms in frames. 
       Or returns a dictionary of atomic-type-wise 
    
    Parameters
    ----------
    calculator : rascal.representations calculator object
                 calculator object with hyperparameters 
                 
    descriptors: numpy array of shape (N_environments,XX)
                 array containing precomputet descriptor features
    
    frames     : list of ase.atoms objects
                 wrapped structures of the dataset
    
    y          : numpy array of shape (N_environments,X)
                 array of atomic properties
                 
    select     : int
                 atomic number to select atomic species
    Returns
    -------
    
    X_element_wise: dict or numpy.array
                    either dict with atomic numbers keys containing the representations in numpy array, 
                    or numpy array with representations of the selected atomic species
    y_element_wise: dict or numpy.array
                    either dict with atomic numbers keys containing the shifts in numpy arrays, 
                    or numpy array with representations of the selected atomic species
    
    """
    
    
    #get unique elements 
    y_element_wise = {}
    X_element_wise = {}
    
    atoms_list = calculator.transform(frames)
    X_repr = descriptors
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
    
    
    
