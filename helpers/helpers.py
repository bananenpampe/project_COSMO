import numpy as np
from ase.io import read
import itertools


def grouper(n, iterable):
    """Helper function that yields an iterable in chunks of n
    """
    #from https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def return_relative_inds(frames, selected_ids, atom_type):
    #TODO: Test this!
    """helper function that returns a numpy array of shape (N_environments_selected, 2)
       entries contain CSD identifier and 
    """
    
    accum_numbers = np.array([len(frame) for frame in frames])

    #bins with accumulated count of atomic environments
    bins = np.cumsum(accum_numbers)

    #array with atomic indices
    frame_names = np.array([frame.info["NAME"] for frame in frames])

    #atomic numbers of all the cumulated atomic environments
    atomic_numbers = np.concatenate([frame.numbers for frame in frames]) 

    #atomic numbers split environments 
    all_species = np.unique(atomic_numbers).tolist()
    number_loc = {i: np.where(atomic_numbers == i)[0] for i in all_species}

    #we get from a X_raw of one atomtype of environment the selected ids (atom_type, selected_ids)
    #first, we convert them into total ids:
    number_loc_absolute = number_loc[atom_type][selected_ids]

    #now convert them into binned environments
    in_bin = np.digitize(number_loc_absolute, bins)
    in_bin_names = frame_names[in_bin]

    #what is left are relative selections within structure (1st in structure A, 0th in structure B)
    #how to? substract in bin[0] with 0, substract in bin[1] with accum[0] in bin[2] with accum[1] etc....
    #add accum_numbers_bin[0] to all

    #find a better solution for this, this is not smart
    bins_concat = np.concatenate([np.array([0]),bins])
    inds_relative = number_loc_absolute - bins_concat[in_bin]

    #finally return dict/np.array with the selected ids
    return np.vstack((in_bin_names,inds_relative)).T

def filter_by_status(frames, status=["PASSING"]):
    """Helper function that filters structures by info dict 'STATUS' entry
    """
    return [frame for frame in frames if frame.info['STATUS'] in status]

def retrieve_features(calculator, chunk):
    """helper function that allows for calling a class method in joblib
    """
    return calculator.transform(chunk).get_features(calculator)


def get_features_in_parallel(frames,calculator,blocksize=100,n_jobs=-1):
    """helper function that returns the features of a calculator (from calculator.transform())
       in parallel
    """
    #for np.concatenate. arrays in list should all have same shape
    hypers = calculator.hypers
    hypers["expansion_by_species_method"] = "user defined"
    hypers["global_species"] = get_all_species(frames).tolist()
    calculator.update_hyperparameters(**hypers)
    return np.concatenate(Parallel(n_jobs=2)(delayed(retrieve_features)(calculator, chunk)\
                                              for chunk in grouper(blocksize,frames)))

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
        atom.wrap(eps=1e-12)
        
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
    
    
    
