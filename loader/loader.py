from helpers import filter_by_status
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species, mask_center_atoms_by_id
from ase.io import read
import numpy as np
from helpers import return_relative_inds


def load_data(TRAINPATH,TESTPATH,physical_property="cs_iso",filter_by="PASSING",filter_by_test=None,selected_ids=None,\
              selected_species=None,random_subsample_train=None,random_subsample_test=None):
    
    #load train and teststructures
    train_structures = read(TRAINPATH,format="extxyz",index=":")
    test_structures = read(TESTPATH,format="extxyz",index=":")
    
    # filter them by status, this might be swapped with masking step, but ID selection 
    # should not be done on unfiltered structures
    
    if filter_by is not None:
        train_structures = filter_by_status(train_structures,status=filter_by)
        
    if filter_by_test is not None:
        test_structures = filter_by_status(test_structures,status=filter_by_test)
    
    #wrap train structures
    for structure in train_structures:
        structure.wrap(eps=1e-12)
    
    #wrap test structures
    for structure in test_structures:
        structure.wrap(eps=1e-12)
    
    
    if (selected_ids is None) and (selected_species is None):
        #pass here because no masking is necessary
        pass
    
    
    elif (selected_ids is not None) and (selected_species is None):
        raise NotImplementedError()
    
    #select only species
    elif (selected_ids is None) and (selected_species is not None):
        
        #select species by atomic numbers
        for structure in train_structures: mask_center_atoms_by_species(structure,species_select=[selected_species])
        for structure in test_structures: mask_center_atoms_by_species(structure,species_select=[selected_species])
    
    elif (selected_ids is not None) and (selected_species is not None):
        #use helper function to get dict/array of the structures
        #make dict that contains list indices ?
        dict_CSD_identifiers = {i.info["NAME"]: n for n,i in enumerate(train_structures) }
        relative_inds = return_relative_inds(train_structures,selected_ids,selected_species)
        

        for structure in train_structures: structure.arrays["center_atoms_mask"] = np.full((len(structure)),False)
        
        for mask in relative_inds:
            frame_number = dict_CSD_identifiers[mask[0]]
            mask_center_atoms_by_id(train_structures[frame_number],id_select=int(mask[1]))
        
        for structure in test_structures: mask_center_atoms_by_species(structure,species_select=[selected_species])
        
    else:
        raise ValueError()
    
    #do subsample here:
    
    if random_subsample_train is not None:
        ids = list(range(len(train_structures)))
        np.random.shuffle(ids)
        ids = ids[:random_subsample_train]
        train_structures = [train_structures[ii] for ii in ids[:random_subsample_train]]
    
    if random_subsample_test is not None:
        ids = list(range(len(test_structures)))
        np.random.shuffle(ids)
        ids = ids[:random_subsample_test]
        test_structures = [test_structures[ii] for ii in ids[:random_subsample_test]]
        
    #extract selected property
    if (selected_ids is None) and (selected_species is None):
        train_properties = np.array([tensor for structure in train_structures for tensor in structure.arrays[physical_property]])
        test_properties = np.array([tensor for structure in test_structures for tensor in structure.arrays[physical_property]])
    else:
        train_properties = np.array([tensor for structure in train_structures for tensor in structure.arrays[physical_property][structure.arrays["center_atoms_mask"]]])
        test_properties = np.array([tensor for structure in test_structures for tensor in structure.arrays[physical_property][structure.arrays["center_atoms_mask"]]])
    
    
    return train_structures, test_structures, train_properties, test_properties