from loader import load_data
from rascal.representations import SphericalInvariants as SOAP

train_structures, test_structures, train_properties, test_properties = load_data("./make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                    "./make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",selected_species=100,random_subsample_test=200)

SOAP_HYPERS = {
    "soap_type": "PowerSpectrum",
    "interaction_cutoff": 4.5,
    "max_radial": 9,
    "max_angular": 9,
    "gaussian_sigma_constant": 0.1,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.5,
    "radial_basis": "GTO",
    "expansion_by_species_method": "user defined",
    "global_species": [1, 6, 7, 8, 16],
    "compute_gradients": False,
    "normalize": True,
}

calculator = SOAP(**SOAP_HYPERS)
print(calculator.transform(train_structures).get_features(calculator))
