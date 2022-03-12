from loader import load_data
import sys

sys.path.append("/home/kellner/packages/project_COSMO/")
sys.path.append("/home/kellner/packages/project_COSMO/loader")

from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.kernel_ridge import KernelRidge
from skcosmo.model_selection import atom_groups_by_frame
from sklearn.model_selection import GroupKFold
from feature_utils.parallel import get_features_in_parallel, get_optimal_radial_basis_hypers_parallel
from rascal.representations import SphericalInvariants as SOAP


train_structures, test_structures, train_properties, test_properties = load_data("/home/kellner/packages/project_COSMO/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                    "/home/kellner/packages/project_COSMO/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",selected_species=6)


hypers = dict(soap_type="PowerSpectrum",
              interaction_cutoff=4.643,
              max_radial=8,
              max_angular=8,
              gaussian_sigma_constant=0.179,
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
              cutoff_function_parameters= dict(rate=1.968,
                    scale= 2.601,
                    exponent=4.698
                        ),
              compute_gradients=False,
              expansion_by_species_method="user defined",
              global_species=[1, 6, 7, 8, 16]
              )

hypers = get_optimal_radial_basis_hypers_parallel(hypers,train_structures,expanded_max_radial=20)
Xtrain = get_features_in_parallel(train_structures,SOAP,hypers)
Xtest = get_features_in_parallel(test_structures,SOAP,hypers)

structure_groups = atom_groups_by_frame(train_structures)



pipe = Pipeline([
    ('model', KernelRidge(kernel_params={"n_jobs":-1}))
])

general_kernel_space = {
    "model":[KernelRidge(kernel_params={"n_jobs":-1})],
    "model__kernel": Categorical(["linear","poly","rbf","laplacian"]),
    "model__gamma": Real(1e-6, 1e+1, prior='log-uniform'),
    "model__degree": Integer(1,8),
    "model__alpha": Real(1e-6, 1e+4, prior='log-uniform'),
    "model__coef0": Integer(0,1)
}

opt = BayesSearchCV(pipe,general_kernel_space,n_iter=100,cv=GroupKFold(n_splits=3),n_jobs=-1,scoring="neg_mean_squared_error",verbose=2)

opt.fit(Xtrain, train_properties, groups=structure_groups)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(Xtest, test_properties))
print("best params: %s" % str(opt.best_params_))