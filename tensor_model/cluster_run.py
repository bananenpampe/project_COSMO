from rascal.utils import (WignerDReal, ClebschGordanReal, 
                          spherical_expansion_reshape, spherical_expansion_conjugate,
                    lm_slice, real2complex_matrix, compute_lambda_soap, xyz_to_spherical, spherical_to_xyz)

from rascal.representations import SphericalExpansion, SphericalInvariants

import sys

sys.path.append("/home/kellner/packages/project_COSMO/")
sys.path.append("/home/kellner/packages/project_COSMO/loader")

import json
from loader import load_data
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import numpy as np

from feature_utils.parallel import get_features_in_parallel, get_optimal_radial_basis_hypers_parallel

import numpy as np

class SASplitter:
    """ CV splitter that takes into account the presence of "L blocks"
    associated with symmetry-adapted regression. Basically, you can trick conventional
    regression schemes to work on symmetry-adapted data y^M_L(A_i) by having the (2L+1)
    angular channels "unrolled" into a flat array. Then however splitting of train/test
    or cross validation must not "cut" across the M block. This takes care of that.
    """
    def __init__(self, L, cv=2):
        self.L = L
        self.cv = cv
        self.n_splits = cv

    def split(self, X, y, groups=None):

        ntrain = X.shape[0]
        if ntrain % (2*self.L+1) != 0:
            raise ValueError("Size of training data is inconsistent with the L value")
        ntrain = ntrain // (2*self.L+1)
        nbatch = (2*self.L+1)*(ntrain//self.n_splits)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for n in range(self.n_splits):
            itest = idx[n*nbatch:(n+1)*nbatch]
            itrain = np.concatenate([idx[:n*nbatch], idx[(n+1)*nbatch:]])
            yield itrain, itest

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    
class SARidge(Ridge):
    """ Symmetry-adapted ridge regression class """

    def __init__(self, L, alpha=1, alphas=None, cv=2, solver='auto',
                 fit_intercept=False, scoring='neg_root_mean_squared_error'):
        self.L = L
        # L>0 components have zero mean by symmetry
        if L>0:
            fit_intercept = False
        self.cv = SASplitter(L, cv)
        self.alphas = alphas
        self.cv_stats = None
        self.scoring = scoring
        self.solver = solver
        super(SARidge, self).__init__(alpha=alpha, fit_intercept=fit_intercept, solver=solver)

    def fit(self, Xm, Ym, X0=None):
        # this expects properties in the form [i, m] and features in the form [i, q, m]
        # in order to train a SA-GPR model the m indices have to be moved and merged with the i

        Xm_flat = np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1]))
        Ym_flat = Ym.flatten()
        if self.alphas is not None:
            # determines alpha by grid search
            rcv = Ridge(fit_intercept=self.fit_intercept)
            gscv = GridSearchCV(rcv, dict(alpha=self.alphas), cv=self.cv, scoring=self.scoring)
            gscv.fit(Xm_flat, Ym_flat)
            self.cv_stats = gscv.cv_results_
            self.alpha = gscv.best_params_["alpha"]

        super(SARidge, self).fit(Xm_flat, Ym_flat)
    def predict(self, Xm, X0=None):

        Y = super(SARidge, self).predict(np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1])))
        return Y.reshape((-1, 2*self.L+1))
    
    
CG = ClebschGordanReal(lmax=2)
CG_lsoap = ClebschGordanReal(lmax=6)

for specie in [1,6,7,8]:
    
    train_structures, test_structures, train_properties, test_properties = load_data("/home/kellner/packages/project_COSMO/make_tensor_data/train_tensor/CSD-3k+S546_shift_tensors.xyz",\
                                                                                    "/home/kellner/packages/project_COSMO/make_tensor_data/test_tensor/CSD-500+104-7_shift_tensors.xyz",\
                                                                                     selected_species=specie,physical_property="cs_tensor")
    
    fh = open("/home/kellner/packages/shiftml-light/data/RR_this_work_models/hypers/{}_hypers.json".format(specie))
    hypers = json.load(fh)
    fh.close()
    
    
    hypers.pop("soap_type")
    hypers.pop("normalize")
    
    if specie in [6,7,8]:
        hypers['max_radial'] = 6
        hypers['max_angular'] = 6
    elif specie == 1:
        hypers['max_radial'] = 8
        hypers['max_angular'] = 8
    
    
        
    hypers = get_optimal_radial_basis_hypers_parallel(hypers,train_structures,expanded_max_radial=20)    
    
    train_properties_coupled = CG.couple(xyz_to_spherical(train_properties.reshape(-1,3,3)))
    test_properties_coupled = CG.couple(xyz_to_spherical(test_properties.reshape(-1,3,3)))

    #build full lambda soaps.
    
    X_train_spex = get_features_in_parallel(train_structures,SphericalExpansion,hypers)
    X_train_spex = 1e3*spherical_expansion_reshape(X_train_spex, **hypers)
    #print(X_train_spex.shape)
    X_test_spex = get_features_in_parallel(test_structures,SphericalExpansion,hypers)
    X_test_spex = 1e3*spherical_expansion_reshape(X_test_spex, **hypers)
    
    
    coupled_results_train = {(1,1):{0:None,1:None,2:None}}
    coupled_results_test = {(1,1):{0:None,1:None,2:None}}
    
    for l in [0,1,2]:

        parity = 1

        if l == 1:
            parity = -1
    
        X_l_train = compute_lambda_soap(X_train_spex, CG_lsoap, l, parity)
        X_l_test = compute_lambda_soap(X_test_spex, CG_lsoap, l, parity)
        
        X_l_train = X_l_train.reshape(X_l_train.shape[0],-1,X_l_train.shape[-1])
        X_l_test = X_l_test.reshape(X_l_test.shape[0],-1, X_l_test.shape[-1])
        

        Y_l_train = train_properties_coupled[(1,1)][l]
        Y_l_test = test_properties_coupled[(1,1)][l]
        
        model = SARidge(L=l, alphas=np.logspace(-6,4,11),cv=5)
        #print(X_l_train.shape)
        #print(Y_l_train.shape)
        model.fit(X_l_train,Y_l_train)               
        

        Y_l_predict_test = model.predict(X_l_test)
        Y_l_predict_train = model.predict(X_l_train)
        mae_test = np.mean(np.abs(Y_l_predict_test-Y_l_test))
        mae_train = np.mean(np.abs(Y_l_predict_train-Y_l_train))
        model_alpha = model.alpha
        
        coupled_results_train[(1,1)][l] = Y_l_predict_train
        coupled_results_test[(1,1)][l] = Y_l_predict_test
        
        print("Species: {}, l: {}, mae train: {}".format(specie,l,mae_train))
        print("Species: {}, l: {}, mae test: {}".format(specie,l,mae_test))
        print("Model alpha: {}".format(model_alpha))
        #do 5-fold CV
        #pass
    
    train_cartesian = spherical_to_xyz(CG.decouple(coupled_results_train))
    test_cartesian = spherical_to_xyz(CG.decouple(coupled_results_test))
    
    train_cartesian = train_cartesian.reshape(train_cartesian.shape[0],9)
    test_cartesian = test_cartesian.reshape(test_cartesian.shape[0],9)
    
    rmse_train = mean_squared_error(train_cartesian,train_properties,squared=False)
    rmse_test = mean_squared_error(test_cartesian,test_properties,squared=False)
    
    
    
    
    print("Species: {}, RMSE tensor cartesian train: {}".format(specie,rmse_train))
    print("Species: {}, RMSE tensor cartesian test: {}".format(specie,rmse_test))
    
    cs_iso_train = np.array([tensor for structure in train_structures for tensor in structure.arrays["cs_iso"][structure.arrays["center_atoms_mask"]]])
    cs_iso_test = np.array([tensor for structure in test_structures for tensor in structure.arrays["cs_iso"][structure.arrays["center_atoms_mask"]]])
    
    
    X_train_tensor = np.concatenate([coupled_results_train[(1,1)][l] for l in [0,1,2]],axis=1)
    X_test_tensor = np.concatenate([coupled_results_test[(1,1)][l] for l in [0,1,2]],axis=1)
    
    model_total = GridSearchCV(Ridge(),{"alpha":np.logspace(-9,4,14)},cv=5,scoring="neg_root_mean_squared_error")
    model_total.fit(X_train_tensor,cs_iso_train)
    c_iso_predict = model_total.predict(X_test_tensor)
    
    rmse_iso = mean_squared_error(c_iso_predict,cs_iso_test,squared=False)
    print("Species: {}, RMSE iso_pred test: {}".format(specie,rmse_iso))