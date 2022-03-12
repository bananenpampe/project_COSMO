from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.base import TransformerMixin, BaseEstimator








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

    def fit(self, Xm, Ym, precomputed=False, X0=None):
        # this expects properties in the form [i, m] and features in the form [i, q, m]
        # in order to train a SA-GPR model the m indices have to be moved and merged with the i
        
        if precomputed is True:
            Xm_flat = Xm
        else:
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
    
    def predict(self, Xm, precomputed=False,X0=None):
        if precomputed is True:
            Y = super(SARidge, self).predict(Xm)
        else:
            Y = super(SARidge, self).predict(np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1])))
        return Y.reshape((-1, 2*self.L+1))
    

class SANystroem(Nystroem,TransformerMixin,BaseEstimator):
    #maybe add L handling later   
    def __init__(
        self,
        L,
        kernel="linear",
        *,
        gamma=None,
        coef0=None,
        degree=None,
        kernel_params=None,
        n_components=100,
        random_state=None,
        n_jobs=None,
        non_linear_kernel="rbf",
        L0_features=None
    ):
        
        if kernel != "linear":
            raise ValueError("Kernel must be linear")
        self.kernel = kernel
        self.L = L
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        
        
    def fit(self, X, y=None, base_indices=None):
        """
        Additions: flatting array
                   passing indices
        Fit estimator to data.
        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
            n_components = n_samples
            warnings.warn(
                "n_components > n_samples. This is not possible.\n"
                "n_components was set to n_samples, which results"
                " in inefficient evaluation of the full kernel."
            )

        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        
        if base_indices is not None:
            basis_inds = base_indices
        else:
            basis_inds = inds[:n_components]
        
        basis = X[basis_inds]
        
        #do the reshape here!!
        basis = np.moveaxis(basis, 2, 1).reshape((-1, basis.shape[1]))
        basis = self._validate_data(basis, accept_sparse="csr")

        basis_kernel = pairwise_kernels(
            basis,
            metric=self.kernel,
            filter_params=True,
            n_jobs=self.n_jobs,
            **self._get_kernel_params(),
        )

        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = basis_inds
        return self 
        
    def transform(self,Xm):
        #[i, q, m] -> [i,m,q] -> [i,m*q]
        Xm_flat = np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1]))
        KMN = super().transform(Xm_flat)
        # TODO: Check if this is correct and write test
        return KMN
        
    #fittransform will call child-class fit and transform ?