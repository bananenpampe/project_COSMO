"""Bagging meta-estimator."""

# Author: Gilles Louppe <g.louppe@gmail.com>
# License: BSD 3 clause


import itertools
import numbers
import numpy as np
from abc import ABCMeta, abstractmethod
from warnings import warn
from functools import partial

from joblib import Parallel

from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state, column_or_1d, deprecated
from sklearn.utils import indices_to_mask
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter, check_is_fitted, _check_sample_weight
from sklearn.utils.fixes import delayed
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._bagging import _parallel_predict_regression, BaseBagging
from joblib import Parallel
from sklearn.utils.fixes import delayed
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from sklearn.linear_model import Ridge



__all__ = ["BaggingClassifier", "BaggingRegressor"]

MAX_INT = np.iinfo(np.int32).max

def small_correction(alpha,M):
    return np.sqrt(-1/M + (M-3)/(M-1) * alpha**2)

def rescale_a_b_g(ystd,alpha,beta,gamma):
    return np.sqrt(alpha**2 * ystd**(gamma+2) + beta**2)  

def rescale_a(ystd,alpha):
    return np.sqrt(alpha**2 * ystd**2)

def neglog_likelihood(ystd,ymeans,ytrue):
    ll = np.mean(-0.5*np.log(2*np.pi*(ystd**2)) - (ymeans-ytrue)**2/(2*(ystd**2)))
    return -ll

def neglog_likelihood_rescale(x,ystd,ymeans,ytrue,rescale,nll=neglog_likelihood):
    ystd = rescale(ystd,*x)
    return nll(ystd,ymeans,ytrue)

def dimensionless_coeff(LLworst,LLbest,LLactual):
    return max((LLworst-LLactual),0)/(LLworst-LLbest)*100

def _generate_indices(random_state, bootstrap, n_population, n_samples, groups=None):
    """Draw randomly sampled indices."""
    # Draw sample indices
    
    if groups is not None:
        
        frac = n_samples/n_population
        
        if bootstrap:
            group_labels = np.unique(groups)
            N_groups = len(group_labels)
            n_to_sample = int(np.around(frac*N_groups))
            
            #assuming equally spaced groups            
            indices_groups = random_state.randint(min(group_labels), max(group_labels), n_to_sample)
            
            indices = np.vstack([np.argwhere(groups == g_ind) for g_ind in indices_groups]).flatten()
            
            return indices
        
        else:
            if frac == 1.0:
                indices = np.arange(n_population)
                np.random.shuffle(indices)
            else:
                splitter = GroupShuffleSplit(n_splits = 2, train_size=frac)
                dummy_ind = np.arange(n_population)
                indices, other = next(splitter.split(dummy_ind,dummy_ind,groups))
            
    else:
        if bootstrap:
            indices = random_state.randint(0, n_population, n_samples)
        else:
            indices = sample_without_replacement(
                n_population, n_samples, random_state=random_state
            )

    return indices


def _generate_bagging_indices(
    random_state,
    bootstrap_features,
    bootstrap_samples,
    n_features,
    n_samples,
    max_features,
    max_samples,
    groups
):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_indices(
        random_state, bootstrap_features, n_features, max_features
    )
    sample_indices = _generate_indices(
        random_state, bootstrap_samples, n_samples, max_samples, groups
    )

    return feature_indices, sample_indices


def _parallel_build_estimators(
    n_estimators,
    ensemble,
    X,
    y,
    groups,
    sample_weight,
    seeds,
    total_n_estimators,
    verbose,
    check_input,
):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_, "sample_weight")
    has_check_input = has_fit_parameter(ensemble.base_estimator_, "check_input")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        if has_check_input:
            estimator_fit = partial(estimator.fit, check_input=check_input)
        else:
            estimator_fit = estimator.fit

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            bootstrap,
            n_features,
            n_samples,
            max_features,
            max_samples,
            groups
        )

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator_fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator_fit(X[indices][:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(X[:, features])

            if n_classes == len(estimator.classes_):
                proba += proba_estimator

            else:
                proba[:, estimator.classes_] += proba_estimator[
                    :, range(len(estimator.classes_))
                ]

        else:
            # Resort to voting
            predictions = estimator.predict(X[:, features])

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba


def _parallel_predict_log_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute log probabilities within a job."""
    n_samples = X.shape[0]
    log_proba = np.empty((n_samples, n_classes))
    log_proba.fill(-np.inf)
    all_classes = np.arange(n_classes, dtype=int)

    for estimator, features in zip(estimators, estimators_features):
        log_proba_estimator = estimator.predict_log_proba(X[:, features])

        if n_classes == len(estimator.classes_):
            log_proba = np.logaddexp(log_proba, log_proba_estimator)

        else:
            log_proba[:, estimator.classes_] = np.logaddexp(
                log_proba[:, estimator.classes_],
                log_proba_estimator[:, range(len(estimator.classes_))],
            )

            missing = np.setdiff1d(all_classes, estimator.classes_)
            log_proba[:, missing] = np.logaddexp(log_proba[:, missing], -np.inf)

    return log_proba


def _parallel_decision_function(estimators, estimators_features, X):
    """Private function used to compute decisions within a job."""
    return sum(
        estimator.decision_function(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )


def _parallel_predict_regression(estimators, estimators_features, X):
    """Private function used to compute predictions within a job."""
    return sum(
        estimator.predict(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    First, we check the first fitted estimator if available, otherwise we
    check the base estimator.
    """
    return lambda self: (
        hasattr(self.estimators_[0], attr)
        if hasattr(self, "estimators_")
        else hasattr(self.base_estimator, attr)
    )


class BaseBagging(BaseEnsemble, metaclass=ABCMeta):
    """Base class for Bagging meta-estimator.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(base_estimator=base_estimator, n_estimators=n_estimators)

        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self._groups = None

    def fit(self, X, y, groups = None, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            multi_output=True,
        )
        
        self._groups = groups
        
        return self._fit(X, y, self._groups, self.max_samples, sample_weight=sample_weight)

    def _parallel_args(self):
        return {}

    def _fit(
        self,
        X,
        y,
        groups=None,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=True,
    ):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.
        max_depth : int, default=None
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        check_input : bool, default=True
            Override value used when fitting base estimator. Only supported
            if the base estimator has a check_input parameter for fit function.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = self.max_features * self.n_features_in_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_in_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs=n_jobs, timeout=9999,verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                groups,
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
                check_input=check_input,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y(self, y):
        if len(y.shape) == 1 or y.shape[1] == 1:
            return column_or_1d(y, warn=True)
        else:
            return y

    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        for seed in self._seeds:
            # Operations accessing random_state must be performed identically
            # to those in `_parallel_build_estimators()`
            feature_indices, sample_indices = _generate_bagging_indices(
                seed,
                self.bootstrap_features,
                self.bootstrap,
                self.n_features_in_,
                self._n_samples,
                self._max_features,
                self._max_samples,
                self._groups
            )

            yield feature_indices, sample_indices

    @property
    def estimators_samples_(self):
        """
        The subset of drawn samples for each base estimator.
        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.
        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        return [sample_indices for _, sample_indices in self._get_estimators_indices()]

    # TODO: Remove in 1.2
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `n_features_` was deprecated in version 1.0 and will be "
        "removed in 1.2. Use `n_features_in_` instead."
    )
    @property
    def n_features_(self):
        return self.n_features_in_


class BaggingRegressor(RegressorMixin, BaseBagging):
    """A Bagging regressor.
    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.
    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.
    Read more in the :ref:`User Guide <bagging>`.
    .. versionadded:: 0.15
    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeRegressor`.
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see `bootstrap_features` for more
        details).
        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.
    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error. Only available if bootstrap=True.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.
    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.
    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    n_features_ : int
        The number of features when :meth:`fit` is performed.
        .. deprecated:: 1.0
            Attribute `n_features_` was deprecated in version 1.0 and will be
            removed in 1.2. Use `n_features_in_` instead.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    estimators_ : list of estimators
        The collection of fitted sub-estimators.
    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.
    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_prediction_` might contain NaN. This attribute exists only
        when ``oob_score`` is True.
    See Also
    --------
    BaggingClassifier : A Bagging classifier.
    References
    ----------
    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.
    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.
    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.
    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    Examples
    --------
    >>> from sklearn.svm import SVR
    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4,
    ...                        n_informative=2, n_targets=1,
    ...                        random_state=0, shuffle=False)
    >>> regr = BaggingRegressor(base_estimator=SVR(),
    ...                         n_estimators=10, random_state=0).fit(X, y)
    >>> regr.predict([[0, 0, 0, 0]])
    array([-2.8720...])
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def predict(self, X):
        """Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            reset=False,
        )

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
            )
            for i in range(n_jobs)
        )

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeRegressor())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few estimators were used "
                "to compute any reliable oob estimates."
            )
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)
        


def _parallel_predict_regression_uncertainty(estimators, estimators_features, X):
    """Private function used to compute predictions within a job."""
    return [estimator.predict(X[:, features])
        for estimator, features in zip(estimators, estimators_features)]


class UncertaintyEnsembleRegressor(BaggingRegressor):
    """
    def _generate_bagging_indices(
    random_state,
    bootstrap_features,
    bootstrap_samples,
    n_features,
    n_samples,
    max_features,
    max_samples,
    ):
        #Randomly draw feature and sample indices.
        # Get valid random state
        random_state = check_random_state(random_state)

        # Draw indices
        feature_indices = _generate_indices(
            random_state, bootstrap_features, n_features, max_features
        )
        sample_indices = _generate_indices(
            random_state, bootstrap_samples, n_samples, max_samples
        )

        return feature_indices, sample_indices

    """
    
    #for each estimator we call _generate_bagging_indices
    
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        
        
        self.rescaled = False
        
        self.alpha = 1
        self.beta = 0
        self.gamma = 0
        
        
        
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        
    
    def _check_rescaled(self):
        assert self.rescaled == True
    
    def _predict_uncertainty(self, X):
        """Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            reset=False,
        )

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression_uncertainty)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
            )
            for i in range(n_jobs)
        )
        
        #fix this later -> unroll nested lists
        
        gathered = [est for job_res in all_y_hat for est in job_res]
        
        return np.vstack(gathered).T
    
    def predict_uncertainty(self,X):
        
        """after rescaling using the internal validation set,
        function returns the mean and rescaled standard deviation of the model
        """
        
        self._check_rescaled()
        
        Ypred = self._predict_uncertainty(X)
        
        Ymean = Ypred.mean(axis=1)
        Ystd = Ypred.std(axis=1,ddof=1)
        
        Ystd = rescale_a_b_g(Ystd,self.alpha,self.beta,self.gamma)
        
        return Ymean, Ystd
        
    def rescale_external(self,Xval,Yval,Xtest,Ytest):
        
        Ypred_val = self._predict_uncertainty(Xval)
        Ypred_test = self._predict_uncertainty(Xtest)
            
        # get means and vars
        Ypred_val_mean = np.mean(Ypred_val,axis=1)
        Ypred_val_std = np.std(Ypred_val,axis=1,ddof=1)

        Ypred_test_mean = Ypred_test.mean(axis=1)
        Ypred_test_std = Ypred_test.std(axis=1,ddof=1)
        
        # minimize the non-linear rescaling objective
        min_res_full_a_b_g = minimize(neglog_likelihood_rescale,args=(Ypred_val_std,Ypred_val_mean,Yval,rescale_a_b_g),x0=np.array([1.,0.,0.]))
        min_res_full_a_b_g = basinhopping()
        alpha_a_b_g, beta_a_b_g, gamma_a_b_g = min_res_full_a_b_g["x"]
        
        #calculate the negative log likelihoods using various rescalings
        
        neglog_likelihood_no_rescale_test = neglog_likelihood(Ypred_test_std,Ypred_test_mean,Ytest)
        neglog_likelihood_a_b_g_rescale_test = neglog_likelihood_rescale([alpha_a_b_g, beta_a_b_g, gamma_a_b_g],Ypred_test_std,Ypred_test_mean,Ytest,rescale_a_b_g)
        
        #calculate RMSEs
        RMSE_test = mean_squared_error(Ypred_test_mean,Ytest,squared=False)
        RMSE_val = mean_squared_error(Ypred_val_mean,Yval,squared=False)
        
        #calculate test residuals
        errors_test = np.abs(Ypred_test_mean-Ytest)
        
        #determine best and worst NLLs
        NLL_worst = neglog_likelihood(RMSE_test*np.ones(Ypred_test_mean.shape),Ypred_test_mean,Ytest)
        NLL_best = neglog_likelihood(errors_test,Ypred_test_mean,Ytest)
        
        coeff_no_rescale = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_no_rescale_test)
        coeff_a_b_g_rescaled = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_a_b_g_rescale_test)
        
        
        
        #print a rescaling report:
        print("""RMSE test: {:.2f}\n
                 RMSE val: {:.2f}\n
                 NLL without rescale: {:.2f}\n
                 NLL with rescale(a,b,g): {:.2f}\n
                 NLL best: {:.2f}\n
                 NLL worst: {:.2f}\n
                 no rescale coeff: {:.2f}\n
                 rescale coeff (a,b,g): {:.2f}\n
                 a,b,g: {:.2f}, {:.2f}, {:.2f}\n""".format(RMSE_test,RMSE_val,\
                                                       neglog_likelihood_no_rescale_test, \
                                                      neglog_likelihood_a_b_g_rescale_test,\
                                                      NLL_best,NLL_worst,\
                                                      coeff_no_rescale, coeff_a_b_g_rescaled,float(alpha_a_b_g), float(beta_a_b_g), float(gamma_a_b_g)))
        
        
        
        self.rescaled = True
        self.alpha = alpha_a_b_g
        self.beta = beta_a_b_g
        self.gamma = gamma_a_b_g
        
        
    def rescale(self,Xtrain,Ytrain,Xtest,Ytest,pre_mask=None,n_missing=5):
        
        """ rescales the LL using an internal validation set
        """

        Ypred_train = self._predict_uncertainty(Xtrain)
        Ypred_test = self._predict_uncertainty(Xtest)
        
        if pre_mask is None:
            sample_mask = np.zeros((Xtrain.shape[0],self.n_estimators),dtype=bool)

            for n,sample_in_training in enumerate(self.estimators_samples_):
                sample_mask[sample_in_training,n] = True 
        else:
            sample_mask = pre_mask
            
        # get means
        Ypred_train_mean_full = np.mean(Ypred_train,axis=1)
        
        # get vars
        Ypred_train_std_full = np.std(Ypred_train,axis=1,ddof=1)
        
        # mask the prediction array
        Ypred_train_masked = np.ma.masked_array(data=Ypred_train,mask=sample_mask)
        
        # get where at least 5 points are missing
        # this will break if n_estimators << 10 or so 
        more_than_n_missing = np.sum(sample_mask,axis=1) < (self.n_estimators - n_missing)
        
        # get means,vars and predictions of masked array and where at least 5 are missing
        Ypred_train_mean_masked = Ypred_train_masked.mean(axis=1)
        Ypred_train_std_masked = Ypred_train_masked.std(axis=1,ddof=1)
        Ypred_train_mean_masked = Ypred_train_mean_masked[more_than_n_missing]
        Ypred_train_std_masked = Ypred_train_std_masked[more_than_n_missing]
        
        
        #test mean and var
        Ypred_test_std = Ypred_test.std(axis=1,ddof=1)
        Ypred_test_mean = Ypred_test.mean(axis=1)
        
        
        
        # minimize the linear rescaling objective and correct for small M
        min_res_full_a = minimize(neglog_likelihood_rescale,args=(Ypred_train_std_masked,Ypred_train_mean_masked,Ytrain[more_than_n_missing],
                                                                       rescale_a),x0=np.array([1.]))
        alpha_a = min_res_full_a["x"]
        alpa_a_small_M_corrected = small_correction(alpha_a,self.n_estimators)
        
        
        # minimize the non-linear rescaling objective
        min_res_full_a_b_g = minimize(neglog_likelihood_rescale,args=(Ypred_train_std_masked,Ypred_train_mean_masked,Ytrain[more_than_n_missing],rescale_a_b_g),x0=np.array([1.,0.,0.]))
        alpha_a_b_g, beta_a_b_g, gamma_a_b_g = min_res_full_a_b_g["x"]
        
        #calculate the negative log likelihoods using various rescalings
        
        neglog_likelihood_no_rescale_test = neglog_likelihood(Ypred_test_std,Ypred_test_mean,Ytest)
        neglog_likelihood_a_rescale_test = neglog_likelihood_rescale([alpha_a],Ypred_test_std,Ypred_test_mean,Ytest,rescale_a)
        neglog_likelihood_a_small_a_rescale_test = neglog_likelihood_rescale([alpa_a_small_M_corrected],Ypred_test_std,Ypred_test_mean,Ytest,rescale_a)
        neglog_likelihood_a_b_g_rescale_test = neglog_likelihood_rescale([alpha_a_b_g, beta_a_b_g, gamma_a_b_g],Ypred_test_std,Ypred_test_mean,Ytest,rescale_a_b_g)
        
        #calculate RMSEs
        RMSE_test = mean_squared_error(Ypred_test_mean,Ytest,squared=False)
        RMSE_train = mean_squared_error(Ypred_train_mean_full,Ytrain,squared=False)
        
        #calculate test residuals
        errors_test = np.abs(Ypred_test_mean-Ytest)
        
        #determine best and worst NLLs
        NLL_worst = neglog_likelihood(RMSE_test*np.ones(Ypred_test_mean.shape),Ypred_test_mean,Ytest)
        NLL_best = neglog_likelihood(errors_test,Ypred_test_mean,Ytest)
        
        coeff_no_rescale = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_no_rescale_test)
        coeff_a_rescaled = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_a_small_a_rescale_test)
        coeff_a_b_g_rescaled = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_a_b_g_rescale_test)
        
        
        
        #print a rescaling report:
        print("""RMSE test: {:.2f}\n
                 RMSE train: {:.2f}\n
                 NLL without rescale: {:.2f}\n
                 NLL with rescale(a): {:.2f}\n
                 NLL with resacle(a) small M correction: {:.2f}\n
                 NLL with rescale(a,b,g): {:.2f}\n
                 NLL best: {:.2f}\n
                 NLL worst: {:.2f}\n
                 no rescale coeff: {:.2f}\n
                 rescale coeff (a): {:.2f}\n
                 rescale coeff (a,b,g): {:.2f}\n
                 alpha rescale(a): {:.2f} \n 
                 a,b,g: {:.2f}, {:.2f}, {:.2f}\n""".format(RMSE_test,RMSE_train,\
                                                       neglog_likelihood_no_rescale_test, \
                                                      neglog_likelihood_a_rescale_test,\
                                                      neglog_likelihood_a_small_a_rescale_test,\
                                                      neglog_likelihood_a_b_g_rescale_test,\
                                                      NLL_best,NLL_worst,coeff_no_rescale,coeff_a_rescaled,\
                                                      coeff_a_b_g_rescaled,alpha_a[0],float(alpha_a_b_g), float(beta_a_b_g), float(gamma_a_b_g)))
        
        
        
        self.rescaled = True
        self.alpha = alpha_a_b_g
        self.beta = beta_a_b_g
        self.gamma = gamma_a_b_g
        
        
def combined_mean_std(pred_1,pred_2,std_1,std_2):
    """passing a (N_samples,) array of predictions, and (N_samples,) standard deviations
    """
    comb_mean = 0.5 * (pred_1 + pred_2)
    comb_std = np.sqrt(std_1**2 + std_2**2)
    
    return comb_mean, comb_std

def neglog_likelihood_rescale_dual(x,ystd_a,ystd_b,ymeans_a,ymeans_b,ytrue,rescale,nll=neglog_likelihood):
    x_a, x_b = np.split(x,2)
    ystd_a = rescale(ystd_a,*x_a)
    ystd_b = rescale(ystd_b,*x_b)
    
    ymeans_comb,ystd_comb = combined_mean_std(ymeans_a,ymeans_b,ystd_a,ystd_b)
    
    return nll(ystd_comb,ymeans_comb,ytrue)


class DualEnsemble:
    #make it simple for now
    
    
    def __init__(self,ensA,ensB):
        
        check_is_fitted(ensA)
        check_is_fitted(ensB)
        
        assert isinstance(ensA.__class__,UncertaintyEnsembleRegressor.__class__)
        assert isinstance(ensB.__class__,UncertaintyEnsembleRegressor.__class__)
        
        ensA.rescaled = True
        ensB.rescaled = True
        
        self.ensA = ensA
        self.ensB = ensB
        
        self.alpha_a = None
        self.alpha_b = None
        self.beta_a = None
        self.beta_b = None
        self.gamma_a = None
        self.gamma_b = None
        
        self.fitted = False
    
    def _check_is_fitted(self):
        if self.fitted is False:
            raise ValueError
            
    def predict(self,Xpred):
        self._check_is_fitted()
        
        Ymean_A, Ystd_A = self.ensA.predict_uncertainty(Xpred)
        Ymean_B, Ystd_B = self.ensB.predict_uncertainty(Xpred)
        
        Ystd_A = rescale_a_b_g(Ystd_A,self.alpha_a,self.beta_a,self.gamma_a)
        Ystd_B = rescale_a_b_g(Ystd_B,self.alpha_b,self.beta_b,self.gamma_b)
        
        Ymean_comb, Ystd_comb = combined_mean_std(Ymean_A, Ymean_B, Ystd_A, Ystd_B)
        
        return Ymean_comb, Ystd_comb
    
    def rescale_external(self,Xval,Yval,Xtest,Ytest):
        
        Ypred_val_a = self.ensA._predict_uncertainty(Xval)
        Ypred_val_b = self.ensB._predict_uncertainty(Xval)
        
        Ypred_test_a = self.ensA._predict_uncertainty(Xtest)
        Ypred_test_b = self.ensB._predict_uncertainty(Xtest)
            
        # get means and vars of val set
        Ypred_val_mean_a = np.mean(Ypred_val_a,axis=1)
        Ypred_val_std_a = np.std(Ypred_val_a,axis=1,ddof=1)
        Ypred_val_mean_b = np.mean(Ypred_val_b,axis=1)
        Ypred_val_std_b = np.std(Ypred_val_b,axis=1,ddof=1)

        # get means and vars of test set
        Ypred_test_mean_a = Ypred_test_a.mean(axis=1)
        Ypred_test_std_a = Ypred_test_a.std(axis=1,ddof=1)
        Ypred_test_mean_b = Ypred_test_b.mean(axis=1)
        Ypred_test_std_b = Ypred_test_b.std(axis=1,ddof=1)
        
        # predict results using no scalind
        Ypred_test_mean_no_rescale, Ypred_test_std_no_rescale = combined_mean_std(Ypred_test_mean_a,Ypred_test_mean_b,Ypred_test_std_a,Ypred_test_std_b)
        
        # neglog_likelihood_rescale_dual(x,ystd_a,ystd_b,ymeans_a,ymeans_b,ytrue,rescale,nll=neglog_likelihood):
        min_res_full_a_b_g = minimize(neglog_likelihood_rescale_dual,args=(Ypred_val_std_a,Ypred_val_std_b,\
                                                                           Ypred_val_mean_a,Ypred_val_mean_b,Yval,rescale_a_b_g),x0=np.array([1.,1.,1.,1.,1.,1.]))
        
        #get params from minimization result 
        alpha_a_b_g_a, beta_a_b_g_a, gamma_a_b_g_a, alpha_a_b_g_b, beta_a_b_g_b, gamma_a_b_g_b = min_res_full_a_b_g["x"]
        print(min_res_full_a_b_g["fun"])
        
        
        #linear rescale
        min_res_full_a = minimize(neglog_likelihood_rescale_dual,args=(Ypred_val_std_a,Ypred_val_std_b,\
                                                                           Ypred_val_mean_a,Ypred_val_mean_b,Yval,rescale_a),x0=np.array([1.,1.]))
        
        alpha_a_a, alpha_a_b = min_res_full_a["x"]
        
        alpha_a_a = small_correction(alpha_a_a,self.ensA.n_estimators)
        alpha_a_b = small_correction(alpha_a_b,self.ensB.n_estimators)
        
        #calculate the negative log likelihoods using various rescalings
        
        neglog_likelihood_no_rescale_test = neglog_likelihood(Ypred_test_std_no_rescale,Ypred_test_mean_no_rescale,Ytest)
        neglog_likelihood_a_rescale_test = neglog_likelihood_rescale_dual(np.array([alpha_a_a,\
                                                                               alpha_a_b]),\
                                                                                Ypred_test_std_a,Ypred_test_std_b,\
                                                                                Ypred_test_mean_a,Ypred_test_mean_b,Ytest,rescale_a)
        neglog_likelihood_a_b_g_rescale_test = neglog_likelihood_rescale_dual(np.array([alpha_a_b_g_a, beta_a_b_g_a, gamma_a_b_g_a,\
                                                                               alpha_a_b_g_b, beta_a_b_g_b, gamma_a_b_g_b]),\
                                                                                Ypred_test_std_a,Ypred_test_std_b,\
                                                                                Ypred_test_mean_a,Ypred_test_mean_b,Ytest,rescale_a_b_g)
        
        
        # rescaled stds
        Ypred_test_std_a_rescaled = rescale_a_b_g(Ypred_test_std_a,alpha_a_b_g_a, beta_a_b_g_a, gamma_a_b_g_a)
        Ypred_test_std_b_rescaled = rescale_a_b_g(Ypred_test_std_b,alpha_a_b_g_b, beta_a_b_g_b, gamma_a_b_g_b)
        
        Ypred_test_std_a_rescaled_a_only = rescale_a(Ypred_test_std_a,alpha_a_a)
        Ypred_test_std_b_rescaled_a_only = rescale_a(Ypred_test_std_b,alpha_a_b)
        
        #combined rescaled prediction
        Ypred_test_mean_rescaled, Ypred_test_std_rescaled = combined_mean_std(Ypred_test_mean_a,Ypred_test_mean_b,Ypred_test_std_a_rescaled,Ypred_test_std_b_rescaled)
        Ypred_test_mean_rescaled_a_only, Ypred_test_std_rescaled_a_only = combined_mean_std(Ypred_test_mean_a,Ypred_test_mean_b, \
                                                                                            Ypred_test_std_a_rescaled_a_only,Ypred_test_std_b_rescaled_a_only)
        
        #calculate RMSEs
        RMSE_test = mean_squared_error(Ypred_test_mean_rescaled,Ytest,squared=False)
        
        #calculate test residuals
        errors_test = np.abs(Ypred_test_mean_rescaled-Ytest)
        
        #determine best and worst NLLs
        NLL_worst = neglog_likelihood(RMSE_test*np.ones(Ypred_test_mean_rescaled.shape),Ypred_test_mean_rescaled,Ytest)
        NLL_best = neglog_likelihood(errors_test,Ypred_test_mean_rescaled,Ytest)
        
        coeff_no_rescale = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_no_rescale_test)
        coeff_a_rescaled = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_a_rescale_test)
        coeff_a_b_g_rescaled = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_a_b_g_rescale_test)
        
        
        
        #print a rescaling report:
        print("""RMSE test: {:.2f}\n
                 NLL without rescale: {:.2f}\n
                 NLL with rescale(a): {:.2f}\n
                 NLL with rescale(a,b,g): {:.2f}\n
                 NLL best: {:.2f}\n
                 NLL worst: {:.2f}\n
                 no rescale coeff: {:.2f}\n
                 rescale coeff (a,): {:.2f}\n
                 rescale coeff (a,b,g): {:.2f}\n
                 (a): a, : {:.2f} \n
                 (b): a, : {:.2f} \n
                 (a): a,b,g: {:.2f}, {:.2f}, {:.2f}\n
                 (b): a,b,g: {:.2f}, {:.2f}, {:.2f}\n""".format(RMSE_test,\
                                                       neglog_likelihood_no_rescale_test, \
                                                        neglog_likelihood_a_rescale_test, \
                                                      neglog_likelihood_a_b_g_rescale_test,\
                                                      NLL_best,NLL_worst,\
                                                      coeff_no_rescale, coeff_a_rescaled, coeff_a_b_g_rescaled,\
                                                        float(alpha_a_a),float(alpha_a_b),\
                                                        float(alpha_a_b_g_a), float(beta_a_b_g_a), float(gamma_a_b_g_a),\
                                                        float(alpha_a_b_g_b), float(beta_a_b_g_b), float(gamma_a_b_g_b) ))
        
        
        
        self.fitted = True
        self.alpha_a = alpha_a_b_g_a
        self.beta_a = beta_a_b_g_a
        self.gamma_a = gamma_a_b_g_a
        self.alpha_b = alpha_a_b_g_b
        self.beta_b = beta_a_b_g_b
        self.gamma_b = gamma_a_b_g_b 
        

class DualEnsemble_bak:
    #make it simple for now
    
    
    def __init__(self,ensA,ensB):
        
        check_is_fitted(ensA)
        check_is_fitted(ensB)
        
        assert isinstance(ensA.__class__,UncertaintyEnsembleRegressor.__class__)
        assert isinstance(ensB.__class__,UncertaintyEnsembleRegressor.__class__)
        
        ensA.rescaled = True
        ensB.rescaled = True
        
        self.ensA = ensA
        self.ensB = ensB
        
        self.alpha_a = None
        self.alpha_b = None
        self.beta_a = None
        self.beta_b = None
        self.gamma_a = None
        self.gamma_b = None
        
        self.fitted = False
    
    def _check_is_fitted(self):
        if self.fitted is False:
            raise ValueError
            
    def predict(self,Xpred_a,Xpred_b):
        self._check_is_fitted()
        
        Ymean_A, Ystd_A = self.ensA.predict_uncertainty(Xpred_a)
        Ymean_B, Ystd_B = self.ensB.predict_uncertainty(Xpred_b)
        
        Ystd_A = rescale_a_b_g(Ystd_A,self.alpha_a,self.beta_a,self.gamma_a)
        Ystd_B = rescale_a_b_g(Ystd_B,self.alpha_b,self.beta_b,self.gamma_b)
        
        Ymean_comb, Ystd_comb = combined_mean_std(Ymean_A, Ymean_B, Ystd_A, Ystd_B)
        
        return Ymean_comb, Ystd_comb
    
    def rescale_external(self,Xval_a,Xval_b,Yval,Xtest_a,Xtest_b,Ytest):
        
        Ypred_val_a = self.ensA._predict_uncertainty(Xval_a)
        Ypred_val_b = self.ensB._predict_uncertainty(Xval_b)
        
        Ypred_test_a = self.ensA._predict_uncertainty(Xtest_a)
        Ypred_test_b = self.ensB._predict_uncertainty(Xtest_b)
            
        # get means and vars of val set
        Ypred_val_mean_a = np.mean(Ypred_val_a,axis=1)
        Ypred_val_std_a = np.std(Ypred_val_a,axis=1,ddof=1)
        Ypred_val_mean_b = np.mean(Ypred_val_b,axis=1)
        Ypred_val_std_b = np.std(Ypred_val_b,axis=1,ddof=1)

        # get means and vars of test set
        Ypred_test_mean_a = Ypred_test_a.mean(axis=1)
        Ypred_test_std_a = Ypred_test_a.std(axis=1,ddof=1)
        Ypred_test_mean_b = Ypred_test_b.mean(axis=1)
        Ypred_test_std_b = Ypred_test_b.std(axis=1,ddof=1)
        
        # predict results using no scalind
        Ypred_test_mean_no_rescale, Ypred_test_std_no_rescale = combined_mean_std(Ypred_test_mean_a,Ypred_test_mean_b,Ypred_test_std_a,Ypred_test_std_b)
        
        # neglog_likelihood_rescale_dual(x,ystd_a,ystd_b,ymeans_a,ymeans_b,ytrue,rescale,nll=neglog_likelihood):
        min_res_full_a_b_g = minimize(neglog_likelihood_rescale_dual,args=(Ypred_val_std_a,Ypred_val_std_b,\
                                                                           Ypred_val_mean_a,Ypred_val_mean_b,Yval,rescale_a_b_g),x0=np.array([1.,1.,1.,1.,1.,1.]))
        
        #get params from minimization result 
        alpha_a_b_g_a, beta_a_b_g_a, gamma_a_b_g_a, alpha_a_b_g_b, beta_a_b_g_b, gamma_a_b_g_b = min_res_full_a_b_g["x"]
        print(min_res_full_a_b_g["fun"])
        
        
        #linear rescale
        min_res_full_a = minimize(neglog_likelihood_rescale_dual,args=(Ypred_val_std_a,Ypred_val_std_b,\
                                                                           Ypred_val_mean_a,Ypred_val_mean_b,Yval,rescale_a),x0=np.array([1.,1.]))
        
        alpha_a_a, alpha_a_b = min_res_full_a["x"]
        
        alpha_a_a = small_correction(alpha_a_a,self.ensA.n_estimators)
        alpha_a_b = small_correction(alpha_a_b,self.ensB.n_estimators)
        
        #calculate the negative log likelihoods using various rescalings
        
        neglog_likelihood_no_rescale_test = neglog_likelihood(Ypred_test_std_no_rescale,Ypred_test_mean_no_rescale,Ytest)
        neglog_likelihood_a_rescale_test = neglog_likelihood_rescale_dual(np.array([alpha_a_a,\
                                                                               alpha_a_b]),\
                                                                                Ypred_test_std_a,Ypred_test_std_b,\
                                                                                Ypred_test_mean_a,Ypred_test_mean_b,Ytest,rescale_a)
        neglog_likelihood_a_b_g_rescale_test = neglog_likelihood_rescale_dual(np.array([alpha_a_b_g_a, beta_a_b_g_a, gamma_a_b_g_a,\
                                                                               alpha_a_b_g_b, beta_a_b_g_b, gamma_a_b_g_b]),\
                                                                                Ypred_test_std_a,Ypred_test_std_b,\
                                                                                Ypred_test_mean_a,Ypred_test_mean_b,Ytest,rescale_a_b_g)
        
        
        # rescaled stds
        Ypred_test_std_a_rescaled = rescale_a_b_g(Ypred_test_std_a,alpha_a_b_g_a, beta_a_b_g_a, gamma_a_b_g_a)
        Ypred_test_std_b_rescaled = rescale_a_b_g(Ypred_test_std_b,alpha_a_b_g_b, beta_a_b_g_b, gamma_a_b_g_b)
        
        Ypred_test_std_a_rescaled_a_only = rescale_a(Ypred_test_std_a,alpha_a_a)
        Ypred_test_std_b_rescaled_a_only = rescale_a(Ypred_test_std_b,alpha_a_b)
        
        #combined rescaled prediction
        Ypred_test_mean_rescaled, Ypred_test_std_rescaled = combined_mean_std(Ypred_test_mean_a,Ypred_test_mean_b,Ypred_test_std_a_rescaled,Ypred_test_std_b_rescaled)
        Ypred_test_mean_rescaled_a_only, Ypred_test_std_rescaled_a_only = combined_mean_std(Ypred_test_mean_a,Ypred_test_mean_b, \
                                                                                            Ypred_test_std_a_rescaled_a_only,Ypred_test_std_b_rescaled_a_only)
        
        #calculate RMSEs
        RMSE_test = mean_squared_error(Ypred_test_mean_rescaled,Ytest,squared=False)
        
        #calculate test residuals
        errors_test = np.abs(Ypred_test_mean_rescaled-Ytest)
        
        #determine best and worst NLLs
        NLL_worst = neglog_likelihood(RMSE_test*np.ones(Ypred_test_mean_rescaled.shape),Ypred_test_mean_rescaled,Ytest)
        NLL_best = neglog_likelihood(errors_test,Ypred_test_mean_rescaled,Ytest)
        
        coeff_no_rescale = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_no_rescale_test)
        coeff_a_rescaled = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_a_rescale_test)
        coeff_a_b_g_rescaled = dimensionless_coeff(NLL_worst,NLL_best,neglog_likelihood_a_b_g_rescale_test)
        
        
        
        #print a rescaling report:
        print("""RMSE test: {:.2f}\n
                 NLL without rescale: {:.2f}\n
                 NLL with rescale(a): {:.2f}\n
                 NLL with rescale(a,b,g): {:.2f}\n
                 NLL best: {:.2f}\n
                 NLL worst: {:.2f}\n
                 no rescale coeff: {:.2f}\n
                 rescale coeff (a,): {:.2f}\n
                 rescale coeff (a,b,g): {:.2f}\n
                 (a): a, : {:.2f} \n
                 (b): a, : {:.2f} \n
                 (a): a,b,g: {:.2f}, {:.2f}, {:.2f}\n
                 (b): a,b,g: {:.2f}, {:.2f}, {:.2f}\n""".format(RMSE_test,\
                                                       neglog_likelihood_no_rescale_test, \
                                                        neglog_likelihood_a_rescale_test, \
                                                      neglog_likelihood_a_b_g_rescale_test,\
                                                      NLL_best,NLL_worst,\
                                                      coeff_no_rescale, coeff_a_rescaled, coeff_a_b_g_rescaled,\
                                                        float(alpha_a_a),float(alpha_a_b),\
                                                        float(alpha_a_b_g_a), float(beta_a_b_g_a), float(gamma_a_b_g_a),\
                                                        float(alpha_a_b_g_b), float(beta_a_b_g_b), float(gamma_a_b_g_b) ))
        
        
        
        self.fitted = True
        self.alpha_a = alpha_a_b_g_a
        self.beta_a = beta_a_b_g_a
        self.gamma_a = gamma_a_b_g_a
        self.alpha_b = alpha_a_b_g_b
        self.beta_b = beta_a_b_g_b
        self.gamma_b = gamma_a_b_g_b 
        