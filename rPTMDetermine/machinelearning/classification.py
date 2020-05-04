"""
This utility is used for constructing data set for machine
learning.
"""
from __future__ import annotations

import collections
import random
from typing import Optional, Protocol
import warnings

import numpy as np
from sklearn import base
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler

from .stacking import Stacking


def _warn(*args, **kwargs):
    pass


class SKEstimator(Protocol):
    def decision_function(self, x) -> np.ndarray:
        ...

    def fit(self, x, y):
        ...


warnings.warn = _warn
np.seterr(divide="ignore", invalid="ignore", over="ignore")

Model = collections.namedtuple("Model", ["estimator", "scaler"])


def subsample(n: int, m: int, nrep: int):
    """
    Random subsampling m samples from total n samples
    nrep times
    """
    # to make the results reproducible, use fixed seeds
    # to randomly shuffle the sequence
    seeds = list(range(nrep))
    idx = np.arange(n, dtype=int)
    randsampleidx = []
    for i in range(nrep):
        random.Random(seeds[i] * 10).shuffle(idx)
        randsampleidx.append(np.sort(idx[:m]))

    return tuple(randsampleidx)


def fisher_scores(X: np.ndarray, y: np.ndarray, k) -> np.ndarray:
    """
    Fisher score.
    Select features with top k Fisher scores.
    Ref:
    Gu Q, et al. Generalized Fisher Score for Feature Selection.
    2012. arXiv:1202.3725.
    """
    n, p = X.shape
    if k >= p or k <= 0:
        return np.arange(p, dtype=int)

    cs, ncs = np.unique(y, return_counts=True)
    # overall mean and std
    gmean = np.mean(X, axis=0)
    # fisher score through each class
    cmean2, cvar2 = np.zeros(p), np.zeros(p)
    for ck, nk in zip(cs, ncs):
        cmean2 += nk * (np.mean(X[y == ck], axis=0) ** 2)
        cvar2 += nk * (np.std(X[y == ck], axis=0) ** 2)

    # fisher scores
    fscore = (cmean2 - n * (gmean ** 2)) / cvar2

    # top n features
    ix = np.argsort(fscore)[::-1]

    return ix[:k]


class CrossValidationClassifier:
    """
    """
    def __init__(
            self,
            estimator: SKEstimator,
            kfold: int,
            num_features: int,
            scaler):
        """
        Initialization
        """
        self.estimator = estimator
        self.kfold = kfold
        self.num_features = num_features
        self._select_feature = False
        self.scaler = scaler

        self._labels: np.ndarry = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> CrossValidationClassifier:
        """
        Classify `X` according to the input label in `y`.
        """
        r, c = X.shape
        # NOTE: This ratio is arbitrary without any judgement.
        if r / c < 2:
            self._select_feature = True

        self._labels = np.sort(np.unique(y))
        self._cross_validation(X, y)

        return self

    def _cross_validation(self, X: np.ndarray, y: np.ndarray):
        """
        Perform cross validation
        """
        self._dcscores = np.zeros(y.shape)

        # cross validation
        skf = StratifiedKFold(n_splits=self.kfold)
        for tridx, vlidx in skf.split(X, y):
            # nested cross validation
            Xc = X[tridx]
            yc = y[tridx]
            Xv = X[vlidx]

            # feature selection
            if self._select_feature:
                selected = fisher_scores(Xc, yc, self.num_features)
                Xc = Xc[:, selected]
                Xv = Xv[:, selected]

            scaler = self.scaler()

            # test set
            Xcn = scaler.fit_transform(Xc)
            self.estimator.fit(Xcn, yc)

            # predicted scores
            Xv = scaler.transform(Xv)

            _predict_scores = self.estimator.decision_function(Xv)
            self._dcscores[vlidx] = _predict_scores

        return self


class EnsembleClassifier:
    """
    Classification using random under-sampling for noisy label
    correction.
    """
    def __init__(
            self,
            estimator: SKEstimator,
            kfold: int,
            # This should be typed as something like
            # Intersection[base.TransformerMixin, base.BaseEstimator]; however,
            # this is not yet supported.
            # See: https://github.com/python/typing/issues/213
            scaler=MaxAbsScaler,
            num_sub_samples: int = 1,
            sub_sample_frac: Optional[float] = None,
            num_features: Optional[float] = None):
        """
        under-sampling with cross validation
        """
        self.estimator = estimator

        self.num_sub_samples = num_sub_samples

        # fraction of samples sampled for validation
        self.sub_sample_frac = sub_sample_frac
        if sub_sample_frac is None or sub_sample_frac >= 1:
            self.sub_sample_frac = 1

        # number of features retained if not enough sample
        if num_features is None:
            num_features = 10

        self.scaler = scaler

        # classifier
        cv_estimator = base.clone(self.estimator)
        self._cvclassify = CrossValidationClassifier(
            cv_estimator,
            kfold,
            num_features,
            self.scaler
        )

    def predict(self, X: np.ndarray, y: np.ndarray):
        """
        Predict the labels along with probabilities using cross validation.

        Args:
            X: Feature matrix, size of n by p
            y: Sample labels, should be -1 and 1, if not,
               corrected to -1 and 1
        """
        self._label_stats(y)
        self._check_X(X)

        # remove features
        if self._del_feature.any():
            X = X[:, ~self._del_feature]

        self._subsample_predict(X, y)
        self._stacking()

    def predicta(self, X: np.ndarray, use_cv: bool = False):
        """
        Predict additional dataset

        """
        n = X.shape[0]
        if self._del_feature.any():
            X = X[:, ~self._del_feature]

        # scores
        scores = np.empty((n, self.num_sub_samples))
        for j, _model in enumerate(self.models):
            X_scaled = _model.scaler.transform(X)
            scores[:, j] = _model.estimator.decision_function(X_scaled)

        # stacking and estimate posterior probability
        if use_cv:
            score = self.stacking.predict_cv(scores)
        else:
            score, _ = self.stacking.predict(scores)

        return score

    def _subsample_predict(self, X: np.ndarray, y: np.ndarray) \
            -> EnsembleClassifier:
        """
        perform prediction using subsampling for imbalanced problem
        """
        # preallocation
        scores = np.empty((self._num, self.num_sub_samples))
        models, indices = [], []

        # class sample indices
        majix, = np.where(y == self._label_major)
        mnjix, = np.where(y == self._label_minor)

        # repeated random undersampling from major class
        nsubsamples = int(self._num_minor * self.sub_sample_frac)
        sub_major = subsample(
            self._num_major, nsubsamples, self.num_sub_samples
        )

        sub_minor = None
        if self.sub_sample_frac < 1:
            sub_minor = subsample(
                self._num_minor, nsubsamples, self.num_sub_samples
            )

        # iterate through subsamples
        for i, subsample in enumerate(sub_major):
            # reconstruct a new estimator
            estimator = base.clone(self.estimator)
            # under-sampled trains
            seltrains = np.zeros(self._num, dtype=bool)
            seltrains[majix[subsample]] = True
            ix = mnjix if sub_minor is None else mnjix[sub_minor[i]]
            seltrains[ix] = True

            X_train = X[seltrains]
            # classification with k-fold cross validation
            self._cvclassify.fit(X_train, y[seltrains])

            # fitting model using subsampled samples
            scaler = self.scaler().fit(X_train)
            Xcn = scaler.transform(X_train)
            clf = estimator.fit(Xcn, y[seltrains])

            # get predictions
            scores[seltrains, i] = self._cvclassify._dcscores

            # out of sample prediction
            if not seltrains.all():
                # scale out of samples
                Xoos = scaler.transform(X[~seltrains])

                # scores
                pred_scores = clf.decision_function(Xoos)
                scores[~seltrains, i] = pred_scores

            # save training information
            models.append(Model(clf, scaler))
            indices.append(np.where(seltrains)[0])

        self.models = models
        self.scores = scores
        self.train_indices = tuple(indices)

        return self

    def _stacking(self):
        """
        Linear combination of probabilities obtained from density estimation
        by stacking.
        """

        # stacking posterior probability
        self.stacking = Stacking()
        _stk_scores, _stk_probs, _qvals = self.stacking.fit(self.scores,
                                                            self._y)
        self.stacked_scores = _stk_scores
        self.stacked_probs = _stk_probs
        self.qvalues = _qvals

        return self

    def _label_stats(self, y):
        """
        set up stats of labels
        """
        cs, nc = np.unique(y, return_counts=True)
        _n = nc.sum()

        if cs.size != 2:
            raise ValueError("Two classes are required.")

        # statistics
        sortix = np.argsort(nc)
        self._labels = np.sort(cs)
        self._num_major = nc[sortix[1]]
        self._label_major = cs[sortix[1]]
        self._num_minor = nc[sortix[0]]
        self._label_minor = cs[sortix[0]]
        self._num = _n
        self._y = y
        # prior of each class
        self.priors = dict(zip(cs, nc / _n))

        return self

    def _check_X(self, X):
        """
        Remove features with majority a single value.
        During cross validation, normalization of this kind
        of features may probably result in NaN values in the
        matrix, raising ValueError.
        """
        r, c = X.shape
        _del_feature = np.zeros(c, dtype=bool)
        for i in range(c):
            _, nuc = np.unique(X[:, i], return_counts=True)
            if np.amax(nuc) / r >= 0.6:
                _del_feature[i] = True

        self._del_feature = _del_feature

        return self


class Classifier:
    """
    Classifier for validation of PTM identifications
    """

    def __init__(
            self,
            estimator: SKEstimator,
            kfold: int = 5,
            num_sub_samples: int = 1,
            sub_sample_frac: float = 1.,
            num_features: Optional[int] = None):
        """
        Cross-validated classification using `estimator`.

        Args:
            estimator: Machine learning classifier.
            kfold: Number of folds in cross validation.
            num_sub_samples: Number of sub samples.
            sub_sample_frac:
            num_features:

        """
        self.estimator = estimator

        # fraction of samples sampled for validation
        if sub_sample_frac is not None and sub_sample_frac < 0:
            raise ValueError("Fraction for sampling must be positive.")
        if sub_sample_frac is None or sub_sample_frac > 1:
            sub_sample_frac = 1

        # model parameters
        self.kfold = kfold
        self.num_sub_samples = num_sub_samples
        self.sub_sample_frac = sub_sample_frac
        self.num_features = num_features

        self.ensemble: Optional[EnsembleClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier:
        """
        Validate the PTM identifications with features in `X` and
        labels in `y`.

        """
        # ensemble model
        num_features = (
            X.shape[1] if self.num_features is None else self.num_features
        )
        if self.num_features is None:
            num_features = X.shape[1]
        ems = EnsembleClassifier(
            self.estimator,
            kfold=self.kfold,
            num_sub_samples=self.num_sub_samples,
            sub_sample_frac=self.sub_sample_frac,
            num_features=num_features
        )
        # reset labels to 0 and 1
        y = self._reset_labels(y)
        ems.predict(X, y)

        self.ensemble = ems

        return self

    def predict(self, X: np.ndarray, use_cv: bool = False):
        """
        Prediction of additional data X
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.ensemble.predicta(X, use_cv=use_cv)

    def _reset_labels(self, y: np.ndarray) -> np.ndarray:
        """
        reset the label to be 0 and 1
        and set back once the learning finished
        """
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("Two classes are required.")

        temp_y = y.copy()
        # correct the label to be 0 and 1
        sorted_classes = sorted(classes)
        if sorted_classes[0] != 0:
            temp_y[y == sorted_classes[0]] = 0
            self._original_labels = dict(zip([0, 1], sorted_classes))
        if sorted_classes[1] != 1:
            temp_y[y == sorted_classes[1]] = 1
        self._original_labels = dict(zip([0, 1], sorted_classes))

        return temp_y
