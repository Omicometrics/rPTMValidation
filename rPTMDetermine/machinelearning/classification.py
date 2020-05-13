"""
This utility is used for constructing data set for machine
learning.
"""
from __future__ import annotations

import collections
import random
from typing import List, Optional, Protocol, Tuple
import warnings

import numpy as np
from sklearn import base
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from .stacking import Stacking


class SKEstimator(Protocol):
    def decision_function(self, x) -> np.ndarray:
        ...

    def fit(self, x, y):
        ...


# TODO: remove warning handling from the machinelearning subpackage
#       If it needs to be included, then it should be done in validate.py, i.e.
#       the executable module used to run rPTMDetermine.
def _warn(*args, **kwargs):
    pass


warnings.warn = _warn
np.seterr(divide="ignore", invalid="ignore", over="ignore")

Model = collections.namedtuple("Model", ["estimator", "scaler"])


SUBSAMPLE_FRACTIONS = [
    (1, 0),
    (0.8, 1000),
    (0.6, 3000),
    (0.3, 5000),
    (0.2, 10000),
    (0.1, 3e4),
    (0.05, 3e5),
    (0.01, 1e6)
]


def construct_model(x_pos: np.ndarray, x_neg: np.ndarray) -> Classifier:
    """
    Constructs validation model.

    """
    print(
        f'No. of positives: {x_pos.shape[0]}\n'
        f'No. of decoys: {x_neg.shape[0]}'
    )

    x = np.concatenate((x_pos, x_neg), axis=0)
    y = np.concatenate(
        (np.ones(x_pos.shape[0]), np.zeros(x_neg.shape[0])),
        axis=0
    )

    # Number of samples in each positive and negatives
    _, ns = np.unique(y, return_counts=True)
    # Number of samples in minor group
    minor_group_size = ns.min()

    sub_sample_fraction = min(
        r for r, t in SUBSAMPLE_FRACTIONS if minor_group_size > t
    )

    model = Classifier(
        GridSearchCV(
            LinearSVC(),
            {'C': [2 ** i for i in range(-12, 4)]},
            cv=5
        ),
        kfold=10,
        num_sub_samples=30,
        sub_sample_fraction=sub_sample_fraction
    )
    model.fit(x, y)

    return model


def subsample(n: int, m: int, repeats: int) -> Tuple[np.ndarray, ...]:
    """
    Random sub-sampling of `m` samples from total `n` samples `repeats` times.

    """
    # to make the results reproducible, use fixed seeds
    # to randomly shuffle the sequence
    seeds = list(range(repeats))
    idx = np.arange(n, dtype=int)
    rand_sample_idx = []
    for i in range(repeats):
        random.Random(seeds[i] * 10).shuffle(idx)
        rand_sample_idx.append(np.sort(idx[:m]))

    return tuple(rand_sample_idx)


def select_by_fisher_scores(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """
    Select features with top `k` Fisher scores.

    Ref:
    Gu Q, et al. Generalized Fisher Score for Feature Selection.
    2012. arXiv:1202.3725.

    Args:
        x:
        y:
        k: Number of features to keep.

    Returns:
        Array with top `k` features.

    """
    n, p = x.shape
    if k >= p or k <= 0:
        return np.arange(p, dtype=int)

    cs, ncs = np.unique(y, return_counts=True)
    # overall mean and std
    gmean = np.mean(x, axis=0)
    # fisher score through each class
    cmean2, cvar2 = np.zeros(p), np.zeros(p)
    for ck, nk in zip(cs, ncs):
        cmean2 += nk * (np.mean(x[y == ck], axis=0) ** 2)
        cvar2 += nk * (np.std(x[y == ck], axis=0) ** 2)

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

        self._labels: Optional[np.ndarray] = None
        self.dc_scores: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> CrossValidationClassifier:
        """
        Classify `x` according to the input label in `y`.
        """
        r, c = x.shape
        # NOTE: This ratio is arbitrary without any judgement.
        if r / c < 2:
            self._select_feature = True

        self._labels = np.sort(np.unique(y))
        self._cross_validation(x, y)

        return self

    def _cross_validation(self, x: np.ndarray, y: np.ndarray):
        """
        Perform cross validation
        """
        self.dc_scores = np.zeros(y.shape)

        skf = StratifiedKFold(n_splits=self.kfold)
        for train_idx, test_idx in skf.split(x, y):
            # nested cross validation
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_test = x[test_idx]

            # feature selection
            if self._select_feature:
                selected = select_by_fisher_scores(
                    x_train, y_train, self.num_features
                )
                x_train = x_train[:, selected]
                x_test = x_test[:, selected]

            scaler = self.scaler()

            # test set
            x_train = scaler.fit_transform(x_train)
            self.estimator.fit(x_train, y_train)

            # predicted scores
            x_test = scaler.transform(x_test)

            _predict_scores = self.estimator.decision_function(x_test)
            self.dc_scores[test_idx] = _predict_scores

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
            sub_sample_fraction: Optional[float] = None,
            num_features: Optional[int] = None):
        """
        under-sampling with cross validation
        """
        self.estimator = estimator

        self.num_sub_samples = num_sub_samples

        # fraction of samples sampled for validation
        self.sub_sample_fraction = (
            1. if sub_sample_fraction is None or sub_sample_fraction >= 1.
            else sub_sample_fraction
        )

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

        self.models: List[Model] = []

    def predict(self, x: np.ndarray, y: np.ndarray):
        """
        Predict the labels along with probabilities using cross validation.

        Args:
            x: Feature matrix, size of n by p.
            y: Sample labels, should be -1 and 1, if not,
               corrected to -1 and 1.

        """
        self._label_stats(y)
        self._check_x(x)

        # remove features
        if self._del_feature.any():
            x = x[:, ~self._del_feature]

        self._subsample_predict(x, y)
        self._stacking()

    def predicta(self, x: np.ndarray, use_cv: bool = False) -> np.ndarray:
        """
        Predict additional dataset

        """
        n = x.shape[0]
        if self._del_feature.any():
            x = x[:, ~self._del_feature]

        # scores
        scores = np.empty((n, self.num_sub_samples))
        for j, _model in enumerate(self.models):
            x_scaled = _model.scaler.transform(x)
            scores[:, j] = _model.estimator.decision_function(x_scaled)

        # stacking and estimate posterior probability
        if use_cv:
            score = self.stacking.predict_cv(scores)
        else:
            score, _ = self.stacking.predict(scores)

        return score

    def _subsample_predict(self, x: np.ndarray, y: np.ndarray) \
            -> EnsembleClassifier:
        """
        Perform prediction using sub-sampling for imbalanced problem.

        """
        scores = np.empty((self._num, self.num_sub_samples))
        models, indices = [], []

        # class sample indices
        major_idx, = np.where(y == self._label_major)
        minor_idx, = np.where(y == self._label_minor)

        # repeated random under-sampling from major class
        num_sub_samples = int(self._num_minor * self.sub_sample_fraction)
        sub_major = subsample(
            self._num_major, num_sub_samples, self.num_sub_samples
        )

        sub_minor = None
        if self.sub_sample_fraction < 1.:
            sub_minor = subsample(
                self._num_minor, num_sub_samples, self.num_sub_samples
            )

        # iterate through sub-samples
        for i, sub in enumerate(sub_major):
            # reconstruct a new estimator
            estimator = base.clone(self.estimator)
            # under-sampled trains
            sel_train = np.zeros(self._num, dtype=bool)
            sel_train[major_idx[sub]] = True
            ix = minor_idx if sub_minor is None else minor_idx[sub_minor[i]]
            sel_train[ix] = True

            x_train = x[sel_train]
            # classification with k-fold cross validation
            self._cvclassify.fit(x_train, y[sel_train])

            # fitting model using sub-sampled samples
            scaler = self.scaler().fit(x_train)
            x_train = scaler.transform(x_train)
            clf = estimator.fit(x_train, y[sel_train])

            # get predictions
            scores[sel_train, i] = self._cvclassify.dc_scores

            # out of sample prediction
            if not sel_train.all():
                # scale out of samples
                x_out_of_sample = scaler.transform(x[~sel_train])
                scores[~sel_train, i] = clf.decision_function(x_out_of_sample)

            # save training information
            models.append(Model(clf, scaler))
            indices.append(np.where(sel_train)[0])

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

    def _check_x(self, x: np.ndarray):
        """
        Removes features with majority a single value.

        During cross validation, normalization of this kind
        of features may probably result in NaN values in the
        matrix, raising ValueError.

        """
        r, c = x.shape
        _del_feature = np.zeros(c, dtype=bool)
        for i in range(c):
            _, nuc = np.unique(x[:, i], return_counts=True)
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
            sub_sample_fraction: float = 1.,
            num_features: Optional[int] = None
    ):
        """
        Cross-validated classification using `estimator`.

        Args:
            estimator: Machine learning classifier.
            kfold: Number of folds in cross validation.
            num_sub_samples: Number of sub samples.
            sub_sample_fraction:
            num_features:

        """
        self.estimator = estimator

        # fraction of samples sampled for validation
        if sub_sample_fraction is not None and sub_sample_fraction < 0:
            raise ValueError(
                'Fraction for sampling must be positive.'
            )
        if sub_sample_fraction is None or sub_sample_fraction > 1:
            sub_sample_fraction = 1

        # model parameters
        self.kfold = kfold
        self.num_sub_samples = num_sub_samples
        self.sub_sample_fraction = sub_sample_fraction
        self.num_features = num_features

        self.ensemble: Optional[EnsembleClassifier] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> Classifier:
        """
        Validate the PTM identifications with features in `X` and
        labels in `y`.

        """
        num_features = (
            x.shape[1] if self.num_features is None else self.num_features
        )
        if self.num_features is None:
            num_features = x.shape[1]
        ems = EnsembleClassifier(
            self.estimator,
            kfold=self.kfold,
            num_sub_samples=self.num_sub_samples,
            sub_sample_fraction=self.sub_sample_fraction,
            num_features=num_features
        )
        # reset labels to 0 and 1
        y = self._reset_labels(y)
        ems.predict(x, y)

        self.ensemble = ems

        return self

    def predict(self, x: np.ndarray, use_cv: bool = False) -> np.ndarray:
        """
        Prediction of additional data `x`.

        """
        if self.ensemble is None:
            raise RuntimeError(
                'fit must be called on Classifier before predict'
            )

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        return self.ensemble.predicta(x, use_cv=use_cv)

    def _reset_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Resets the label to be 0 and 1 and stores the original labels for
        restoring once the learning is finished.

        Args:
            y: Array of class labels.

        """
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError('Two classes are required.')

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
