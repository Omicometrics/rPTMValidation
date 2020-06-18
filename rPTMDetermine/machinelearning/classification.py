"""
This utility is used for constructing data set for machine
learning.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Generator, List, Optional, Protocol, Sequence, Tuple, Union
import warnings

import numpy as np
from sklearn import base
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler

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


@dataclasses.dataclass
class Model:
    estimator: SKEstimator
    # TODO: type hint for this
    scaler: Optional


class ModelMetrics:
    """
    Model performance evaluation metrics.

    """
    def __init__(self, data: Union[np.ndarray, Sequence[float]]):
        """
        Initializes the class by computing the mean and standard deviation of
        the input `data.

        Args:
            data: Input numpy array.

        """
        self.mean = np.mean(data)
        self.std = np.std(data)


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
            num_features: Optional[int] = None,
            stacking_kfold: int = 3):
        """
        Under-sampling with cross validation.

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
        
        self.stacking_kfold = stacking_kfold

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
            x: Feature matrix, size of n samples by p variables.
            y: Sample labels, should be 0 and 1, and will be corrected to these 
               if not.

        """
        self.groups = y
        self._label_stats(y)
        self._check_x(x)

        # remove features
        if self._del_feature.any():
            x = x[:, ~self._del_feature]

        self._subsample_predict(x, y)
        self._stacking(y, kfold=self.stacking_kfold)
        # evaluate the performance of constructed model.
        self._evaluate(y)

    def predicta(self, x: np.ndarray, use_cv: bool = False) -> np.ndarray:
        """
        Predict unknown feature matrix.

        Args:
            x: Feature matrix, size n samples by p features.
            use_cv: Cross validated scores if setting to True, otherwise
                    native ensemble machine learning scores are returned.

        Returns:
            Numpy array of scores.

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
        models = []

        # iterate through subsamples
        for i, train_index in enumerate(self._subsample_split(y)):
            # reconstruct a new estimator
            estimator = base.clone(self.estimator)

            x_train, y_train = x[train_index], y[train_index]
            # classification with k-fold cross validation
            self._cvclassify.fit(x_train, y_train)

            # fitting model using sub-sampled samples
            scaler = self.scaler().fit(x_train)
            x_train = scaler.transform(x_train)
            clf = estimator.fit(x_train, y_train)

            # get predictions
            scores[train_index, i] = self._cvclassify.dc_scores

            # out of sample prediction
            if y_train.size != self._num:
                # scale out of samples
                x_out_of_sample = scaler.transform(x[~train_index])

                # out of sample scores
                pred_scores = clf.decision_function(x_out_of_sample)
                scores[~train_index, i] = pred_scores

            # save training information
            models.append(Model(clf, scaler))

        self.models = models
        self.scores = scores

        return self

    def _subsample_split(
            self,
            y: np.ndarray
    ) -> Generator[np.ndarray, None, None]:
        """
        Split the data matrix into train data and out-of-samples.

        Args:
            y: Label matrix.

        Yields:

        """
        major_idx, = np.where(y == self._label_major)
        # Minority class sample indices
        minor_idx, = np.where(y == self._label_minor)

        num_sub_samples = int(self._num_minor * self.sub_sample_fraction)
        # repeated random undersampling from major class
        sub_majors = subsample(
            self._num_major, num_sub_samples, self.num_sub_samples
        )

        # repeated random subsampling from minor class if the
        # fraction is lower than 1.
        sub_minors = [None] * self.num_sub_samples
        if self.sub_sample_fraction < 1:
            sub_minors = subsample(
                self._num_minor, num_sub_samples, self.num_sub_samples
            )

        # do data matrix splitting
        for sub_major, sub_minor in zip(sub_majors, sub_minors):
            train_index = np.zeros_like(y, dtype=bool)
            train_index[major_idx[sub_major]] = True
            train_index[
                minor_idx if sub_minor is None else minor_idx[sub_minor]
            ] = True
            yield train_index

    def _stacking(
            self,
            y: np.ndarray,
            kfold: int = 3
    ) -> EnsembleClassifier:
        """
        Linear combination of scores using stacking.

        Args:
            y: Label matrix.
            kfold: Number of cross validation folds for stacking.

        """
        # Stacking posterior probability
        self.stacking = Stacking(kfold=kfold)
        self.stacked_scores, self.stacked_probs = \
            self.stacking.fit(self.scores, y)

        return self

    def _evaluate(self, y):
        """
        Evaluates the performance of the model by calculating
        sensitivity, specificity, area under ROC curve (AUC),
        and balanced accuracy (BA).

        """
        # bins for calculating matrics
        bins, stp = np.linspace(0, 1, num=1001, retstep=True)
        bins = np.insert(bins, 0, 0 - stp)
        bins = np.append(bins, 1 + stp)

        test_index = self.stacking.test_index

        sensitivities = []
        specificities = []
        balanced_accuracies = []
        aucs = []
        for i in range(self.stacking_kfold):
            y_test = y[test_index == i]
            probs = self.stacked_probs[test_index == i]
            # counts of each class under current fold as testing data
            cs, nc = np.unique(y_test, return_counts=True)
            sensitivity = np.count_nonzero(probs[y_test == 1] >= 0.5) / nc[1]
            sensitivities.append(sensitivity)
            specificity = np.count_nonzero(probs[y_test == 0] < 0.5) / nc[0]
            specificities.append(specificity)
            balanced_accuracies.append((sensitivity + specificity) / 2)
            # AUC
            h0, _ = np.histogram(probs[y_test == 0], bins)
            h1, _ = np.histogram(probs[y_test == 1], bins)
            # calculate TPR and FPR at each threshold defined by bins.
            tpr = (1 - np.cumsum(h1) / nc[1])[::-1]
            fpr = (1 - np.cumsum(h0) / nc[0])[::-1]
            # AUC by trapezoidal rule
            auc = ((tpr[1:] + tpr[:-1]) * np.diff(fpr)).sum() / 2
            aucs.append(auc)

        self.sensitivity = ModelMetrics(sensitivities)
        self.specificity = ModelMetrics(specificities)
        self.balanced_accuracy = ModelMetrics(balanced_accuracies)
        self.auc = ModelMetrics(aucs)

    def _label_stats(
            self,
            y: np.ndarray
    ) -> EnsembleClassifier:
        """
        Sets up stats of labels.

        Args:
            y: Label matrix.

        """
        cs, nc = np.unique(y, return_counts=True)

        if cs.size != 2:
            raise ValueError("Two classes are required.")

        # statistics
        sortix = np.argsort(nc)
        self._num_major = nc[sortix[1]]
        self._label_major = cs[sortix[1]]
        self._num_minor = nc[sortix[0]]
        self._label_minor = cs[sortix[0]]
        self._num = nc.sum()

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
    Classifier for validation of PTM identifications.

    """
    def __init__(
            self,
            estimator: SKEstimator,
            kfold: int = 10,
            num_sub_samples: int = 30,
            sub_sample_fraction: float = 1.,
            num_features: Optional[int] = None,
            stacking_kfold: int = 3
    ):
        """
        Initializes the Classifier.

        Args:
            estimator: Scikit-learn estimator to be used for classification.
            kfold: Number of cross validation folds.
            num_sub_samples: Number of subsampling iterations, for class
                             imbalance.
            sub_sample_fraction: Fraction of samples subsampled for model
                                 construction.
            num_features: Number of features used for model construction.
            stacking_kfold: Number of cross validation folds for stacking.

        """
        self.estimator = estimator

        # fraction of samples sampled for validation
        self.sub_sample_fraction = (
            1. if sub_sample_fraction is None or sub_sample_fraction >= 1.
            else sub_sample_fraction
        )
        self.kfold = kfold
        self.num_sub_samples = num_sub_samples
        self.num_features = num_features
        self.stacking_kfold = stacking_kfold

        self._ensemble = self._make_ensemble()

    def fit(self, x: np.ndarray, y: np.ndarray) -> Classifier:
        """
        Validate PTM identifications.

        Args:
            x: Feature matrix with size n samples by p features.
            y: Labels of samples. Negatives are 0 and positives are 1.

        Returns:
            Classifier.

        """
        # reset labels to 0 and 1
        y = self._label_reset(y)
        self._ensemble.predict(x, y)
        return self

    def predict(
            self,
            x: Union[Sequence[Sequence[float]], np.ndarray],
            use_cv: bool = False
    ) -> np.ndarray:
        """
        Calculates validation scores.

        Args:
            x: Feature matrix for score calculation.
            use_cv: Type of scores returned. Setting to `True` will return
            cross validated scores with size of `n` samples by `k` fold
            cross validations, otherwise singe score for each PSM
            is returned. Default is `False`.

        Returns:
            Scores for the input feature matrix.

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        return self._ensemble.predicta(x, use_cv=use_cv)

    @property
    def train_scores(self):
        """ Train scores. """
        return self._ensemble.stacked_scores

    @property
    def train_groups(self):
        """Groups of training samples. """
        return self._ensemble.groups

    @property
    def stacking_test_index(self):
        """ Test index in stacking cross validation. """
        return self._ensemble.stacking.test_index

    @property
    def model_scores(self):
        """ Metrics for evaluating validation model. """
        # TO DO: Calculate sensitivity, specificity, accuracy
        # AUC to evaluate the performance of the classification
        # model.
        return {
            "sensitivity": self._ensemble.sensitivity,
            "specificity": self._ensemble.specificity,
            "balanced accuracy": self._ensemble.balanced_accuracy,
            "AUC": self._ensemble.auc
        }

    def _label_reset(self, y):
        """
        Reset the label to be 0 and 1 and set back once
        the learning finished.

        """
        cs = np.unique(y)
        if cs.size != 2:
            raise ValueError("Two classes are required.")

        _tempy = y.copy()
        # correct the label to be 0 and 1
        cssort = sorted(cs)
        if cssort[0] != 0:
            _tempy[y == cssort[0]] = 0
            self._original_labels = dict(zip([0, 1], cssort))
        if cssort[1] != 1:
            _tempy[y == cssort[1]] = 1
        self._original_labels = dict(zip([0, 1], cssort))

        return _tempy

    def _make_ensemble(self) -> EnsembleClassifier:
        """
        Constructs an EnsembleClassifier using the parameters defined on the
        class.

        """
        return EnsembleClassifier(
            self.estimator,
            self.kfold,
            num_sub_samples=self.num_sub_samples,
            sub_sample_fraction=self.sub_sample_fraction,
            num_features=self.num_features,
            stacking_kfold=self.stacking_kfold
        )
