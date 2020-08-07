"""
Peptide validation using ensemble SVM.

"""
from __future__ import annotations

import bisect
import dataclasses
from typing import List, Optional, Sequence, Tuple

import cloudpickle
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler

from .randomforest import RandomForest
from rPTMDetermine.psm_container import PSMContainer


@dataclasses.dataclass
class Scaler:
    centers: np.ndarray
    normalizer: np.ndarray


def calculate_q_values(scores, groups):
    """
    Estimates q values with correction by percentage of incorrect
    identifications.

    Args:
        scores: Scores calculated using the constructed validation model.
        groups: Groups of identifications, 0 for decoys and 1 for targets.

    Returns:
        q values with score thresholds, with size 2 * n, where n
        is the number of bins for digitizing. The first row is
        thresholds, second is q values.

    References:
    [1] Storey JD, et al. Statistical significance for genomewide
        studies. PNAS. 2003, 100, 9440-9445.
    [2] Kall L, et al. Assigning significance to peptides identified
        by tandem mass spectrometry using decoy databases. J
        Proteome Res. 2008, 7, 29–34.

    """
    # score bins
    thresholds = np.linspace(scores.min(), scores.max(), 1000)

    # number of decoys and targets
    nt = groups.sum()
    nd = groups.size - nt

    # bin scores
    counts_overall, _ = np.histogram(scores, thresholds)
    counts_decoy, _ = np.histogram(scores[groups == 0], thresholds)

    # densities
    densities_decoy = np.cumsum(counts_decoy)
    densities_target = np.cumsum(counts_overall) - densities_decoy

    # fdr
    tix = densities_target < nt
    fdr = (nd - densities_decoy[tix]) / (nt - densities_target[tix])

    # PIT
    dix = densities_decoy > 0
    r = np.median(densities_target[dix] / densities_decoy[dix])

    # q values
    q, _q_min = [], 1
    for _fdr in fdr:
        if _fdr > _q_min:
            q.append(_q_min)
        else:
            q.append(_fdr)
            _q_min = _fdr

    return np.array([thresholds[1:][tix], np.array(q) * r])


def calculate_threshold_score(
        q_values: np.ndarray,
        q_threshold: float
) -> float:
    """
    Finds the score value at `q_threshold`.

    Args:
        q_values: Array of calculated q-values.
        q_threshold: Cutoff q-value.

    Returns:
        Score for the required cutoff.

    """
    q_values = q_values[:, q_values[1].argsort()]
    scores, q_values = q_values[0], q_values[1]
    # if all identifications having q values lower than q_threshold
    # return min scores
    if q_values[-1] <= q_threshold:
        return scores[-1]

    j = bisect.bisect_left(q_values, q_threshold)
    return scores[j + 1]


@dataclasses.dataclass
class _Metrics:
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    auc: float


@dataclasses.dataclass
class ModelMetrics:
    avg_sensitivity: np.ndarray
    std_sensitivity: np.ndarray
    avg_specificity: np.ndarray
    std_specificity: np.ndarray
    avg_balanced_accuracy: np.ndarray
    std_balanced_accuracy: np.ndarray
    avg_auc: np.ndarray
    std_auc: np.ndarray


def _combine_metrics(metrics: Sequence[_Metrics]) -> ModelMetrics:
    sensitivities, specificities, bal_accuracies, aucs = [], [], [], []
    for est_metrics in metrics:
        sensitivities.append(est_metrics.sensitivity)
        specificities.append(est_metrics.specificity)
        bal_accuracies.append(est_metrics.balanced_accuracy)
        aucs.append(est_metrics.auc)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    bal_accuracies = np.array(bal_accuracies)
    aucs = np.array(aucs)
    return ModelMetrics(
        np.mean(sensitivities), np.std(sensitivities),
        np.mean(specificities), np.std(specificities),
        np.mean(bal_accuracies), np.std(bal_accuracies),
        np.mean(aucs), np.std(aucs)
    )


class ValidationModel:
    """
    Validation of peptide identification.

    """
    def __init__(self,
                 model_features: Optional[Sequence[str]] = None,
                 cv: Optional[int] = None,
                 q_threshold: float = 0.01,
                 n_estimators: int = 100,
                 max_samples: Optional[float] = 10000,
                 n_jobs: int = -1,
                 max_depth: int = 2,
                 oob_score: bool = False):
        """
        Initializes the ValidationModel.

        Args:
            model_features: Features to be used in model construction and
                            prediction.
            cv: fold for cross validation.
            q_threshold: q-value threshold.
            n_estimators: number of estimators for random forest.
            max_samples: the number of samples to draw from X to train
                         each base estimator in random forest.
            n_jobs: number of jobs to run in parallel.
            max_depth: the maximum depth of the tree.
            oob_score: Whether to use out-of-sampling samples to estimate
                       the balanced accuracy.

        """
        self._base_estimator = make_pipeline(
            MinMaxScaler(),
            RandomForest(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                random_state=1,
                max_samples=max_samples,
                max_density_samples=10000,
                oob_score=oob_score
            )
        )
        self._estimators: List[Pipeline] = []
        self.oob_score = oob_score
        self.cv = cv
        self.q_threshold = q_threshold

        self.model_features = model_features

        self._scaler: Optional[Scaler] = None
        self._decoy_train_index: Optional[np.ndarray] = None
        self._oob_scores: Optional[np.ndarry] = []

        self.metrics: Optional[ModelMetrics] = None
        self.prob_metrics: Optional[ModelMetrics] = None
        self.cv_scores: Optional[np.ndarray] = None
        self.test_scores: Optional[List[np.ndarray]] = None

    def fit(
            self,
            pos_psms: PSMContainer,
            decoy_psms: PSMContainer,
            neg_psms: Optional[PSMContainer] = None
    ):
        """Constructs validation model.

        Args:
            pos_psms: Positive (pass FDR control) identifications.
            decoy_psms: Decoy identifications.
            neg_psms: Negative (fail FDR control) identifications. Only required
                      if cv is not None when constructing the ValidationModel.

        """
        if self.cv is not None and neg_psms is None:
            raise RuntimeError(
                'Parameter `neg_psms` must be passed to ValidationModel.fit '
                'when CV is used'
            )

        # training data matrix
        x, y, train_psms = self._train_matrix(pos_psms, decoy_psms)

        if self.cv is not None:
            skf = StratifiedKFold(n_splits=self.cv)

            x_neg = self._features_to_matrix(neg_psms)

            mx, nx = [], []
            val_metrics, prob_metrics = [], []
            cv_scores = np.empty(x.shape[0])
            test_scores_x = []

            for train_index, test_index in skf.split(x, y):
                model = clone(self._base_estimator)
                model.fit(x[train_index], y[train_index])
                self._estimators.append(model)

                # Compute and normalize scores
                y_test, x_test = y[test_index], x[test_index]
                ntest = test_index.size

                # combine testing feature matrix with negative feature matrix
                x_val = np.concatenate((x_test, x_neg), axis=0)

                if self.oob_score:
                    x_val = model["minmaxscaler"].transform(x_val)
                    scores, score_matrix =\
                        model["randomforest"].decision_function(
                            x_val, return_matrix=True
                        )
                    self._set_oob_score(score_matrix[:ntest], y_test)
                else:
                    scores = model.decision_function(x_val)

                test_scores = scores[:ntest]

                # dnp: return the scores
                test_scores_x.append(test_scores)
                psms = (PSMContainer([train_psms[j] for j in test_index])
                        + neg_psms)
                threshold, md = self._normalize_scores(scores, psms)

                cv_scores[test_index] = (test_scores - threshold) / md
                mx.append(threshold)
                nx.append(md)
                val_metrics.append(self._evaluate(y_test, test_scores, 0))
                prob_metrics.append(
                    self._evaluate(y_test, model.predict_proba(x_test)[:, 1], 0.5)
                )

            mx, nx = np.array(mx), np.array(nx)
            self._scaler = Scaler(mx, nx)
            self.metrics = _combine_metrics(val_metrics)
            self.prob_metrics = _combine_metrics(prob_metrics)
            self.cv_scores = cv_scores
            self.test_scores = test_scores_x

        else:
            model = clone(self._base_estimator)
            model.fit(x, y)
            self._estimators.append(model)

    def predict(self, x):
        if self.cv is not None:
            return np.array([e.predict(x) for e in self._estimators]).T

        return self._estimators[0].predict(x)

    def predict_proba(self, x):
        if self.cv is not None:
            return np.array([e.predict_proba(x) for e in self._estimators]).T

        return self._estimators[0].predict_proba(x)

    def decision_function(self, x, scale: bool = True):
        if self.cv is not None:
            scores = np.array(
                [e.decision_function(x) for e in self._estimators]
            ).T
            return self._scale_scores(scores) if scale else scores

        return self._estimators[0].decision_function(x)

    def validate(self, psms: PSMContainer) -> np.ndarray:
        """
        Validates `psms` using the trained model.

        """
        return self.decision_function(
            psms.to_feature_array(features=self.model_features)
        )

    @property
    def thresholds(self):
        return (
            self._scaler.centers[0] if self.cv is None else self._scaler.centers
        )

    @property
    def feature_importances_(self):
        return tuple(e.steps[1].feature_importances_ for e in self._estimators)

    @property
    def oob_scores(self):
        """
        Out of subsampling scores.

        Returns:
            n_estiamtors: number of estimators
            oob_scores: Out-of-subsampling scores
            oob_std: standard deviation of out-of-subsampling scores

        """
        scores = np.array(self._oob_scores)
        n = scores.shape[1]
        return np.arange(1, n+1) * 10, scores.mean(axis=0), scores.std(axis=0)

    def _evaluate(self, y_true, scores, threshold):
        """
        """
        _, nc = np.unique(y_true, return_counts=True)
        sensitivity = \
            np.count_nonzero(scores[y_true == 1] >= threshold) / nc[1]
        specificity = \
            np.count_nonzero(scores[y_true == 0] < threshold) / nc[0]
        bal_accuracy = (sensitivity + specificity) / 2
        auc = roc_auc_score(y_true, scores)
        return _Metrics(sensitivity, specificity, bal_accuracy, auc)

    def _normalize_scores(self, scores, psms):
        """
        Normalizes scores such that the score at the defined q_threshold is 0
        and the median of the decoy scores is -1.

        """
        scores, groups = self._reconstruct_psms(psms, scores)
        q_values = calculate_q_values(scores, groups)
        threshold = calculate_threshold_score(q_values, self.q_threshold)
        md = abs(np.median((scores - threshold)[groups == 0]))
        return threshold, md

    def _scale_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Scales classification scores.

        """
        return (scores - self._scaler.centers) / self._scaler.normalizer

    def _set_oob_score(self, scores: np.ndarray, y: np.ndarray):
        """
        Calculate out-of-sampling scores.

        """
        oobs = []
        n = scores.shape[1]
        for i in range(10, n + 10, 10):
            s = scores[:, :i].mean(axis=1)
            m = self._evaluate(y, s, 0)
            oobs.append(m.balanced_accuracy)
        self._oob_scores.append(oobs)

    def _train_matrix(self, pos_psms, decoy_psms):
        """
        Creates training data matrix

        """
        # to make this local, thus avoid looking up it in iteration
        model_features = self.model_features

        npos, ndecoy = len(pos_psms), len(decoy_psms)

        train_psms = PSMContainer(pos_psms + decoy_psms)

        # data matrices
        x = np.empty((npos + ndecoy, len(model_features)))
        for i, p in enumerate(train_psms):
            x[i] = p.features.to_list(model_features)

        y = np.concatenate((np.ones(npos), np.zeros(ndecoy)), axis=0)

        return x, y, train_psms

    def _features_to_matrix(self, psms):
        """ Creates negative data matrix """
        # to make this local, thus avoid looking up it in iteration
        model_features = self.model_features

        x = np.empty((len(psms), len(model_features)))

        for i, p in enumerate(psms):
            x[i] = p.features.to_list(model_features)

        return x

    @staticmethod
    def _reconstruct_psms(
            psms: PSMContainer,
            scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstructs PSMs based on scores for FDR estimation.

        """
        # compare the identifications to assign top score to each spectrum
        score_rec = {}
        for psm, score in zip(psms, scores):
            uid = (psm.data_id, psm.spec_id)
            if uid not in score_rec or score > score_rec[uid][0]:
                score_rec[uid] = (score, 1 if psm.target else 0)

        score_info = np.array(list(score_rec.values()))
        return score_info[:, 0], score_info[:, 1]

    def to_file(self, pickle_file: str):
        """
        Dumps the ValidationModel to a cloudpickle file.

        Args:
            pickle_file: Path to the output cloudpickle file.

        """
        with open(pickle_file, 'wb') as fh:
            cloudpickle.dump(self, fh)

    @staticmethod
    def from_file(pickle_file: str) -> ValidationModel:
        """
        Reconstructs a ValidationModel from a cloudpickle file.

        Args:
            pickle_file: Path to the cloudpickle file.

        Returns:
            ValidationModel

        """
        with open(pickle_file, 'rb') as fh:
            return cloudpickle.load(fh)
