"""
Peptide validation using ensemble SVM.

"""
from __future__ import annotations

import bisect
import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import cloudpickle
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
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
    def __init__(
            self,
            model_features: Optional[Sequence[str]] = None,
            cv: Optional[int] = None,
            q_threshold: float = 0.01,
            n_jobs: int = -1
    ):
        """
        Initializes the ValidationModel.

        Args:
            model_features: Features to be used in model construction and
                            prediction.

        """
        self._base_estimator = make_pipeline(
            MinMaxScaler(),
            RandomForest(
                n_estimators=100,
                max_depth=2,
                n_jobs=n_jobs,
                random_state=1,
                class_weight='balanced',
                max_density_samples=10000
            )
        )
        self._estimators: List[RandomForest] = []
        self.cv = cv
        self.q_threshold = q_threshold

        self.model_features = model_features

        self._scaler: Optional[Scaler] = None
        self._decoy_train_index: Optional[np.ndarray] = None

        self.metrics: Optional[ModelMetrics] = None
        self.prob_metrics: Optional[ModelMetrics] = None

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

        # data matrices
        x_pos = pos_psms.to_feature_array(features=self.model_features)
        x_decoy = decoy_psms.to_feature_array(features=self.model_features)

        # Construct model using unmodified peptide identifications
        x = np.concatenate((x_pos, x_decoy), axis=0)
        y = np.concatenate(
            (np.ones(x_pos.shape[0]), np.zeros(x_decoy.shape[0])),
            axis=0
        )

        train_psms = PSMContainer(pos_psms + decoy_psms)

        if self.cv is not None:
            skf = StratifiedKFold(n_splits=self.cv)
            mx, nx = [], []
            val_metrics, prob_metrics = [], []
            cv_scores = np.empty(x.shape[0])
            test_scores_x = []
            x_neg = neg_psms.to_feature_array(features=self.model_features)
            for train_index, test_index in skf.split(x, y):
                model = clone(self._base_estimator)
                model.fit(x[train_index], y[train_index])
                self._estimators.append(model)
                # Compute and normalize scores
                y_test, x_test = y[test_index], x[test_index]
                test_scores = model.decision_function(x_test)
                scores = np.concatenate(
                    (test_scores, model.decision_function(x_neg)), axis=0
                )
                # dnp: return the scores
                test_scores_x.append(test_scores)
                psms = \
                    PSMContainer([train_psms[j] for j in test_index]) + neg_psms
                threshold, md = self._normalize_scores(scores, psms)
                # dnp: return the normalized scores
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

    def predict(self, X):
        if self.cv is not None:
            return np.array([e.predict(X) for e in self._estimators]).T

        return self._estimators[0].predict(X)

    def predict_proba(self, X):
        if self.cv is not None:
            return np.array([e.predict_proba(X) for e in self._estimators]).T

        return self._estimators[0].predict_proba(X)

    def decision_function(self, X, scale: bool = True):
        if self.cv is not None:
            scores = np.array(
                [e.decision_function(X) for e in self._estimators]
            ).T
            return self._scale_scores(scores) if scale else scores

        return self._estimators[0].decision_function(X)

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
        return tuple(e.feature_importances_ for e in self._estimators)

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
