"""
Peptide validation using ensemble SVM.

"""
import bisect
import collections
import dataclasses
import itertools
from typing import Optional, Sequence, Tuple

import numpy as np

from .machinelearning import Classifier, SKEstimator, subsample_negative
from .psm_container import PSMContainer


@dataclasses.dataclass
class Scaler:
    centers: np.ndarray
    normalizer: np.ndarray


# Fraction for subsampling from minority class.
# NOTE: these fractions are set empirically.
SUBSAMPLE_FRACTIONS = [
    (1, 0),
    (0.8, 1000),
    (0.6, 3000),
    (0.3, 5000),
    (0.2, 10000),
    (0.1, 3e4),
    (0.08, 4e4),
    (0.06, 6e4),
    (0.03, 1e5),
    (0.02, 2e5),
    (0.008, 5e5),
    (0.004, 1e6)
]


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


class ValidationModel:
    """
    Validation of peptide identification.

    """
    def __init__(
            self,
            estimator: SKEstimator,
            model_features: Optional[Sequence[str]] = None,
            **kwargs
    ):
        """
        Initializes the class.

        Args:
            estimator: Scikit-learn classifier.
            model_features: Features to be used in model construction and
                            prediction.
            **kwargs: Additional keyword arguments for the composed classifier.

        """
        self.classifier = Classifier(estimator, **kwargs)
        self.model_features = model_features

        self._scaler: Optional[Scaler] = None
        self._decoy_train_index: Optional[np.ndarray] = None

    def fit_and_normalize(
            self,
            pos_psms: PSMContainer,
            decoy_psms: PSMContainer,
            neg_psms: PSMContainer,
            **kwargs
    ) -> Tuple[PSMContainer, PSMContainer, PSMContainer]:
        """
        Fits and normalizes the validation model.

        Args:
            pos_psms: Positive (passed FDR) peptide spectrum matches.
            decoy_psms: Decoy peptide spectrum matches.
            neg_psms: Negative (failed FDR) peptide spectrum matches.

        """
        self.fit(pos_psms, decoy_psms)
        decoy_psms = self.normalize(pos_psms, decoy_psms, neg_psms, **kwargs)
        return pos_psms, decoy_psms, neg_psms

    def fit(self, pos_psms: PSMContainer, decoy_psms: PSMContainer):
        """ Construct validation model. """
        # data matrices
        x_pos = pos_psms.to_feature_array(features=self.model_features)
        x_decoy = decoy_psms.to_feature_array(features=self.model_features)

        # Subsampling to reduce the number of decoys for training
        dists = subsample_negative(x_pos, x_decoy)
        sorted_idx = np.argsort(dists)[::-1]
        decoy_train_index = np.sort(sorted_idx[:int(sorted_idx.size * 0.2)])
        self._decoy_train_index = decoy_train_index

        # Construct model using unmodified peptide identifications
        x_decoy = x_decoy[decoy_train_index]
        x = np.concatenate((x_pos, x_decoy), axis=0)
        num_pos, num_decoy = x_pos.shape[0], x_decoy.shape[0]
        y = np.concatenate((np.ones(num_pos), np.zeros(num_decoy)), axis=0)

        # Fraction for subsampling
        sub_sample_fraction = min(
            _r for _r, _t in SUBSAMPLE_FRACTIONS if min(num_pos, num_decoy) > _t
        )
        # Update parameter for fraction of subsampling
        self.classifier.sub_sample_fraction = sub_sample_fraction

        # train the model.
        self.classifier.fit(x, y)

    def normalize(
            self,
            pos_psms: PSMContainer,
            decoy_psms: PSMContainer,
            neg_psms: PSMContainer,
            q_threshold: float = 0.01
    ) -> PSMContainer:
        """
        Normalize scores so that the score at `q_threshold` is 0 and median of
        decoy scores is -1.

        Args:
            pos_psms: Positive (passed FDR) peptide spectrum matches.
            decoy_psms: Decoy peptide spectrum matches.
            neg_psms: Negative (failed FDR) peptide spectrum matches.
            q_threshold: q value threshold for scaling. Default is 0.01.

        """
        # update PSMs with scores
        self._update_training_psms(pos_psms, decoy_psms)

        # predict rest decoys not used in training.
        decoy_index = np.arange(len(decoy_psms), dtype=int)
        decoy_index = np.delete(decoy_index, self._decoy_train_index)
        train_decoys = PSMContainer(
            [decoy_psms[i] for i in self._decoy_train_index]
        )
        rest_decoys = PSMContainer([decoy_psms[i] for i in decoy_index])
        if decoy_index.size > 0:
            self.validate_psms(rest_decoys)

        # validate negatives
        self.validate_psms(neg_psms)

        # validation scores of additional identifications
        train_psms = PSMContainer(pos_psms + train_decoys)
        extra_psms = PSMContainer(neg_psms + rest_decoys)
        extra_scores = np.array([psm.ml_scores for psm in extra_psms])
        train_scores = np.array([psm.ml_scores for psm in train_psms])

        # calculate score threshold at q_threshold, and normalize the scores.
        mx, nx = [], []
        for i in range(self.classifier.stacking_kfold):
            test_index, = np.where(self.classifier.stacking_test_index == i)
            # construct scores
            psm_set = \
                PSMContainer([train_psms[j] for j in test_index]) + extra_psms
            score_set = np.concatenate(
                (train_scores[test_index], extra_scores[:, i]), axis=0
            )
            scores, groups = self._reconstruct_psms(psm_set, score_set)

            # reconstruct identifications to calculate q values.
            q_values = calculate_q_values(scores, groups)

            # do normalization
            m = calculate_threshold_score(q_values, q_threshold)
            md = np.median(scores[groups == 0])
            mx.append(m)
            nx.append(m - md)

        mx, nx = np.array(mx), np.array(nx)
        self._scaler = Scaler(mx, nx)

        # normalize training scores
        self._normalize_training_scores(pos_psms, decoy_psms)

        # normalize extra identifications
        for psm in itertools.chain(neg_psms, rest_decoys):
            psm.ml_scores = (psm.ml_scores - mx) / nx

        decoy_psms = train_decoys + rest_decoys

        return decoy_psms

    def validate_psms(
            self,
            psms: PSMContainer,
            use_cv: bool = True
    ):
        """
        Validates unknown PSMs in-place.

        Args:
            psms: PSM objects to validate.
            use_cv: Type of scores returned. Setting to `True` will return
                    cross validated scores with size of n samples by k fold
                    cross validations, otherwise singe score for each PSM
                    is returned. Default is `True`.

        """
        if len(psms) == 0:
            return psms

        x = psms.to_feature_array(features=self.model_features)

        scores = self.predict(x, use_cv=use_cv)
        for psm, _scores in zip(psms, scores):
            psm.ml_scores = _scores

    def predict(self, x: np.ndarray, use_cv: bool = True) -> np.ndarray:
        """
        Calculate validation scores.

        Args:
            x: Feature matrix for score calculation.
            use_cv: Type of scores returned. Setting to `True` will return
                    cross validated scores with size of `n` samples by k fold
                    cross validations, otherwise singe score for each PSM
                    is returned. Default is `True`.

        Returns:
            Scores.

        """
        scores = self.classifier.predict(x, use_cv=use_cv)
        # if the scaler is not assigned or cross validation scores
        # are not required, return native model scores.
        if self._scaler is None or not use_cv:
            return scores

        # otherwise, return normalized scores.
        return self._scale_scores(scores)

    def _scale_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Scales classification scores.

        """
        return (scores - self._scaler.centers) / self._scaler.normalizer

    def _normalize_training_scores(
            self,
            pos_psms: PSMContainer,
            decoy_psms: PSMContainer
    ):
        """
        Normalizes training scores.

        """
        for i, (c, n) in enumerate(
                zip(self._scaler.centers, self._scaler.normalizer)
        ):
            scores = self.classifier.train_scores[
                self.classifier.stacking_test_index == i
            ]
            self.classifier.train_scores[
                self.classifier.stacking_test_index == i
            ] = (scores - c) / n
        self._update_training_psms(pos_psms, decoy_psms)

    @staticmethod
    def _reconstruct_psms(
            psms: PSMContainer,
            scores: np.ndarray
    ):
        """
        Reconstructs PSMs based on scores for FDR estimation.

        """
        # compare the identifications to assign top score to each spectrum
        score_rec = collections.defaultdict()
        for _psm, _sk in zip(psms, scores):
            _sp_id = (_psm.data_id, _psm.spec_id)
            if _sp_id not in score_rec or _sk > score_rec[_sp_id][0]:
                score_rec[_sp_id] = (_sk, 1 if _psm.target else 0)

        score_info = np.array(list(score_rec.values()))
        return score_info[:, 0], score_info[:, 1]

    def _update_training_psms(
            self,
            pos_psms: PSMContainer,
            decoy_psms: PSMContainer
    ):
        """
        Updates PSM scores in training.

        """
        num_pos = len(pos_psms)
        # Update positive PSMs
        for i, psm in enumerate(pos_psms):
            psm.ml_scores = self.classifier.train_scores[i]

        # Update decoy PSMs for training
        for i, scores in zip(
                self._decoy_train_index, self.classifier.train_scores[num_pos:]
        ):
            decoy_psms[i].ml_scores = scores
