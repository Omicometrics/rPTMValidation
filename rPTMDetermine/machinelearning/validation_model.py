"""
Peptide validation using ensemble SVM.

"""
from __future__ import annotations

import bisect
import random
import dataclasses
import collections
import itertools
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import cloudpickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler

from .randomforest import RandomForest


@dataclasses.dataclass
class Scaler:
    center: float
    normalizer: float


def _calculate_q(scores: np.ndarray,
                 groups: np.ndarray,
                 thresholds: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Estimate q values under different thresholds.

    Args:
        scores (np.ndarray): Scores
        groups (np.ndarray): Labels of the identifications. 1 for targets,
                             0 for decoys.
        thresholds (np.ndarray): Score thresholds for q-value calculation.

    """
    # number of decoys and targets
    nt = groups.sum()
    nd = groups.size - nt

    # bin scores
    counts_all, _ = np.histogram(scores, thresholds)
    counts_dec, _ = np.histogram(scores[groups == 0], thresholds)

    # densities
    densities_dec = np.cumsum(counts_dec)
    densities_tar = np.cumsum(counts_all) - densities_dec
    ix = densities_tar < nt

    # fdr
    fdr = (nd - densities_dec[ix]) / (nt - densities_tar[ix])

    return ix, fdr


def _estimate_p0(score_digit: np.ndarray,
                 groups: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Estimate fraction of null hypotheses using bootstrapping.

    References:
    [1] JD Storey, JE Taylor, D Siegmund. Strong Control, Conservative
        Point Estimation and Simultaneous Conservative Consistency of
        False Discovery Rates: A Unified Approach. J R Stat Soc Series
        B Stat Methodol. 2004, 66, 187-205.
    [2] JD Storey. A Direct Approach to False Discovery Rates. J R Stat
        Soc Series B Stat Methodol. 2002, 64, 479-498.
    """
    nbin = score_digit.max() + 1
    # bin counts
    counts_all = np.bincount(score_digit)
    counts_dec = np.bincount(score_digit[groups == 0])
    # cumulative sum
    cum_sum_all = np.cumsum(counts_all)
    cum_sum_dec = np.cumsum(counts_dec)
    if cum_sum_dec.size < nbin:
        cum_sum_dec = np.append(
            cum_sum_dec, [cum_sum_dec[-1]] * (nbin - cum_sum_dec.size)
        )

    cum_sum_tar = cum_sum_all - cum_sum_dec
    ix = cum_sum_dec > 0

    return ix, cum_sum_tar[ix] / cum_sum_dec[ix]


def calculate_q_values(scores: np.ndarray,
                       groups: np.ndarray) -> Tuple[np.ndarray, float]:
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
    mscore, nscore = scores.max(), scores.min()
    thresholds = np.linspace(nscore, mscore, 1001)
    nlam = 100
    lambda_ = np.linspace(nscore, mscore, nlam + 1)[1:]

    # digitize scores
    score_digit = (nlam / (scores.max() - scores.min())
                   * (scores - scores.min())).astype(int)
    score_digit[score_digit == nlam] = nlam - 1

    # calculate FDR with different p0
    ix, fdr = _calculate_q(scores, groups, thresholds)

    # bootstrap for estimating p0
    ixp, p0 = _estimate_p0(score_digit, groups)
    mp0 = p0.min()

    # do bootstrap
    n, nb = scores.size, 1000
    tix = np.arange(n)
    mse_bootstrap = np.zeros((nb, nlam))
    for i in range(nb):
        ixb = np.random.choice(tix, size=n, replace=True)
        ixt, p0b = _estimate_p0(score_digit[ixb], groups[ixb])
        mse_bootstrap[i][:ixt.size][ixt] = p0b - mp0

    # lambda at minimum MSE
    mse = (mse_bootstrap * mse_bootstrap).mean(axis=0)
    # use 0.9 to avoid sudden drop at the end.
    tix, = np.where(mse > 0)
    i = tix[np.argmin(mse[mse > 0][:int(nlam * 0.9)])]
    p0_best = p0[lambda_[ixp] == lambda_[i]]

    # q values
    q, _q_min = [], 1
    for _fdr in fdr:
        if _fdr > _q_min:
            q.append(_q_min)
        else:
            q.append(_fdr)
            _q_min = _fdr

    return np.array([thresholds[1:][ix], np.array(q) * p0_best]), p0_best


def calculate_threshold_score(q_values: np.ndarray, threshold: float) -> float:
    """
    Finds the score value at `q_threshold`.

    Args:
        q_values: Array of calculated q-values.
        threshold: Cutoff q-value.

    Returns:
        Score for the required cutoff.

    """
    q_values = q_values[:, q_values[1].argsort()]
    scores, q_values = q_values[0], q_values[1]
    # return min scores if all q values are lower than threshold
    if q_values[-1] <= threshold:
        return scores[-1]

    j = bisect.bisect_left(q_values, threshold)
    return scores[j - 1]


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
                 cv: Optional[int] = 3,
                 q_threshold: float = 0.01,
                 n_estimators: Optional[int] = None,
                 max_samples: Optional[int] = 3000,
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
        self._estimators: List[Pipeline] = []
        self.oob_score = oob_score
        self.cv = cv
        self.q_threshold = q_threshold

        self.model_features = model_features

        self.score_threshold: Optional[float] = None
        self._scaler: Optional[Scaler] = None
        self._p0: Optional[np.ndarray] = None
        self._rf_params = {
            "n_estimators": n_estimators, "max_depth": max_depth,
            "n_jobs": n_jobs, "random_state": 1, "max_samples": max_samples,
            "max_density_samples": 10000, "oob_score": oob_score
        }

        self.metrics: Optional[ModelMetrics] = None
        self.prob_metrics: Optional[ModelMetrics] = None

        self._best_params: List[tuple] = []
        self._internal_cv_params: List[dict] = []

    def fit(self,
            pos_psms: PSMContainer,
            decoy_psms: PSMContainer,
            neg_psms: PSMContainer):
        """Constructs validation model.

        Args:
            pos_psms: Positive (pass FDR control) identifications.
            decoy_psms: Decoy identifications.
            neg_psms: Negative (fail FDR control) identifications. Only required
                      if cv is not None when constructing the ValidationModel.

        Returns:
            val_psms: List of validated PSMs
            model_index: Spectrum ID
        """
        # randomize data matrix to avoid bias caused by ordered data
        train_psms, rest_psms, pos_uids, dec_uids, neg_uids =\
            self._randomize_psm(pos_psms, decoy_psms, neg_psms)

        # get training and negative uids
        npos, ndec, nneg = len(pos_uids), len(dec_uids), len(neg_uids)
        train_uids = np.array(pos_uids + dec_uids, dtype=str)
        y = np.concatenate((np.ones(npos), np.zeros(ndec)), axis=0)

        # stratified K fold
        ncv = self.cv
        skf = StratifiedKFold(n_splits=ncv)

        # do cross-validation
        val_metrics, prob_metrics = [], []
        for i, (tr_idx, te_idx) in enumerate(skf.split(train_uids, y)):

            # training data matrix and negatives
            psms_tr = list(
                itertools.chain(*[train_psms[u] for u in train_uids[tr_idx]])
            )
            psms_te = list(
                itertools.chain(*[train_psms[u] for u in train_uids[te_idx]])
            )

            # train the model and do validation
            model, sc_te, mt, pmt, pm, bpm = self._fit_model(psms_tr, psms_te)

            val_metrics.append(mt)
            prob_metrics.append(pmt)
            self._internal_cv_params += pm
            self._best_params += bpm
            self._estimators.append(model)

        # perform validation for all PSMs
        psms = pos_psms + decoy_psms + neg_psms
        _ = self.validate(psms)
        # reconstruct PSMs
        rec_psms = self._reconstruct_psms(psms)

        # perform FDR control
        val_scores = np.array([p.site_score for p in rec_psms])
        groups = np.array([1 if p.target else 0 for p in rec_psms])
        # FDR
        qvals, p0 = calculate_q_values(val_scores, groups)
        # FDR threshold
        opt_thr = calculate_threshold_score(qvals, 0.01)

        self.metrics = _combine_metrics(val_metrics)
        self.prob_metrics = _combine_metrics(prob_metrics)
        self.score_threshold = opt_thr
        self._p0 = p0
        self._scaler = Scaler(
            center=opt_thr,
            normalizer=abs(np.median(val_scores[groups == 0]) - opt_thr)
        )

        return [p for p in rec_psms if p.site_score >= opt_thr and p.target]

    def predict(self, x):
        if self.cv is not None:
            return np.array([e.predict(x) for e in self._estimators]).T

        return self._estimators[0].predict(x)

    def predict_proba(self, x):
        if self.cv is not None:
            return np.array([e.predict_proba(x) for e in self._estimators]).T

        return self._estimators[0].predict_proba(x)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.cv is not None:
            return np.array(
                [e.decision_function(x) for e in self._estimators]
            ).T

        return self._estimators[0].decision_function(x)

    def validate(self, psms):
        """ Validates `psms` using the trained model. """
        n, k = len(psms), 10000
        nb = int(n / k) + 1
        # :: calculate scores in blocks
        for i in range(nb):
            i0, i1 = i*k, min((i+1)*k, n)
            x = self._features_to_matrix(psms[i0:i1])
            scores = self.decision_function(x)
            # average the scores
            mscores = scores.mean(axis=1)
            for p, score, mscore in zip(psms[i0:i1], scores, mscores):
                p.validation_score = mscore
                p.ml_scores = score
        return psms

    @property
    def thresholds(self):
        return self.score_threshold

    @property
    def feature_importances_(self):
        return tuple(e.steps[1].feature_importances_ for e in self._estimators)

    def _fit_model(self, psms_train, psms_test):
        """ Fit the model and do validation """
        # make pipeline
        model = self._make_pipeline()

        # train and test data matrix
        xtr, ytr = self._features_to_matrix(psms_train, return_y=True)
        xte, yte = self._features_to_matrix(psms_test, return_y=True)

        # internal cross validation: 5 fold, to select best parameters
        params, best_params = [], []
        if model is None:
            params_, bparams = self._internal_cv(xtr, ytr)
            model = self._make_pipeline(*bparams)
            params.append(params_)
            best_params.append(bparams)
        model.fit(xtr, ytr)

        # evaluation
        scores = model.decision_function(xte)

        # model estimation
        metrics = self._evaluate(yte, scores, threshold=0)
        prob_metrics = self._evaluate(yte, model.predict_proba(xte)[:, 1])

        return model, scores, metrics, prob_metrics, params, best_params

    @staticmethod
    def _evaluate(y_true, scores, threshold=0.5):
        """
        Assess the performance of the validation model by calculating
        sensitivity, specificity, balanced accuracy and ROC.

        """
        _, nc = np.unique(y_true, return_counts=True)
        sensitivity = \
            np.count_nonzero(scores[y_true == 1] >= threshold) / nc[1]
        specificity = \
            np.count_nonzero(scores[y_true == 0] < threshold) / nc[0]
        bal_accuracy = (sensitivity + specificity) / 2
        auc = roc_auc_score(y_true, scores)
        return _Metrics(sensitivity, specificity, bal_accuracy, auc)

    def _features_to_matrix(self, psms, return_y=False):
        """ Creates data matrix. """
        # to make this local, thus avoid looking up it in iteration
        model_features = self.model_features

        x, y = np.empty((len(psms), len(model_features))), np.zeros(len(psms))
        for i, p in enumerate(psms):
            x[i] = p.features.to_list(model_features)
            if p.target:
                y[i] = 1
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0

        if return_y:
            return x, y
        return x

    @staticmethod
    def _reconstruct_psms(psms) -> list:
        """ Reconstructs PSMs based on scores for FDR estimation. """
        # compare the identifications to assign top score to each spectrum
        score_rec = collections.defaultdict()
        for psm in psms:
            uid = (psm.data_id, psm.spec_id)
            if (uid not in score_rec
                    or psm.validation_score > score_rec[uid].validation_score):
                score_rec[uid] = psm

        return list(score_rec.values())

    @staticmethod
    def _randomize_psm(pos_psms, decoy_psms, neg_psms):
        """ Randomize data. """

        def _combine_psms(cpsms, group_psms):
            """ Combine PSMs according to spectrum ID """
            for psm in group_psms:
                uid = f"{psm.data_id}#{psm.spec_id}"
                cpsms[uid].append(psm)
            return cpsms

        # seed of randomness for reproducibility
        random.seed(1)

        train_psms = collections.defaultdict(list)
        # positives
        train_psms = _combine_psms(train_psms, pos_psms)
        pos_uids = list(train_psms.keys())
        random.shuffle(pos_uids)

        # decoys
        dec_uids = list(set(f"{p.data_id}#{p.spec_id}" for p in decoy_psms))
        train_psms = _combine_psms(train_psms, decoy_psms)
        random.shuffle(dec_uids)

        # negatives
        rest_psms = collections.defaultdict(list)
        rest_psms = _combine_psms(rest_psms, neg_psms)
        neg_uids = list(set(f"{p.data_id}#{p.spec_id}" for p in neg_psms)
                        - set(train_psms.keys()))
        random.shuffle(neg_uids)

        return train_psms, rest_psms, pos_uids, dec_uids, neg_uids

    def _internal_cv(self, x, y, cv=5):
        """ Internal cross-validation: defaulted 5 fold. """
        # max number of samples
        if x.shape[0] < 500:
            raise ValueError("Too few PSMs for constructing validation model."
                             " Not supported.")

        if x.shape[0] < 1000:
            warnings.warn(
                "Too few PSMs for constructing validation model",
                DeprecationWarning
            )
            max_samples, n_estimators = [x.shape[0]], 500
        else:
            max_samples = [k for k in [1000, 2000, 3000, 5000, 8000]
                           if k < int(x.shape[0] / cv) - 1]
            n_estimators = 500

        best_params, params, best_ba = None, {}, 0
        skf = StratifiedKFold(n_splits=cv)
        for m in max_samples:
            # do cross validation
            scores = np.empty((x.shape[0], n_estimators))
            for tr_idx, te_idx in skf.split(x, y):
                est_ = self._make_pipeline(n_estimators, m)
                est_.fit(x[tr_idx], y[tr_idx])

                # test data score matrix
                x_test = est_["minmaxscaler"].transform(x[te_idx])
                _, sx = est_["randomforest"].decision_function(
                    x_test, return_matrix=True
                )
                scores[te_idx] = sx

            # get the parameter with highest balanced_accuracy
            bas = []
            for i in range(100, n_estimators + 10, 10):
                s = scores[:, :i].mean(axis=1)
                mtr = self._evaluate(y, s, threshold=0)
                if mtr.balanced_accuracy > best_ba:
                    best_params = (i, m)
                    best_ba = mtr.balanced_accuracy
                # record the parameters
                bas.append(mtr.balanced_accuracy)
            params[m] = bas

        return params, best_params

    def _make_pipeline(self, n_estimators=None, max_sample=None):
        """ Create model pipeline. """
        rf_params = self._rf_params.copy()
        if n_estimators is None:
            if rf_params["n_estimators"] is not None:
                if rf_params["max_samples"] is None:
                    raise ValueError("max_sample must be set if"
                                     " n_estimators is specified.")
                return make_pipeline(MinMaxScaler(), RandomForest(**rf_params))
            return None

        rf_params.update([("n_estimators", n_estimators),
                          ("max_samples", max_sample)])
        return make_pipeline(MinMaxScaler(), RandomForest(**rf_params))

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
