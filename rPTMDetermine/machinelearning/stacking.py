"""
Stacking for combining multiple classifiers for final classification.
This is stacked generalization (Wolpert DH. Neural Networks 1992, 5(2),
241-259).
Note that current module only deals with binary classification and
the label for two classes should be 0 and 1.
"""
import bisect
import collections
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold


# weights
Weight = collections.namedtuple("Weight", ["w", "c"])
# Scaler
Scaler = collections.namedtuple("Scaler", ["center", "normalizer"])
# Models from cross validation
Model = collections.namedtuple("Model", ["weights", "scaler", "combination"])


def calculate_q_values(scores, groups):
    """
    Calculates q values.

    Note:
    1. In groups, 0 is assigned to a negative identification,
       otherwise, it will be considered as a positive identification.
    2. The input scores must be sorted in DESCENDING ORDER
       in advance.

    Args:

    """
    # Calculate fdr
    fdrs = []
    num_decoy, num_target = 0, 0
    for _, group in zip(scores, groups):
        if group == 0:
            num_decoy += 1
        else:
            num_target += 1
        fdrs.append(
            num_decoy / num_target
            if num_target > 0 and num_decoy < num_target
            else 1
        )

    # Calculate q values according to obtained FDRs
    q_values = []
    fdr_c_min = 1
    for fdr in fdrs[::-1]:
        if fdr > fdr_c_min:
            q_values.append(fdr_c_min)
        else:
            q_values.append(fdr)
            fdr_c_min = fdr

    return np.array(q_values)[::-1]


class Stacking:
    """
    Stacked generalization using regularized discriminant analysis.

    """

    # Tuple for covariance and mean of the training data. The covariance
    # matrix is factorized by SVD into score matrix U and diagonal matrix
    # D storing the singular values
    _Cov = collections.namedtuple("Cov", ["mean", "cov", "U", "D", "prior"])

    def __init__(self, kfold: int = 3):
        # k fold cross validation
        self.kfold = kfold
        alphas = np.linspace(0.03, 0.51, 17)
        alphas = np.insert(alphas, [0, 0], [0.01, 0.02])
        self.alphas = alphas[:, np.newaxis]

    def _construct_cov(
            self,
            x: np.ndarray,
            y: np.ndarray,
            do_svd: bool = True
    ) -> Dict[int, _Cov]:
        """
        Constructs the covariances with decompositions by SVD for training.

        """
        y_size = y.size
        covariances = {}
        for y_label in [0, 1]:
            sel_x = x[y == y_label]
            sel_x_mean = np.mean(sel_x, axis=0)
            covariance = np.cov(sel_x.T)
            # do SVD
            u, d = None, None
            if do_svd:
                u, d, _ = np.linalg.svd(covariance)
            covariances[y_label] = self._Cov(
                sel_x_mean, covariance, u, d, np.log(sel_x.shape[0] / y_size)
            )

        return covariances

    def _construct_cov_cv(
            self,
            x: np.ndarray,
            y: np.ndarray,
            fold: int = 3
    ) -> List[Tuple[Dict[int, _Cov], np.ndarray]]:
        """
        Construct covariance matrix for RDA with double cross validation.

        """
        # Cross validation for outer data partition
        skf = StratifiedKFold(n_splits=fold)
        # Calculate covariances and mean values for cross validation
        partitions = []
        for tr_idx, val_idx in skf.split(x, y):
            covariances = self._construct_cov(
                x[tr_idx], y[tr_idx], do_svd=False
            )
            partitions.append((covariances, val_idx))
        return partitions

    @staticmethod
    def _calculate_weight(u: np.ndarray, d, m: float, alpha, prior):
        """
        Calculate RDA weights using SVD
        """
        # constants for each class
        _scorr = np.dot(u.T, m).flatten()
        _C = _scorr / (d * alpha + (1 - alpha) * np.mean(d))
        w0 = prior - np.dot(_C, _scorr) / 2
        # weights
        w = np.dot(_C, u.T)
        return w, w0

    def _calculate_weights(
            self,
            covariances: Dict[int, _Cov],
            alpha
    ) -> Dict[int, Weight]:
        """
        Calculate the weights for the classification.

        """
        weights = collections.defaultdict()
        for _label in [0, 1]:
            w, w0 = self._calculate_weight(
                covariances[_label].U,
                covariances[_label].D,
                covariances[_label].mean,
                alpha,
                covariances[_label].prior
            )
            weights[_label] = Weight(w, w0)
        return weights

    def _rda_cv(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cv_covariances: List[Tuple[Dict[int, _Cov], np.ndarray]],
            sel_index
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Regularized linear discriminant analysis
        """
        x = x[:, sel_index]

        # Select the best alpha, with probability (1 / (1 + exp(x)))
        # higher than 0.99, which corresponds to score difference x
        # lower than -ln(99).
        t = -np.log(99)

        # scores
        npos, nnegs = np.zeros(self.alphas.size), np.zeros(self.alphas.size)

        # cross validation to calculate scores
        for covariances, val_idx in cv_covariances:

            # validation set
            x_val = x[val_idx]
            y_val = y[val_idx]

            # get weights
            weights = {label: None for label in [0, 1]}
            for label in [0, 1]:
                covariance = covariances[label].cov[sel_index]
                covariance = covariance[:, sel_index]
                mean_x = covariances[label].mean[sel_index]

                # if only a single model is selected.
                if covariance.ndim < 2:
                    covariance = covariance.flatten()
                    w0 = covariances[label].prior - \
                        (mean_x ** 2 / (covariance * 2))
                    w = mean_x / covariance
                else:
                    # svd
                    u, d, _ = np.linalg.svd(covariance)
                    w, w0 = self._calculate_weight(
                        u,
                        d,
                        mean_x,
                        self.alphas,
                        covariances[label].prior
                    )
                weights[label] = Weight(w, w0)

            # predictions on testing data
            dw = weights[0].w - weights[1].w
            dc = weights[0].c - weights[1].c

            # score differences
            if dw.ndim == 2:
                val_scores = np.dot(x_val, dw.T) + dc
            else:
                val_scores = x_val * dw + dc

            npos += np.count_nonzero(val_scores[y_val == 1] <= t, axis=0)
            nnegs += np.count_nonzero(val_scores[y_val == 0] > t, axis=0)

        return npos, nnegs

    def _model_selection(self, x: np.ndarray, y: np.ndarray):
        """
        Performs model selection using double cross validation and backward
        elimination.

        """
        _c = x.shape[1]
        # Preallocate the covariance matrices and SVD decompositions
        partitions = self._construct_cov_cv(x, y)
        # Number of each group
        _, _ns = np.unique(y, return_counts=True)

        # Best combinations in each iteration
        best_num, best_del, best_alpha = [], [], []
        # Iterate through all combinations with backward elimination
        _sel = np.ones(_c, dtype=bool)
        for i in range(_c):
            _ix_c, _ns_c, _sel_alphas = np.where(_sel)[0], [], []
            if _ix_c.size == 2:
                break

            for j in _ix_c:
                _sel[j] = False

                # select best parameters using cross validation
                npos, nnegs = self._rda_cv(x, y, partitions, _sel)
                # AUC
                auc = (npos / _ns[1] + nnegs / _ns[0]) / 2
                # optimal classification
                k = np.argmax(auc)

                _ns_c.append(auc[k])
                _sel_alphas.append(self.alphas[k])
                _sel[j] = True

            # select the best combination
            j = np.argmax(_ns_c)
            _sel[_ix_c[j]] = False

            # store them
            best_del.append(_ix_c[j])
            best_num.append(_ns_c[j])
            best_alpha.append(_sel_alphas[j])

        # get the best combinations with number
        j = np.argmax(best_num)
        _sel = np.ones(_c, dtype=bool)
        _sel[best_del[:j+1]] = False

        return np.where(_sel)[0], best_num[j], best_alpha[j]

    @staticmethod
    def _rda_prediction(
            x: np.ndarray,
            weights: Dict[int, Weight]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction using RDA
        """
        scores = {_label: np.dot(x, weights[_label].w) + weights[_label].c
                  for _label in [0, 1]}
        return scores[1], 1 / (1 + np.exp(scores[0] - scores[1]))

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Adaptively fits the data using regularized discriminant analysis.

        """
        y = self._check_y(y)

        if x.shape[1] == 1:
            return x, None

        # do model selection and parameter selection using double
        # cross validation
        _n = y.size
        scores, probs, qvals = np.zeros(_n), np.zeros(_n), np.zeros(_n)

        # 3 fold cross validation for outer data partition
        skf = StratifiedKFold(n_splits=self.kfold, random_state=0)
        model_cv = []
        for train_idx, test_idx in skf.split(x, y):
            # outer cross validation
            x_train = x[train_idx]
            y_train = y[train_idx]

            # model selection using cross validation to get the best
            # combinations of models and optimized regularization
            # parameter.
            b_comb, b_num, b_alpha = self._model_selection(x_train, y_train)

            # do prediction
            covariances = self._construct_cov(x_train[:, b_comb], y_train)
            weights = self._calculate_weights(covariances, b_alpha)

            # test X
            x_test = x[test_idx][:, b_comb]
            _scores, _probs = self._rda_prediction(x_test, weights)

            # normalize the scores and get q values
            _scores, q_values, _scaler = self._normalizer(_scores, y[test_idx])

            # save the weights and scalings for future prediction
            model_cv.append(Model(weights, _scaler, b_comb))

            scores[test_idx] = _scores
            probs[test_idx] = _probs
            qvals[test_idx] = q_values

        # model selection and fitting
        b_comb, b_num, b_alpha = self._model_selection(x, y)
        covariances = self._construct_cov(x[:, b_comb], y)
        weights = self._calculate_weights(covariances, b_alpha)
        # predict training scores to get scalers which ensure
        # the q values of validated identifications
        # lower than 0.01
        _scores, _ = self._rda_prediction(x[:, b_comb], weights)
        _, _, _scaler = self._normalizer(_scores, y)

        # final models
        self.combination_index = b_comb
        self.model_cv = tuple(model_cv)
        self._best_num_ = b_num
        self._best_alpha = b_alpha
        self.stacked_weights = weights
        self.scaler = _scaler

        return scores, probs, qvals

    def predict(
            self,
            x: Union[np.ndarray, Iterable[Iterable[float]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Output predicted scores.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return self._rda_prediction(
            x[:, self.combination_index],
            self.stacked_weights
        )

    def predict_cv(
            self,
            x: Union[np.ndarray, Iterable[Iterable[float]]]
    ) -> np.ndarray:
        """
        Output predicted scores obtained from cross validation for validation
        of new identifications, by consensus voting.

        If all scores of an identification fall in q values less than
        0.01 defined in cross validation fit, then the identification
        is confirmed.
        This can guarantee that the error rates (i.e., q value) in
        validated identifications are still less than 0.01.

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # do prediction
        scores = np.empty((x.shape[0], len(self.model_cv)))
        for i, model in enumerate(self.model_cv):
            x_sel = x[:, model.combination]
            score, _ = self._rda_prediction(x_sel, model.weights)
            score -= model.scaler.center
            score /= model.scaler.normalizer
            scores[:, i] = score

        return scores

    @staticmethod
    def _normalizer(scores, y):
        """
        Normalize the scores so that median of decoy scores
        becomes -1.0 and score at q value 0.01 is 0.0
        """
        sort_index = np.argsort(scores)[::-1]

        # get q values
        q_values = calculate_q_values(scores[sort_index], y[sort_index])

        # if all identifications having q values lower than 0.01
        # return min scores
        if q_values[-1] <= 0.01:
            m = scores.min()
        else:
            j = bisect.bisect_left(q_values, 0.01)
            # if j equals 0, meaning no identification has q value
            # lower than 0.01, return 0
            j = sort_index[j]
            m = 0 if j == 0 else scores[j]

        # keep score at q value 0.01 be 0
        norm_scores = scores - m
        # normalize the score with median of decoy scores to be -1
        dm = -np.median(norm_scores[y == 0])
        norm_scores /= dm

        # sort back q values to make it consistent with original
        # order of scores
        _index = np.argsort(sort_index)

        return norm_scores, q_values[_index], Scaler(m, dm)

    def _check_y(self, y):
        """
        check the number of groups in y and whether
        the labels are 0 and 1, if not, correct it
        """
        _labels, _ns = np.unique(y, return_counts=True)

        if _labels.size != 2:
            raise ValueError("Only binary data are accepted.")

        # label correction
        if _labels.min() != 0:
            y[y == _labels.min()] = 0

        if _labels.max() != 1:
            y[y == _labels.max()] = 1

        # calculate priors for each group
        self.priors = _ns / _ns.sum()
        self._num_class = list(dict(zip(_labels, _ns)))

        return y
