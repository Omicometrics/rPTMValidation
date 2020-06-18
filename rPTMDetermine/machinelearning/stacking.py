"""
Stacking for combining multiple classifiers for final classification
using regularized discriminant analysis (RDA). This is so-called stacked
generalization.

Note that current module only deals with binary classification and
the label for two classes should be 0 and 1.

References:
[1] Guo YQ, et al. Biostatistics. 2007, 8, 86–100.
[2] Wolpert DH. Neural Networks 1992, 5(2), 241-259.

"""
import collections
import dataclasses
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclasses.dataclass
class Weight:
    w: np.ndarray
    c: np.ndarray


@dataclasses.dataclass
class Model:
    weights: Dict[int, Weight]
    combination: np.ndarray


class Stacking:
    """
    Stacked generalization using regularized discriminant analysis (RDA).

    """

    # Covariance and mean of training data. The covariance matrix is
    # factorized by SVD into score matrix U and singular values D.
    _Cov = collections.namedtuple("Cov", ["mean", "cov", "U", "D", "prior"])

    def __init__(self, kfold: int = 3):
        """
        Initializes the class.

        Args:
            kfold: The number of cross validation folds.

        """
        self.kfold = kfold
        # regularization parameter
        alphas = np.linspace(0.03, 0.51, 17)
        alphas = np.insert(alphas, [0, 0], [0.01, 0.02])
        self.alphas: np.ndarray = alphas[:, np.newaxis]

        self.combination_index: Optional[np.ndarray] = None
        self.model_cv: Optional[Tuple] = None
        self._best_num: Optional[float] = None
        self._best_alpha: Optional[float] = None
        self.stacked_weights: Optional[Dict[int, Weight]] = None
        self.test_index: Optional[np.ndarray] = None

    def _construct_cov(
            self,
            x: np.ndarray,
            y: np.ndarray,
            do_svd: bool = True
    ) -> Dict[int, _Cov]:
        """
        Constructs the covariances with decompositions by SVD for training.

        Args:
            x: Feature matrix.
            y: Label matrix.
            do_svd: Boolean indicating whether SVD should be performed.
                    Defaults to `True`.

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
            kfold: int = 3
    ) -> List[Tuple[Dict[int, _Cov], np.ndarray]]:
        """
        Constructs covariance matrix for RDA with double CV.

        Args:
            x: Feature matrix.
            y: Label matrix.
            kfold: Number of cross validation folds.

        Returns:
            Covariances and test indices for each cross validation fold.

        """
        # 3 fold cross validation for outer data partition
        skf = StratifiedKFold(n_splits=kfold, random_state=0)
        # calculate covariances and mean values for cross validation
        partitions = []
        for train_idx, test_idx in skf.split(x, y):
            covariances = self._construct_cov(
                x[train_idx], y[train_idx], do_svd=False
            )
            partitions.append((covariances, test_idx))
        return partitions

    @staticmethod
    def _calculate_weight(
            u: np.ndarray,
            d,
            m: float,
            alpha,
            prior
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates RDA weights using SVD.

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
        Calculates the weights for the classification.

        """
        weights: Dict[int, Weight] = {}
        for label in [0, 1]:
            covariance = covariances[label]
            w, w0 = self._calculate_weight(
                covariance.U,
                covariance.D,
                covariance.mean,
                alpha,
                covariance.prior
            )
            weights[label] = Weight(w, w0)
        return weights

    def _rda_cv(
            self,
            x: np.ndarray,
            y: np.ndarray,
            cv_covariances: List[Tuple[Dict[int, _Cov], np.ndarray]],
            sel_index
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Regularized linear discriminant analysis.

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
                covariance = covariances[label].cov[sel_index][:, sel_index]
                prior = covariances[label].prior
                mean_x = covariances[label].mean[sel_index]

                # if only a single model is selected.
                if covariance.ndim < 2:
                    covariance = covariance.flatten()
                    w0 = prior - (mean_x ** 2 / (covariance * 2))
                    w = mean_x / covariance
                else:
                    # svd
                    u, d, _ = np.linalg.svd(covariance)
                    w, w0 = self._calculate_weight(
                        u, d, mean_x, self.alphas, prior
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

    def _model_selection(
            self,
            x: np.ndarray,
            y: np.ndarray,
            kfold: int = 3
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select model using double cross validation and backward elimination.

        Args:
            x: Feature matrix.
            y: Label matrix.
            kfold: Number of cross validation folds.

        """
        _c = x.shape[1]
        # Preallocate the covariance matrices and SVD decompositions
        partitions = self._construct_cov_cv(x, y, kfold=kfold)
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
            j = int(np.argmax(_ns_c))
            _sel[_ix_c[j]] = False

            # store them
            best_del.append(_ix_c[j])
            best_num.append(_ns_c[j])
            best_alpha.append(_sel_alphas[j])

        # get the best combinations with number
        j = int(np.argmax(best_num))
        _sel = np.ones(_c, dtype=bool)
        _sel[best_del[:j+1]] = False

        return np.where(_sel)[0], best_num[j], best_alpha[j]

    @staticmethod
    def _rda_prediction(
            x: np.ndarray,
            weights: Dict[int, Weight]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Prediction using RDA. """
        scores = {label: np.dot(x, weights[label].w) + weights[label].c
                  for label in [0, 1]}
        return scores[1], 1 / (1 + np.exp(scores[0] - scores[1]))

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptively fits the data using RDA.

        Args:
            x: Feature matrix with size of n samples by p features.
            y: Groups of samples, with size of n.

        Returns:
            Tuple of stacked scores and posterior probabilities.

        """
        y = self._check_y(y)

        if x.shape[1] == 1:
            return x, None

        # do model selection and parameter selection using double
        # cross validation
        y_size = y.size
        scores, probs = np.zeros(y_size), np.zeros(y_size)

        # Cross validation for outer data partition
        skf = StratifiedKFold(n_splits=self.kfold, random_state=0)
        test_indices = np.empty(y_size)
        model_cv = []
        for i, (train_idx, test_idx) in enumerate(skf.split(x, y)):
            # outer cross validation
            x_train = x[train_idx]
            y_train = y[train_idx]
            test_indices[test_idx] = i

            # model selection using cross validation to get the best
            # combinations of models and optimized regularization
            # parameter.
            b_comb, b_num, b_alpha = self._model_selection(
                x_train, y_train, kfold=2
            )

            # do prediction
            covariances = self._construct_cov(x_train[:, b_comb], y_train)
            weights = self._calculate_weights(covariances, b_alpha)

            # test X
            x_test = x[test_idx][:, b_comb]
            _scores, _probs = self._rda_prediction(x_test, weights)

            # save the weights and scalings for future prediction
            model_cv.append(Model(weights, b_comb))

            scores[test_idx] = _scores
            probs[test_idx] = _probs

        # model selection and fitting
        b_comb, b_num, b_alpha = self._model_selection(x, y)
        covariances = self._construct_cov(x[:, b_comb], y)
        weights = self._calculate_weights(covariances, b_alpha)
        # predict training scores to get scalers which ensure
        # the q values of validated identifications
        # lower than 0.01
        _scores, _ = self._rda_prediction(x[:, b_comb], weights)

        # final models
        self.combination_index = b_comb
        self.model_cv = tuple(model_cv)
        self._best_num = b_num
        self._best_alpha = b_alpha
        self.stacked_weights = weights
        self.test_index = test_indices.astype(int)

        return scores, probs

    def predict(
            self,
            x: Union[np.ndarray, Iterable[Iterable[float]]]
    ):
        """
        Calculate predicted scores.

        Args:
            x: Feature matrix for prediction.

        Returns:
            Validation scores.

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
        Calculates scores using cross validation.

        Args:
            x: Feature matrix.

        Returns:
            Prediction scores from cross validation.

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # do prediction
        scores = np.empty((x.shape[0], len(self.model_cv)))
        for i, model in enumerate(self.model_cv):
            x_sel = x[:, model.combination]
            score, _ = self._rda_prediction(x_sel, model.weights)
            scores[:, i] = score

        return scores

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
