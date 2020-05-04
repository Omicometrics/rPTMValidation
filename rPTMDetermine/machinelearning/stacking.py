"""
Stacking for combining multiple classifiers for final classification.
This is stacked generalization (Wolpert DH. Neural Networks 1992, 5(2),
241-259).
Note that current module only deals with binary classification and
the label for two classes should be 0 and 1.
"""
import bisect
import collections

import numpy as np
from sklearn.model_selection import StratifiedKFold


# weights
Weight = collections.namedtuple("Weight", ["w", "c"])
# Scaler
Scaler = collections.namedtuple("Scaler", ["center", "normalizer"])
# Models from cross validation
Model = collections.namedtuple("Model", ["weights", "scaler", "combination"])


def _get_qvalues(scores, groups):
    """
    Calculate q values.
    Note:
    1. In groups, 0 is assigned to a decoy identification,
       otherwise, it will be considered as a target identification.
    2. The input scores must be sorted in DESCENDING ORDER
       in advance.
    """

    # calculate fdr
    fdr = []
    _nd, _nt = 0, 0
    for _s, _g in zip(scores, groups):
        if _g == 0:
            _nd += 1
        else:
            _nt += 1
        fdr.append(_nd / _nt if _nt > 0 and _nd < _nt else 1)

    # calculate q values according to obtained fdrs
    qvalues = []
    _fdr_c_min = 1
    for _fdr in fdr[::-1]:
        if _fdr > _fdr_c_min:
            qvalues.append(_fdr_c_min)
        else:
            qvalues.append(_fdr)
            _fdr_c_min = _fdr

    return np.array(qvalues)[::-1]


class Stacking():
    """
    Stacked generalization using regularized discriminant analysis
    """

    # tuple for covariance and mean of the training data. The covariance
    # matrix is factorized by SVD into score matrix U and diagonal matrix
    # D storing the singular values
    _Cov = collections.namedtuple("Cov", ["mean", "cov", "U", "D", "prior"])

    def __init__(self, kfold=3):
        # k fold cross validation
        self._kfold = kfold
        alphas = np.linspace(0.03, 0.51, 17)
        alphas = np.insert(alphas, [0, 0], [0.01, 0.02])
        self.alphas = alphas[:, np.newaxis]

    def _construct_cov(self, X, y, do_svd=True):
        """
        Construct the covariances with decompositions by SVD
        for training.
        """
        _n = y.size
        covs = {}
        for _y in [0, 1]:
            _X_c = X[y == _y]
            _x = np.mean(_X_c, axis=0)
            _cov = np.cov(_X_c.T)
            # do SVD
            u, d = None, None
            if do_svd:
                u, d, _ = np.linalg.svd(_cov)
            covs[_y] = self._Cov(_x, _cov, u, d, np.log(_X_c.shape[0] / _n))
        return covs

    def _construct_cov_cv(self, X, y, fold=3):
        """
        Construct covariance matrix for RDA with
        double cross validation
        """
        # 3 fold cross validation for outer data partition
        skf = StratifiedKFold(n_splits=fold, random_state=0)
        # calcualte covariances and mean values for cross
        # validation
        partitions = []
        for tr_idx, val_idx in skf.split(X, y):
            _cov = self._construct_cov(X[tr_idx], y[tr_idx], do_svd=False)
            partitions.append((_cov, val_idx))
        return partitions

    def _calculate_weight(self, u, d, m, alpha, prior):
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

    def _calculate_weights(self, covs, alpha):
        """
        Calculate the weights for the classification
        """
        weights = collections.defaultdict()
        for _label in [0, 1]:
            w, w0 = self._calculate_weight(covs[_label].U, covs[_label].D,
                                           covs[_label].mean, alpha,
                                           covs[_label].prior)
            weights[_label] = Weight(w, w0)
        return weights

    def _rda_cv(self, X, y, cv_covs, sel_index):
        """
        Regularized linear discriminant analysis
        """
        X = X[:, sel_index]

        # select the best alpha, with probability (1 / (1 + exp(x)))
        # higher than 0.99, which corresponds to score difference x
        # lower than -ln(99).
        t = -np.log(99)

        # scores
        npos, nnegs = np.zeros(self.alphas.size), np.zeros(self.alphas.size)

        # cross validation to calculate scores
        for _cov, _val_idx in cv_covs:

            # validation set
            Xval = X[_val_idx]
            yval = y[_val_idx]

            # get weights
            _weights = {_label: None for _label in [0, 1]}
            for _label in [0, 1]:
                _cov_c = _cov[_label].cov[sel_index]
                _cov_c = _cov_c[:, sel_index]
                _x = _cov[_label].mean[sel_index]

                # if only a single model is selected.
                if _cov_c.ndim < 2:
                    _cov_c = _cov_c.flatten()
                    w0 = _cov[_label].prior - (_x ** 2 / (_cov_c * 2))
                    w = _x / _cov_c
                else:
                    # svd
                    u, d, _ = np.linalg.svd(_cov_c)
                    w, w0 = self._calculate_weight(u, d, _x, self.alphas,
                                                   _cov[_label].prior)
                _weights[_label] = Weight(w, w0)

            # predictions on testing data
            dw = _weights[0].w - _weights[1].w
            dc = _weights[0].c - _weights[1].c

            # score differences
            if dw.ndim == 2:
                val_scores = np.dot(Xval, dw.T) + dc
            else:
                val_scores = Xval * dw + dc

            npos += np.count_nonzero(val_scores[yval == 1] <= t, axis=0)
            nnegs += np.count_nonzero(val_scores[yval == 0] > t, axis=0)

        return npos, nnegs

    def _model_selection(self, X, y):
        """
        Do model selection using double cross validation and
        backward elimination.
        """
        _c = X.shape[1]
        # preallocate the covariance matrices and SVD decompositions
        partitions = self._construct_cov_cv(X, y)
        # number of each group
        _, _ns = np.unique(y, return_counts=True)

        # best combinations in each iteration
        best_num, best_del, best_alpha = [], [], []
        # iterate through all combinations with backward elimination
        _sel = np.ones(_c, dtype=bool)
        for i in range(_c):
            _ix_c, _ns_c, _sel_alphas = np.where(_sel)[0], [], []
            if _ix_c.size == 2:
                break

            for j in _ix_c:
                _sel[j] = False

                # select best parameters using cross validation
                npos, nnegs = self._rda_cv(X, y, partitions, _sel)
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

    def _rda_prediction(self, X, weights):
        """
        Prediction using RDA
        """
        scores = {_label: np.dot(X, weights[_label].w) + weights[_label].c
                  for _label in [0, 1]}
        return scores[1], 1 / (1 + np.exp(scores[0] - scores[1]))

    def fit(self, X, y):
        """
        Adatpively fit the data using regularized discriminant analysis.
        """
        y = self._check_y(y)

        # do model selection and parameter selection using double
        # cross validation
        _n = y.size
        scores, probs, qvals = np.zeros(_n), np.zeros(_n), np.zeros(_n)

        # 3 fold cross validation for outer data partition
        skf = StratifiedKFold(n_splits=self._kfold, random_state=0)
        model_cv = []
        for tr_idx, te_idx in skf.split(X, y):
            # outer cross validation
            Xtr = X[tr_idx]
            ytr = y[tr_idx]

            # model selection using cross validation to get the best
            # combinations of models and optimized regularization
            # parameter.
            b_comb, b_num, b_alpha = self._model_selection(Xtr, ytr)

            # do prediction
            _covs = self._construct_cov(Xtr[:, b_comb], ytr)
            _weights = self._calculate_weights(_covs, b_alpha)

            # test X
            Xte = X[te_idx][:, b_comb]
            _scores, _probs = self._rda_prediction(Xte, _weights)

            # normalize the scores and get q values
            _scores, _qvalues, _scaler = self._normalizer(_scores, y[te_idx])

            # save the weights and scalings for future prediction
            model_cv.append(Model(_weights, _scaler, b_comb))

            scores[te_idx] = _scores
            probs[te_idx] = _probs
            qvals[te_idx] = _qvalues

        # model selection and fitting
        b_comb, b_num, b_alpha = self._model_selection(X, y)
        _covs = self._construct_cov(X[:, b_comb], y)
        _weights = self._calculate_weights(_covs, b_alpha)
        # predict training scores to get scalers which ensure
        # the q values of validated identifications
        # lower than 0.01
        _scores, _ = self._rda_prediction(X[:, b_comb], _weights)
        _, _, _scaler = self._normalizer(_scores, y)

        # final models
        self.combination_index = b_comb
        self.model_cv = tuple(model_cv)
        self._best_num_ = b_num
        self._best_alpha = b_alpha
        self.stacked_weights = _weights
        self.scaler = _scaler

        return scores, probs, qvals

    def predict(self, X):
        """
        Output predicted scores.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return self._rda_prediction(X[:, self.combination_index],
                                    self.stacked_weights)

    def predict_cv(self, X):
        """
        Output predicted scores obtained from cross validation
        for validation of new identifications, by consensus voting.
        If all scores of an identification fall in q values less than
        0.01 defined in cross validation fit, then the identification
        is confirmed.
        This can guarantee that the error rates (i.e., q value) in
        validated identifications are still less than 0.01.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # do prediction
        scores = np.empty((X.shape[0], len(self.model_cv)))
        for i, _model in enumerate(self.model_cv):
            X_c = X[:, _model.combination]
            _score, _ = self._rda_prediction(X_c, _model.weights)
            _score -= _model.scaler.center
            _score /= _model.scaler.normalizer
            scores[:, i] = _score

        return scores

    def _normalizer(self, scores, y):
        """
        Normalize the scores so that median of decoy scores
        becomes -1.0 and score at q value 0.01 is 0.0
        """
        sort_index = np.argsort(scores)[::-1]

        # get q values
        qvalues = _get_qvalues(scores[sort_index], y[sort_index])

        # if all identifications having q values lower than 0.01
        # return min scores
        if qvalues[-1] <= 0.01:
            m = scores.min()
        else:
            j = bisect.bisect_left(qvalues, 0.01)
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

        return norm_scores, qvalues[_index], Scaler(m, dm)

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
