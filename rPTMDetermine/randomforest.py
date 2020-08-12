import dataclasses
import numbers
import threading

from warnings import warn
from typing import Dict, List, Optional, Tuple, Callable
from joblib import Parallel, delayed

import numpy as np
from numpy import fft
from scipy.stats import ks_2samp
from scipy.sparse import issparse

from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import _accumulate_prediction
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_random_state
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier


MAX_INT = np.iinfo(np.int32).max
# Number of bins for x discretization for kernel density estimation.
Nbins = 1000


@dataclasses.dataclass()
class Node:
    feature_idx: Optional[int] = None
    class_sizes: Optional[List[int]] = None
    used: bool = True
    log_ratios: Optional[np.ndarray] = None
    bin_edges: Optional[np.ndarray] = None
    kde_evals: Optional[List[np.ndarray]] = None
    gofs: Optional[List[Tuple[float, float]]] = None
    determine: Optional[Callable] = None
    root_feature_idx: Optional[int] = None


def _kernel_density_fft(x: np.ndarray, bins: np.ndarray):
    """
    Kernel density estimation using fast fourier transformation.
    Args:
        x: x
        bins: bins for x

    Notes:
        Only Gaussian kernel is implemented.

    References:
        [1] Gramacki A. FFT-Based Algorithms for Kernel Density
            Estimation and Bandwidth Selection. In Nonparametric
            Kernel Density Estimation and Its Computational Aspects.
            Springer. 2018, 85-118.
        [2] Silverman BW. Kernel Density Estimation Using the Fast
            Fourier Transform. J Royal Stat Soc. C. 1982, 31, 93-99.
        [3] Wand MP, Jones MC. Kernel Smoothing. Chapman & Hall, New
            York. 1995.
    """
    n = x.size
    m = Nbins
    # scott bandwidth: 1.059 * A * nobs ** (-1/5.),
    # where A is min(std(X), IQR/1.34)
    iqr = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349
    bw = 1.059 * np.minimum(iqr, x.std(ddof=1)) * (n ** (-0.2))

    # prepare constants
    counts, _ = np.histogram(x, bins=bins)
    grid_size = int(2 ** np.ceil(np.log2(3 * m - 1)))
    delta = np.diff(bins).mean()
    # gaussian kernel
    kai = (np.exp(-(np.arange(m) * (delta / bw)) ** 2 / 2)
           / (np.sqrt(2 * np.pi) * bw * n))

    # counts vector
    vector_counts = np.zeros(grid_size)
    vector_counts[m:m + counts.size] = counts
    # kernel function vector
    vector_kernel = np.zeros(grid_size)
    vector_kernel[:m - 1] = kai[1:][::-1]
    vector_kernel[m - 1: 2 * m - 1] = kai

    # fast fourier transform
    c = fft.fft(vector_counts)
    k = fft.fft(vector_kernel)
    f = fft.ifft(c * k).real[2 * m - 1: 3 * m - 1]

    return f


def _tree_fit_kdes(x, y, bin_edges, tree_clf,
                   eval_gof=False, record_root=False):
    """
    For the given `tree`, fits and evaluates Gaussian KDEs at each
    leaf parent node.

    """
    nodes = {}
    node_features = tree_clf.tree_.feature
    node_indicator = tree_clf.decision_path(x).A.astype(bool)
    thresholds = tree_clf.tree_.threshold

    y = y.ravel()
    for leaf_idx, leaf_feature in enumerate(node_features):
        if leaf_feature >= 0:
            # Not a leaf
            continue
        parent_node = leaf_idx - 1
        feature = node_features[parent_node]
        if feature < 0:
            parent_node = leaf_idx - 2
            feature = node_features[parent_node]

        # if the parent node has been used
        if parent_node in nodes:
            continue

        sample_indices = node_indicator[:, parent_node]
        pos_data = x[sample_indices & (y == 1), feature]
        neg_data = x[sample_indices & (y == 0), feature]

        node = Node(feature_idx=feature,
                    class_sizes=[neg_data.size, pos_data.size])

        if neg_data.size >= 100 and pos_data.size >= 100:
            # Bin sample data and evaluate using both KDEs
            feature_bins = bin_edges[feature]
            fitted_densities = []
            for data in [neg_data, pos_data]:
                data_densities = _kernel_density_fft(data, feature_bins)
                if np.isnan(data_densities).all():
                    node.used = False
                    break
                # fitted densities with upper and lower limit be 1e-10 and 1e10
                fitted_densities.append(np.clip(data_densities, 1e-10, 1e10))
                # goodness-of-fit
                if eval_gof:
                    gof = ks_2samp(data, data_densities)
                    node.gofs.append(gof)

            # natural log of negative densities to positive densities
            if node.used:
                node.bin_edges = feature_bins
                node.log_ratios = \
                    np.log(fitted_densities[1]) - np.log(fitted_densities[0])
                node.root_feature_idx = node_features[0]

                # record the path from root to next level parent nodes
                if record_root:
                    node.root_feature_idx = node_features[0]
                    if pos_data[0] <= thresholds[0]:
                        node.determine = lambda a: a <= thresholds[0]
                    else:
                        node.determine = lambda a: a > thresholds[0]
        else:
            node.used = False

        nodes[parent_node] = node

    return nodes


def _simple_tree_decide_kde(x, tree_clf, nodes: Dict[int, Node], bin_indices):
    """ Simplifies tree parent nodes access if depth==2 """
    scores = np.zeros(x.shape[0])
    for parent_node, node in nodes.items():
        if node.used:
            sample_indices = node.determine(x[:, node.root_feature_idx])
            if sample_indices.any():
                bin_index = bin_indices[sample_indices, node.feature_idx]
                scores[sample_indices] = node.log_ratios[bin_index]
    return scores


def _tree_decide_kde(x, tree_clf, nodes: Dict[int, Node], bin_indices):

    node_indicator = tree_clf.decision_path(x).A.astype(bool)
    
    scores = np.zeros(x.shape[0])
    for parent_node, node in nodes.items():
        if node.used:
            sample_indices = node_indicator[:, parent_node]
            if sample_indices.any():
                bin_index = bin_indices[sample_indices, node.feature_idx]
                scores[sample_indices] = node.log_ratios[bin_index]

    return scores


def _get_n_samples_subsampling(n_samples: int, max_samples: Optional[float]):
    """
    Get the number of samples in a subsampled sample.
    Args:
        n_samples: Number of samples in minor class.
        max_samples: The maximum number of samples to draw
                     from the total available:
                     - if float, this indicates a fraction of the total
                       and should be the interval `(0, 1)`;
                     - if int, this indicates the exact number of samples;
                     - if None, this indicates the total number of samples.

    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = ("`max_samples` must be in range 1 to number of minor"
                   " class {} but got value {}")
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _random_subsampling(y, max_samples, random_states, ntimes):
    """ Random subsampling without replacement """
    # to make the results reproducible, use fixed seeds
    if random_states is None:
        seed = 1
        random_states = [seed]
    else:
        seed = len(random_states) + 1
        random_states.append(seed)

    if y.ndim == 2:
        # reshape the column array back
        y = y[:, 0]

    rng = np.random.default_rng(seed)

    # group y and get number of samples in subsampling
    groups, group_num = np.unique(y, return_counts=True)

    nsub = _get_n_samples_subsampling(group_num.min(), max_samples)

    # random subsampling
    random_sample_index = np.empty((ntimes, nsub * group_num.size), dtype=int)
    for i, (g, n) in enumerate(zip(groups, group_num)):
        index, = np.where(y == g)
        start, end = nsub * i, nsub * (i+1)
        for j in range(ntimes):
            random_index = rng.choice(n, size=nsub, replace=False)
            random_sample_index[j, start:end] = index[random_index]

    return random_sample_index, random_states


def _parallel_build_trees(tree, x, y, tree_idx, n_trees, verbose=0):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    tree.fit(x, y, check_input=False)

    return tree


class RandomForest(RandomForestClassifier):
    def __init__(
            self,
            n_estimators=100,
            *,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            max_density_samples=1000,
            eval_gof=False
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )

        self.max_density_samples = max_density_samples
        self.eval_gof = eval_gof
        self.subsample_random_states = None
        self.classes_ = []
        self.n_classes_ = []
        self.bin_edges = None

        if not hasattr(self, "estimators_"):
            self.estimators_ = []
            self._nodes: List[Dict[int, Node]] = []

    def fit(self, x, y):
        """
        This overrides sklearn's random forest fit method by replacing
        the bootstrap sampling to random subsampling with equal sample
        size for all groups.
        So, comparing to original parameters, the following methods:
            sample_weight
            _validate_y_class_weight
            _get_n_samples_bootstrap
        and attribute bootstrap are not used.

        """
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        x, y = self._validate_data(
            x, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )

        if issparse(x):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            x.sort_indices()

        # Remap output
        self.n_features_ = x.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array"
                 " was expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        for k in range(self.n_outputs_):
            classes_k = np.unique(y[:, k])
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start:
            # Free allocated memory, if any
            self.estimators_ = []
            self._nodes = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # do subsampling for n_more_estimators times
            random_sample_index, self.subsample_random_states =\
                _random_subsampling(
                    y, self.max_samples, self.subsample_random_states,
                    n_more_estimators
                )

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(
                    t, x[index], y[index], i, len(trees), verbose=self.verbose
                )
                for i, (t, index) in enumerate(zip(trees, random_sample_index))
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

            # do kernel density estimation to parent nodes
            if self.bin_edges is None:
                self._bin_x(x)

            nodes = Parallel(n_jobs=self.n_jobs, max_nbytes=1e6)(
                delayed(_tree_fit_kdes)(
                    x[index], y[index], self.bin_edges, e,
                    self.eval_gof, record_root=self.max_depth == 2
                ) for e, index in zip(trees, random_sample_index)
            )
            self._nodes.extend(nodes)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def decision_function(self, x, return_matrix=False):
        """ Decision function. """
        # bin the data in advance
        bin_indices = self._bin_index(x)

        if self.max_depth == 2:
            score_matrix = Parallel(
                n_jobs=self.n_jobs, max_nbytes=1e6, backend='multiprocessing'
            )(
                delayed(_simple_tree_decide_kde)(
                    x, e, self._nodes[i], bin_indices
                ) for i, e in enumerate(self.estimators_)
            )
        else:
            score_matrix = Parallel(
                n_jobs=self.n_jobs, max_nbytes=1e6, backend='multiprocessing'
            )(
                delayed(_tree_decide_kde)(x, e, self._nodes[i], bin_indices)
                for i, e in enumerate(self.estimators_)
            )

        score_matrix = np.array(score_matrix).T
        if return_matrix:
            return score_matrix.mean(axis=1), score_matrix
        return score_matrix.mean(axis=1)

    def predict(self, x):
        """
        Overrides sklearn's random forest predict
        """
        proba = self.predict_proba(x)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_),
                                   dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)
            return predictions

    def predict_proba(self, x):
        """
        Overrides sklearn's random forest predict_proba
        """
        check_is_fitted(self)
        # Check data
        x = self._validate_X_predict(x)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((x.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict_proba, x, all_proba,
                                            lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_log_proba(self, X):
        """
        Overrides sklearn's random forest predict_log_proba
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

    def _bin_x(self, x):
        """ Bin X """
        nbins = Nbins
        bin_edges = []
        for i in range(x.shape[1]):
            bin_edges.append(np.histogram_bin_edges(x[:, i], bins=nbins))
        self.bin_edges = bin_edges

    def _bin_index(self, x):
        """ Bin index for the variable X """
        bin_indices = np.empty(x.shape, dtype=int)
        for i, edges in enumerate(self.bin_edges):
            index = np.clip(
                np.searchsorted(edges, x[:, i]) - 1,
                0, edges.shape[0] - 2
            )
            bin_indices[:, i] = index
        return bin_indices
