import dataclasses
from typing import Dict, List, Optional, Tuple

from joblib import Parallel, delayed
import numba
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from statsmodels.sandbox.nonparametric.kernels import Gaussian
from statsmodels.nonparametric.kde import KDEUnivariate, kernel_switch


@dataclasses.dataclass()
class Node:
    feature_idx: int
    class_sizes: np.ndarray
    used: bool
    bin_edges: Optional[np.ndarray]
    kde_evals: List[np.ndarray]
    gofs: List[Tuple[float, float]]


def iqr(arr):
    return np.subtract(*np.percentile(arr, (75, 25)))


@numba.njit(fastmath=True, parallel=True)
def gau(x):
    return 0.3989422804014327 * np.exp(-x * x / 2.0)


class CustomGaussianKernel(Gaussian):
    """
    Gaussian (Normal) Kernel. This is subclassed here and the pointer to the
    'gau' kernel class swapped to this class in order to provide a more
    efficient implementation of the lambda shape function. This achieves
    ~ 2x speed-up versus the numpy version used in the standard implementation.

    """
    def __init__(self, h=1.0):
        super().__init__(h=h)
        self._shape = gau


def _gaussian_kde(
        data, random_state, max_samples, bw_method='normal_reference'
):
    """
    Fits a Gaussian KDE to `data`.

    """
    # This overrides the 'gau' entry of the kernel_switch dictionary
    # defined in statsmodels in order to use our accelerated implementation
    kernel_switch['gau'] = CustomGaussianKernel
    np.random.seed(random_state)
    sample = np.random.choice(
        data, size=min(max_samples, data.size), replace=False
    )
    try:
        kde = KDEUnivariate(sample)
        kde.fit(bw=bw_method)
    except RuntimeError:
        raise ValueError(
            'Failed to establish bandwidth due to improper sample distribution.'
        )
    return kde


def _tree_fit_kdes(
        X, y, tree, sample_leaves_parents, max_kde_samples, random_state=None
):
    """
    For the given `tree`, fits and evaluates Gaussian KDEs at each leaf parent
    node.

    """
    nodes = {}
    for leaf_idx, leaf_feature in enumerate(tree.feature):
        if leaf_feature >= 0:
            # Not a leaf
            continue
        parent_idx = leaf_idx - 1
        feature = tree.feature[parent_idx]
        if feature < 0:
            parent_idx = leaf_idx - 2
            feature = tree.feature[parent_idx]
        sample_indices = sample_leaves_parents == parent_idx
        pos_data = X[sample_indices & (y == 1), feature]
        neg_data = X[sample_indices & (y == 0), feature]

        node = Node(
            feature, np.array([neg_data.shape[0], pos_data.shape[0]]),
            True, None, [], []
        )

        if neg_data.size >= 100 and pos_data.size >= 100:
            # Bin sample data and evaluate using both KDEs
            bin_edges = np.histogram_bin_edges(
                X[:, feature], bins=1000
            )
            node.bin_edges = bin_edges
            bin_edge_diff = np.diff(bin_edges, axis=0)
            for data in [neg_data, pos_data]:
                try:
                    kde = _gaussian_kde(data, random_state, max_kde_samples)
                except ValueError:
                    node.used = False
                    break
                fitted_densities = \
                    kde.evaluate(bin_edges[1:] + bin_edge_diff / 2)
                gof = ks_2samp(data, fitted_densities)
                node.kde_evals.append(fitted_densities)
                node.gofs.append(gof)
        else:
            node.used = False

        nodes[parent_idx] = node

    return nodes


def _tree_decide_kde(X, sample_leaves, nodes: Dict[int, Node]):
    scores = np.zeros(X.shape[0])
    for parent_idx, node in nodes.items():
        sample_indices = sample_leaves == parent_idx
        if not sample_indices.any():
            continue

        if not node.used:
            continue

        sample_data = X[sample_indices, node.feature_idx]

        bin_indices = np.clip(
            np.searchsorted(node.bin_edges, sample_data) - 2,
            0,
            node.bin_edges.shape[0] - 1
        )

        total_size = node.class_sizes.sum()
        pos = (node.class_sizes[1] / total_size) * \
            node.kde_evals[1][bin_indices]
        neg = (node.class_sizes[0] / total_size) * \
            node.kde_evals[0][bin_indices]

        # Clip the score ranges to ensure that the log transform is
        # valid
        pos = np.clip(pos, None, 1e100)
        neg = np.clip(neg, 1e-100, None)

        scores[sample_indices] = np.log10(1 + pos / neg)

    return scores


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
            max_density_samples=1000
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

        self._nodes: List[Dict[int, Node]] = []

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)

        indicators, index_by_tree = self.decision_path(X)
        indices = zip(index_by_tree, index_by_tree[1:])

        self._nodes = Parallel(
            n_jobs=self.n_jobs, max_nbytes=1e6, backend='multiprocessing'
        )(
            delayed(_tree_fit_kdes)(
                X,
                y,
                tree_clf.tree_,
                # Keep only the leaf parent node index
                self._leaf_parent_indices(indicators, begin, end),
                self.max_density_samples,
                self.random_state
            ) for tree_clf, (begin, end) in zip(self.estimators_, indices)
        )

        return self

    def decision_function(self, X):
        indicators, index_by_tree = self.decision_path(X)
        indices = zip(index_by_tree, index_by_tree[1:])

        return sum(Parallel(
            n_jobs=self.n_jobs, max_nbytes=1e6, backend='multiprocessing'
        )(
            delayed(_tree_decide_kde)(
                X,
                self._leaf_parent_indices(indicators, begin, end),
                self._nodes[tree_idx]
            ) for tree_idx, (begin, end) in enumerate(indices)
        )) / self.n_estimators

    def _leaf_parent_indices(self, indicators, begin, end):
        return indicators[:, begin:end].indices[
            self.max_depth - 1::self.max_depth + 1
        ]
