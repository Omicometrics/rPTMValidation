import dataclasses
from typing import Dict, List, Optional

import numba
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from statsmodels.sandbox.nonparametric.kernels import Gaussian
from statsmodels.nonparametric.kde import KDEUnivariate, kernel_switch


@dataclasses.dataclass()
class Leaf:
    feature_idx: int
    class_sizes: np.ndarray
    bin_edges: Optional[np.ndarray]
    kde_evals: Optional[List[np.ndarray]]


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

        self._leaves: List[Dict[int, Leaf]] = []
        # TODO: generalize this to any tree depth
        # TODO: check that this is the correct tree node indexing
        #      1
        #   2     5
        #  3 4   6 7
        #  (based on inspection of tree.feature array)
        self._leaf_parents = {2: 1, 3: 1, 5: 4, 6: 4}

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)

        # This overrides the 'gau' entry of the kernel_switch dictionary
        # defined in statsmodels in order to use our accelerated implementation
        kernel_switch['gau'] = CustomGaussianKernel

        indicators, index_by_tree = self.decision_path(X)

        indices = zip(index_by_tree, index_by_tree[1:])
        for tree_classifier, (begin, end) in zip(self.estimators_, indices):
            tree = tree_classifier.tree_
            sample_leaves = indicators[:, begin:end].indices[2::3]

            leaves = {}
            for leaf_idx, parent_idx in self._leaf_parents.items():
                sample_indices = sample_leaves == leaf_idx
                feature = tree.feature[parent_idx]
                pos_data = X[sample_indices & (y == 1), feature]
                neg_data = X[sample_indices & (y == 0), feature]

                if pos_data.size == 0 or neg_data.size == 0:
                    bin_edges, evals = None, None
                else:
                    kdes = [
                        self._gaussian_kde(data)
                        for data in [neg_data, pos_data]
                    ]

                    # Bin sample data and evaluate using both KDEs
                    bin_edges = np.histogram_bin_edges(
                        X[sample_indices, feature], bins=1000
                    )
                    evals = [
                        kde.evaluate(bin_edges) for kde in kdes
                    ]

                leaves[leaf_idx] = Leaf(
                    feature,
                    np.array([neg_data.shape[0], pos_data.shape[0]]),
                    bin_edges,
                    evals
                )

            self._leaves.append(leaves)

        return self

    def decision_function(self, X):
        scores = np.zeros(X.shape[0])

        indicators, index_by_tree = self.decision_path(X)
        indices = zip(index_by_tree, index_by_tree[1:])
        for tree_idx, (begin, end) in enumerate(indices):
            sample_leaves = indicators[:, begin:end].indices[
                self.max_depth::self.max_depth + 1
            ]
            for leaf_idx in self._leaf_parents:
                sample_indices = sample_leaves == leaf_idx
                if not sample_indices.any():
                    continue
                leaf = self._leaves[tree_idx][leaf_idx]

                if leaf.bin_edges is None:
                    continue

                sample_data = X[sample_indices, leaf.feature_idx]

                bin_indices = np.clip(
                    np.searchsorted(leaf.bin_edges, sample_data) - 1,
                    0,
                    leaf.bin_edges.shape[0] - 1
                )

                total_size = leaf.class_sizes.sum()
                pos = (leaf.class_sizes[1] / total_size) *\
                    leaf.kde_evals[1][bin_indices]
                neg = (leaf.class_sizes[0] / total_size) *\
                    leaf.kde_evals[0][bin_indices]

                scores[sample_indices] += np.log10(2 - (neg / (pos + neg)))

        return scores

    def _gaussian_kde(self, data, bw_method=0.2):
        kde = KDEUnivariate(
            np.random.choice(
                data,
                size=max(self.max_density_samples, data.shape[0])
            )
        )
        kde.fit(bw=bw_method)
        return kde
