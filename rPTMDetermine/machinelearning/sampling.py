import numpy as np


def subsample_negative(
        x_pos: np.ndarray,
        x_neg: np.ndarray
) -> np.ndarray:
    """
    Subsamples negative class for model construction.

    To remove noise and select representative samples for model construction,
    negatives that are nearest to positives (mean distances to 3 positives) are
    selected.

    Args:
        x_pos: Positive class feature matrix.
        x_neg: Negative class feature matrix.

    Return:
        Closest distances of negatives to positives.

    References:
    [1] Kubat M, et al. Addressing the Curse of Imbalanced Training
        Sets: One-Sided Selection. In Proceedings of the Fourteenth
        International Conference on Machine Learning. 1997.

    """
    # concatenate data matrix
    min_x = np.minimum(x_pos.min(axis=0), x_neg.min(axis=0))
    max_x = np.maximum(x_pos.max(axis=0), x_neg.max(axis=0))

    # min-max standardization
    x_decoy_std = (x_neg - min_x) / (max_x - min_x)
    x_pos_std = (x_pos - min_x) / (max_x - min_x)

    # parameters
    n_decoy = x_neg.shape[0]
    n_pos = x_pos_std.shape[0]
    dists = np.empty(n_decoy)

    # normalize data matrix to unit length
    x_decoy_std /= \
        np.sqrt((x_decoy_std * x_decoy_std).sum(axis=1))[:, np.newaxis]
    x_pos_std /= np.sqrt((x_pos_std ** 2).sum(axis=1))[:, np.newaxis]

    # calculate distances
    n_block = 300
    block_size = int(n_decoy / n_block) + 1
    for i in range(n_block):
        idx = np.arange(
            block_size * i, min(block_size * (i+1), n_decoy), dtype=int
        )
        dist = np.dot(x_decoy_std[idx], x_pos_std.T)
        dists[idx] = np.partition(dist, n_pos - 3)[:, n_pos - 3:].mean(axis=1)

    return dists
