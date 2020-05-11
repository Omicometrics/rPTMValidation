from typing import List, Optional

import numpy as np


def subsample_negative(
        x_pos: np.ndarray,
        x_neg: np.ndarray,
        retain_fraction: float = 0.2
) -> np.ndarray:
    """
    Sub-samples negative identifications to remove those matches of lowest
    quality and bring the number of identifications closer to the positive
    set.

    """
    x = np.concatenate((x_pos, x_neg), axis=0)
    min_x = x.min(axis=0)
    max_x = x.max(axis=0)
    range_x = max_x - min_x

    # min-max standardization
    x_decoy_std = (x_neg - min_x) / range_x
    x_pos_std = (x_pos - min_x) / range_x

    n_decoy = x_neg.shape[0]

    # pre-allocate for distances between decoys and positive features
    dists = np.empty(n_decoy)

    # normalizer
    norm_decoy = np.sqrt((x_decoy_std * x_decoy_std).sum(axis=1))
    norm_pos = np.sqrt((x_pos_std * x_pos_std).sum(axis=1))

    # calculate distances using dot product to avoid memory problem due to
    # very large size of distance matrix constructed, separate them into
    # 300 blocks
    block_size = int(n_decoy / 300) + 1
    for i in range(300):
        _ix = np.arange(
            block_size * i,
            min(block_size * (i + 1), n_decoy),
            dtype=int
        )
        # calculate distance using dot product
        dist_curr = \
            np.dot(x_decoy_std[_ix], x_pos_std.T) / \
            (norm_decoy[_ix][:, np.newaxis] * norm_pos)

        dist_curr.sort(axis=1)
        # for each decoy, get the closest 3 distances to positives,
        # and calculate their mean.
        dists[_ix] = dist_curr[:, -3:].mean(axis=1)

    sorted_ix = np.argsort(dists)[::-1]
    return x_neg[sorted_ix[:int(n_decoy * retain_fraction)]]
