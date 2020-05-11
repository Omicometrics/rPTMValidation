import operator
from typing import List, Optional, Tuple

import numpy as np

from . import classification


def calculate_score_threshold(
        model: classification.Classifier,
        x_pos: np.ndarray,
        x_neg: np.ndarray,
        q_value: float = 0.01
) -> float:
    """
    Calculates the classification score threshold for the fitted `model`.

    Args:
        model: Classification model.
        x_pos: Positive feature array.
        x_neg: Negative feature array.
        q_value: Required q value.

    Returns:
        Score threshold.

    """
    pos_scores = model.predict(x_pos, use_cv=True)
    neg_scores = model.predict(x_neg, use_cv=True)
    return calculate_score_threshold_from_scores(
        pos_scores, neg_scores, q_value=q_value
    )


def calculate_fdr_threshold(
        scores: List[Tuple[float, int]],
        q_value: float
) -> Optional[float]:
    """
    Calculate FDR threshold.

    """
    nums, fdr = [0, 0], []
    for score, label in scores:
        nums[label] += 1
        if nums[1] > 0 and nums[0] / nums[1] <= q_value:
            fdr.append(score)
    return min(fdr) if fdr else None


def calculate_score_threshold_from_scores(
        pos_scores: np.ndarray,
        neg_scores: np.ndarray,
        q_value: float = 0.01
) -> Optional[float]:
    """
    Calculates the score threshold at the given `q_value`.

    Once the ensemble validation model is constructed, criteria should be set up
    to identify whether new identification can be accepted as validated.
    In rPTMDetermine2, regularized discriminant analysis (Guo Y, et al.
    Biostatistics, 2007, 8(1), 86–100) is used to combine 30 scores from 30
    models in the ensemble with 3 fold cross validation. In each cross
    validation, a score criterion is set at FDR 1%. This means, for the final
    validation, 3 models will be used to calculate scores for the new
    identification. Therefore, as the strength of ensemble machine learning,
    there are several ways to make final decision.
    1. All 3 scores pass the cross validation score criteria. Since each
       criterion is at 1%, this way can make sure final decision does not exceed
       the set FDR level. Note that all scores have been normalized thus the
       criteria at FDR 1% is set to be 0.
    2. Sum 3 scores together, and set another criterion to make the decision.
    This function implements the second approach.

    """
    pos_sum = pos_scores.sum(axis=1)
    neg_sum = neg_scores.sum(axis=1)

    # If the maximum negative score is lower than the minimum positive score,
    # return the minimum normal score
    min_pos_sum = pos_sum.min()
    if neg_sum.max() <= min_pos_sum:
        return min_pos_sum

    # concatenate to make final list for FDR calculation
    neg_sum = neg_sum[neg_sum >= min_pos_sum]
    scores = sorted(
        list(zip(pos_sum, np.ones(pos_sum.shape, dtype=int))) +
        list(zip(neg_sum, np.zeros(neg_sum.shape, dtype=int))),
        key=operator.itemgetter(0),
        reverse=True
    )

    return calculate_fdr_threshold(scores, q_value)


def evaluate_fdr(
        model: classification.Classifier,
        x_pos: np.ndarray,
        x_neg: np.ndarray,
        score_threshold: float,
        use_consensus: bool = True
) -> float:
    """
    Evaluates the FDR of the given data sets after classification using `model`.

    """
    pos_scores = model.predict(x_pos, use_cv=True)
    neg_scores = model.predict(x_neg, use_cv=True)

    count_fn = count_consensus_votes if use_consensus else count_majority_votes
    num_val_pos = count_fn(pos_scores, score_threshold)
    num_val_neg = count_fn(neg_scores, score_threshold)

    return num_val_neg / num_val_pos


def count_consensus_votes(
        x: np.ndarray,
        threshold: float
) -> int:
    """
    Counts the number of rows in `x` which achieve a strict consensus or pass
    the score `threshold`.

    Args:
        x: An array with rows of scores.
        threshold: The score threshold.

    """
    return np.count_nonzero(
        (x >= 0).sum(axis=1).all() | (x.sum(axis=1) >= threshold)
    )


def count_majority_votes(
        x: np.ndarray,
        threshold: float
) -> int:
    """
    Counts the number of rows in `x` which achieve a majority vote or pass
    the score `threshold`.

    Args:
        x: An array with rows of scores.
        threshold: The score threshold.

    """
    # Perform ceiling division to find the number of votes required for a
    # majority
    required_votes = - (- x.shape[1] // 2)
    return np.count_nonzero(
        (x >= 0).sum(axis=1) >= required_votes | (x.sum(axis=1) >= threshold)
    )


def passes_consensus(
        x: np.ndarray,
        threshold: float
) -> bool:
    """
    Determines whether the scores in `x` achieve consensus or pass the score
    `threshold`.

    Args:
        x: An array with a single row of scores.
        threshold: The score threshold.

    """
    return (x >= 0).all() or x.sum() >= threshold


def passes_majority(
        x: np.ndarray,
        threshold: float
) -> bool:
    """
    Determines whether the scores in `x` achieve a majority verdict or pass the
    score `threshold`.

    Args:
        x: An array with a single row of scores.
        threshold: The score threshold.

    """
    required_votes = - (- x.shape[1] // 2)
    return (x >= 0).sum() >= required_votes or x.sum() >= threshold
