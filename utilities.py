#! /usr/bin/env python3
"""
A script providing utility functions for peptide modification validation.

"""
from bisect import bisect_left
import math
from typing import List, Tuple


def slice_list(values, nslices=800) -> Tuple[List[int], List[float]]:
    """
    Slices the list of values into nslices segments.

    Args:
        values (list): A list of floats.
        nslices (int, optional): The number of slices to split.

    Returns:
        tuple: index at which each slice begins in values,
               the value at which each slice begins.

    """
    size = (values[-1] - values[0]) / nslices
    # bounds contains the lower bound of each slice
    idxs, bounds = [], []
    for ii in range(nslices + 1):
        pos = bisect_left(values, size * ii + values[0])
        if pos == 0:
            idxs.append(0)
            bounds.append(values[0])
        elif pos < len(values):
            idxs.append(pos - 1)
            bounds.append(values[pos - 1])
        else:
            idxs.append(pos - 1)
            bounds.append(values[-1])

    return idxs, bounds


def longest_sequence(seq: List[int]) -> Tuple[int, List[int]]:
    """
    Finds the length (and subsequence) of the longest consecutive sequence
    of integers in a sorted list. This algorithm is O(n) in compexity.

    Args:
        seq (list): A sorted list of integers.

    Returns:
        A tuple of maximum consecutive sequence length and the subsequence.

    """
    if not seq:
        return 0, None

    max_len, num_lens, max_num = -1, {}, None
    for num in seq:
        num_len = num_lens[num] = num_lens.get(num - 1, 0) + 1
        if num_len > max_len:
            max_len, max_num = num_len, num
    return max_len, list(range(max_num - max_len * 1 + 1, max_num + 1, 1))


def sort_lists(key: int, *args):
    """
    Sorts the lists given in *args using the list indicated by key.

    Args:
        key (int): The index in *args of the list to be used for ordering.
        *args: The lists to sort by a common list.

    Returns:
        tuple of ordered lists.

    """
    return zip(*sorted(zip(*args), key=lambda t: t[key]))


def log_binom_prob(k: int, n: int, p: float) -> float:
    """
    Calculates the negative base-10 log of the cumulative binomial
    probability.

    Args:
        k (int): The number of successes.
        n (int): The number of trials.
        p (float): The probability of success.

    Returns:
        The negative base-10 log of the binomial CDF.

    """
    logn = [math.log10(i + 1) for i in range(n)]
    nk = n - k
    s = sum(logn)

    pbt, pbf = math.log10(p), math.log10(1. - p)
    # the initial binomial
    pbk = []
    s1, s2 = sum(logn[:k]), sum(logn[:nk])
    pbk.append(s - s1 - s2 + k * pbt + nk * pbf)
    # calculate the cumulative using recursive iteration
    for i in range(k, n):
        s1 += logn[i]
        s2 -= logn[nk - 1]
        nk -= 1
        pbk.append(s - s1 - s2 + (i + 1) * pbt + nk * pbf)
    m = min(pbk)

    # to avoid OverflowError
    try:
        return - m - math.log10(sum(10 ** (x - m) for x in pbk))
    except OverflowError:
        pbk2 = [x - m for x in pbk]
        m2 = max(pbk2)
        return - m - m2 - math.log10(sum(10 ** (x - m2) for x in pbk2))
