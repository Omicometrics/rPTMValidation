#! /usr/bin/env python3
"""
A script providing utility functions for peptide modification validation.

"""
from bisect import bisect_left
import collections
import operator
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
                    TypeVar)


T = TypeVar('T')

Slices = collections.namedtuple("Slices", ["idxs", "bounds"])


def slice_list(values: List[float], nslices: int = 800) -> Slices:
    """
    Slices the list of values into nslices segments.

    Args:
        values (list): A list of floats.
        nslices (int, optional): The number of slices to split.

    Returns:
        Slices: index at which each slice begins in values,
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

    return Slices(idxs, bounds)


def longest_sequence(seq: Sequence[int]) -> Tuple[int, Optional[List[int]]]:
    """
    Finds the length (and subsequence) of the longest consecutive sequence
    of integers in a sorted list. This algorithm is O(n) in complexity.

    Args:
        seq: A sorted list of integers.

    Returns:
        A tuple of maximum consecutive sequence length and the subsequence.

    """
    if not seq:
        return 0, None

    max_len = -1
    num_lens: Dict[int, int] = {}
    max_num = -1
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
        tuple of ordered tuples.

    """
    return zip(*sorted(zip(*args), key=operator.itemgetter(key)))


def deduplicate(items: Iterable[T]) -> List[T]:
    """
    Deduplicates a list, retaining the order of the elements.

    Args:
        items (list): The items to deduplicate.

    Returns:
        The deduplicated list of items, retaining the order.

    """
    seen: Set[Any] = set()
    seen_add = seen.add
    return [x for x in items if not (x in seen or seen_add(x))]
