import unittest

from rPTMDetermine.utilities import (
    deduplicate,
    longest_sequence,
    sort_lists
)


class TestDeduplicate(unittest.TestCase):
    def test_no_duplicates(self):
        """
        Tests that the input is unchanged when no duplicates are present.

        """
        entries = [1, 2, 3, 4, 5]
        self.assertEqual(
            entries,
            deduplicate(entries)
        )

    def test_duplicates_removed(self):
        """
        Tests that when duplicates are removed, order of the elements is
        retained.

        """
        entries = [1, 1, 2, 3, 3, 3, 4, 5, 6, 6]
        self.assertEqual(
            [1, 2, 3, 4, 5, 6],
            deduplicate(entries)
        )


class TestSortLists(unittest.TestCase):
    def test_sort_two(self):
        """
        Tests that both lists are sorted relative to order in the second of two
        lists.

        """
        l1 = list('abcde')
        l2 = [2, 1, 4, 3, 5]
        self.assertEqual(
            (tuple('badce'), (1, 2, 3, 4, 5)),
            tuple(sort_lists(1, l1, l2))
        )

    def test_sort_three(self):
        """
        Tests that all lists are sorted relative to order in the second of three
        lists.

        """
        l1 = list('abcde')
        l2 = [2, 1, 4, 3, 5]
        l3 = [53., 47., 9., -0.1, 27.1]
        self.assertEqual(
            (tuple('badce'), (1, 2, 3, 4, 5), (47., 53., -0.1, 9., 27.1)),
            tuple(sort_lists(1, l1, l2, l3))
        )


class TestLongestSequence(unittest.TestCase):
    def test_longest_detected_end(self):
        seq = [1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15, 16]
        self.assertEqual(
            (5, [12, 13, 14, 15, 16]),
            longest_sequence(seq)
        )

    def test_longest_detected_middle(self):
        seq = [1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15]
        self.assertEqual(
            (5, [5, 6, 7, 8, 9]),
            longest_sequence(seq)
        )


if __name__ == '__main__':
    unittest.main()
