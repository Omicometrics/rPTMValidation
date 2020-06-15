import unittest

import numpy as np

from rPTMDetermine.machinelearning.scoring import (
    count_consensus_votes,
    count_majority_votes,
    passes_consensus,
    passes_majority
)


class TestConsensusScoring(unittest.TestCase):
    def test_passes_consensus(self):
        """
        Tests that a set of all-positive scores passes consensus voting.

        """
        scores = np.array([0.1, 0.2, 0.3])
        self.assertEqual(True, passes_consensus(scores))

    def test_fails_consensus(self):
        """
        Tests that negative scores, below the threshold, result in failing
        the consensus vote.

        """
        scores = np.array([-0.4, 0.1, 0.1])
        self.assertEqual(False, passes_consensus(scores))

    def test_count_votes(self):
        """
        Tests that consensus votes are correctly counted.

        """
        scores = np.array([
            [0.1, 0.1, 0.2],  # pass
            [-0.2, -0.1, 0.],  # fail
            [-0.2, 0.1, 0.2],  # fail
        ])
        self.assertEqual(1, count_consensus_votes(scores))


class TestMajorityScoring(unittest.TestCase):
    def test_passes_majority_with_consensus(self):
        """
        Tests that a set of all-positive scores passes majority voting.

        """
        scores = np.array([0.1, 0.2, 0.3])
        self.assertEqual(True, passes_majority(scores))

    def test_passes_majority(self):
        """
        Tests that a set of all-positive scores passes majority voting.

        """
        scores = np.array([-0.1, 0.2, 0.3])
        self.assertEqual(True, passes_majority(scores))

    def test_fails_majority(self):
        """
        Tests that negative scores, below the threshold, result in failing
        the majority vote.

        """
        scores = np.array([-0.4, -0.1, 0.1])
        self.assertEqual(False, passes_consensus(scores))

    def test_count_votes(self):
        """
        Tests that majority votes are correctly counted.

        """
        scores = np.array([
            [0.1, 0.2, 0.3],  # pass
            [-0.2, -0.1, 0.],  # fail
            [-0.2, 0.1, 0.2],  # pass
            [-0.4, 0.5, 0.6],  # pass
        ])
        self.assertEqual(3, count_majority_votes(scores))


if __name__ == '__main__':
    unittest.main()
