# -*- coding: utf-8 -*-

"""A module for testing clustering metrics."""

import unittest

import numpy as np

from bert_coref.clustering import compute_purity


class ClusteringMetricTest(unittest.TestCase):
    """A test case for clustering metrics."""

    def setUp(self) -> None:
        """Fixture setup: create a cluster and a class label array."""
        self._clusters = np.array([1, 2, -1, -1, 2, 2, 1])
        self._labels = np.array([3, 4, 4, 4, 4, 3, 3])

    def test_compute_purity(self) -> None:
        """Test the purity implementation."""
        expected_result = 0.8
        result = compute_purity(self._clusters, self._labels)
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
