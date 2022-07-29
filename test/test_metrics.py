#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test the custom metrics."""

import tensorflow as tf

from bert_coref import IgnorantBinaryMCC, IgnorantPrecision


class MetricTest(tf.test.TestCase):
    """A test to verify that the custom metrics are correct."""

    def setUp(self) -> None:
        """Fixture setup. Create the following tensors:
            1. An `y_true` tensor of shape `(batch_size, sequence_length)`.
            2. An `y_pred` tensor of shape `(batch_size, sequence_length)`.
        """
        super(MetricTest, self).setUp()
        self._y_true = tf.convert_to_tensor([
            [1, 1],
            [1, -1],
            [0, -1]
        ])
        self._y_pred = tf.convert_to_tensor([
            [0.9, -0.42],
            [0.85, -0.3],
            [0.7, -0.75]
        ])

    def test_ignorant_binary_mcc(self) -> None:
        """Test if the implementation of the binary MCC with an
        ignored label is correct.
        """
        mcc = IgnorantBinaryMCC(ignore_label=-1)
        # noinspection PyCallingNonCallable
        result = mcc(self._y_true, self._y_pred)
        expected_result = -0.33333334
        self.assertEqual(expected_result, result)

    def test_ignorant_precision(self) -> None:
        """Test if the implementation of the precision with an
        ignored label is correct.
        """
        precision = IgnorantPrecision(ignore_label=-1)
        # noinspection PyCallingNonCallable
        result = precision(self._y_true, self._y_pred)
        expected_result = 0.66666667
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    tf.test.main()
