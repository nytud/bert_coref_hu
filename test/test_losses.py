#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test the loss functions."""


import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

from bert_coref import ClusterTripletLoss, IgnorantBinaryCrossentropy


class LossTest(tf.test.TestCase):
    """A test case to verify that the training objectives
    are defined correctly.
    """

    def setUp(self) -> None:
        """Fixture setup. It defines the following tensors:

        1. An `y_pred` tensor of shape
            `(batch_size, sequence_length, hidden_size)`.
        2. An `y_true` tensor of shape `(batch_size, sequence_length)`.
        3. An `y_pred_binary` tensor of shape `(batch_size, sequence_length)`
            (for binary classification).
        4. An `y_true_binary` tensor of shape `(batch_size, sequence_length)`
            (for binary classification).
        """
        super(LossTest, self).setUp()
        self._y_pred = tf.convert_to_tensor([
            [
                [1.2, 0.9, 1.2, 1.],
                [0., 0., 0., 0.],
                [-1., -1., -1., -1.],
                [1., 0., 1., 0.5]
            ],
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.1, 0.1, 0.1, 0.1],
                [-1., -1., -1., -1.],
                [0.1, 0.1, 1., 0.1]
            ]
        ])
        self._y_true = tf.convert_to_tensor([
            [0, 2, 0, 1],
            [1, 2, 0, 1]
        ])
        self._y_pred_binary = tf.convert_to_tensor([
            [
                [0.2, -1.3, 1.4, -0.7],
                [-0.2, -1.4, 1.3, 0.7]
            ],
            [
                [0.45, 1.23, 1.1, -0.9],
                [-1.23, -0.45, -1.1, 0.9]
            ]
        ])
        self._y_true_binary = tf.convert_to_tensor([
            [
                [1, 0, -1, 0],
                [1, 0, 1, 1]
            ],
            [
                [1, -1, 1, 0],
                [0, 0, 1, -1]
            ]
        ])

    def test_cluster_triplet_loss(self) -> None:
        """Test the cluster-based triplet loss."""
        expected_loss_values = (0., 0.075)
        loss_objective = ClusterTripletLoss(margin=0.5)
        # noinspection PyCallingNonCallable
        loss_value_1 = loss_objective(self._y_true, self._y_pred)
        loss_objective.margin = 1.
        # noinspection PyCallingNonCallable
        loss_value_2 = loss_objective(self._y_true, self._y_pred)
        self.assertAllClose(
            expected_loss_values, (loss_value_1, loss_value_2))

    def test_ignorant_binary_crossentropy(self) -> None:
        """Test the binary crossentropy loss with an ignored label."""
        ignore_label = -1
        ignorant_bce = IgnorantBinaryCrossentropy(ignore_label=ignore_label)
        simple_bce = BinaryCrossentropy(from_logits=True)
        # noinspection PyCallingNonCallable
        ignorant_loss = ignorant_bce(self._y_true_binary, self._y_pred_binary)
        small_loss_labels = tf.where(
            self._y_true_binary == ignore_label, 1, self._y_true_binary)
        large_loss_labels = tf.where(
            self._y_true_binary == ignore_label, 0, self._y_true_binary)
        simple_loss_small = simple_bce(small_loss_labels, self._y_pred_binary)
        simple_loss_large = simple_bce(large_loss_labels, self._y_pred_binary)
        self.assertGreater(ignorant_loss, simple_loss_small)
        self.assertGreater(simple_loss_large, ignorant_loss)


if __name__ == "__main__":
    tf.test.main()
