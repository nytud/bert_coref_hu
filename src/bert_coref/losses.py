# -*- coding: utf-8 -*-

"""A module for custom loss functions."""

from typing import Dict, List, Union

import tensorflow as tf
from numpy import inf
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as keras_backend


@tf.function
def select_anchors(
        y_pred: tf.Tensor,
        y_true: tf.Tensor,
        anchor_label: int = 2
) -> tf.Tensor:
    """Select anchors from a tensor of embeddings assuming that each
    data point in the batch has exactly one anchor.

    Args:
        y_pred: An embedding tensor of shape
            `(batch_size, sequence_length, embedding_size)`.
        y_true: The labels, a tensor of shape `(batch_size, sequence_length)`.
        anchor_label: The label that indicates anchors. Defaults to `2`.

    Returns:
        The anchors, a tensor of shape `(batch_size, 1, embedding_size)`.
    """
    masked_preds = tf.where(
        tf.expand_dims(y_true, axis=-1) == anchor_label, y_pred, 0.)
    return tf.reduce_sum(masked_preds, axis=1, keepdims=True)


@tf.function
def calculate_distances(
        references: tf.Tensor,
        anchors: tf.Tensor
) -> tf.Tensor:
    """Calculate Euclidean distances between vectors.

    Args:
        references: Vectors with which the `anchors` will be compared.
            A tensor of shape `(batch_size, sequence_length, embedding_size)`.
        anchors: A tensor of anchor vectors of shape
            `(batch_size, 1, embedding_size)`.

    Returns:
        A tensor of shape `(batch_size, sequence_length)`. Each element is
        the distance of a token representation from the anchor in the
        corresponding batch.
    """
    diffs = tf.subtract(references, anchors)
    return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1))


@tf.function
def mean_positive_distances(
        distances: tf.Tensor,
        y_true: tf.Tensor,
        positive_label: int = 1
) -> tf.Tensor:
    """Calculate the mean distance of the anchors
    from the positive references.

    Args:
        distances: A tensor whose values are Euclidean distances. It is
            of shape `(batch_size, sequence_length)`.
        y_true: The labels, a tensor of shape `(batch_size, sequence_length)`.
        positive_label: The label that indicates positive references.
            Defaults to `1`.

    Returns:
        The mean distance tensor of shape `(batch_size)`.
    """
    indicator = y_true == positive_label
    num_positives = tf.maximum(
        tf.convert_to_tensor(1, dtype=distances.dtype),
        tf.reduce_sum(tf.cast(indicator, distances.dtype), axis=1)
    )
    distances = tf.where(indicator, distances, 0.)
    return tf.divide(tf.reduce_sum(distances, axis=1), num_positives)


@tf.function
def min_negative_distance(
        distances: tf.Tensor,
        y_true: tf.Tensor,
        negative_label: int = 0
) -> tf.Tensor:
    """Calculate the minimal distance of the anchors
    from the negative references.

    Args:
        distances: A tensor whose values are Euclidean distances. It is
            of shape `(batch_size, sequence_length)`.
        y_true: The labels, a tensor of shape `(batch_size, sequence_length)`.
        negative_label: The label that indicates negative references.
            Defaults to `0`.

    Returns:
        The minimum distance tensor of shape `(batch_size)`.
    """
    distances = tf.where(y_true == negative_label, distances, inf)
    return tf.reduce_min(distances, axis=1)


@tf.function
def binary_crossentropy(y_true: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    """Calculate the binary crossentropy from probabilities
    without reduction.
    """
    epsilon = keras_backend.epsilon()
    y_hat = keras_backend.clip(tf.squeeze(y_hat), epsilon, 1 - epsilon)
    label1_term = tf.cast(y_true, y_hat.dtype) * tf.math.log(y_hat + epsilon)
    # noinspection PyTypeChecker
    label0_term = tf.cast(1 - y_true, y_hat.dtype) * tf.math.log(
        1. - y_hat + epsilon)
    return -1 * (label1_term + label0_term)


@tf.function
def binary_crossentropy_from_logits(
        y_true: tf.Tensor,
        y_pred: tf.Tensor
) -> tf.Tensor:
    """Calculate the binary crossentropy from probabilities
    without reduction.
    """
    y_pred = tf.sigmoid(y_pred)
    return binary_crossentropy(y_true, y_pred)


class ClusterTripletLoss(Loss):
    """A class for a loss function aimed at minimizing the distances
    between embeddings of tokens that belong to the same cluster and
    maximizing the distances between tokens belonging to different
    clusters.

    This loss calculates the mean loss over the batch, it does not
    support any other reduction techniques.
    """

    def __init__(self, margin: float, name: str = "cluster_triplet") -> None:
        """Initialize the loss objective.

        Args:
            margin: The triplet loss margin, a non-negative float.
            name: The loss name. Defaults to `'cluster_triplet'`.
        """
        assert margin >= 0, \
            f"`margin` must be a non-negative float, got {margin}."
        super(ClusterTripletLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            name=name
        )
        self.margin = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate the loss.

        Args:
            y_true: The ground truth labels, one per token. This is a tensor
                of shape `(batch_size, sequence_length, embedding_size)`.
                The possible values are:
                    2: for the anchor token (one token in each data point)
                    1: for the tokens belonging to the same class as the anchor
                    0: for the tokens belonging to another class
                    any other label: for the tokens that are to be ignored
            y_pred: The token embeddings. This is tensor of shape
                (`batch_size`, `sequence_length`).

        Returns:
            The loss value as a scalar tensor.
        """
        anchors = select_anchors(y_pred, y_true)
        distances = calculate_distances(y_pred, anchors)
        positive_dist = tf.reduce_mean(
            mean_positive_distances(distances, y_true))
        negative_dist = tf.reduce_mean(
            min_negative_distance(distances, y_true))
        return tf.maximum(0., positive_dist - negative_dist + self.margin)

    def get_config(self) -> Dict[str, Union[List, Dict, int, float, str]]:
        """Get a loss configuration object."""
        base_config = super().get_config()
        return {**base_config, "margin": self.margin}


class IgnorantBinaryCrossentropy(Loss):
    """Binary crossentropy with an ignored label.

    This loss calculates the mean loss over the samples, it does not
    support any other reduction techniques.
    """

    def __init__(
            self,
            ignore_label: int,
            from_logits: bool = True,
            name: str = "ignorant_binary_crossentropy"
    ) -> None:
        """Initialize the loss objective.
        
        Args:
            ignore_label: The label that will be ignored.
            from_logits: It specifies whether the inputs are logits
                (rather than probability distributions). Defaults to `True`.
            name: The loss name. Defaults to `'ignorant_binary_crossentropy'`.
        """
        super(IgnorantBinaryCrossentropy, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            name=name
        )
        self._base_loss_fn = binary_crossentropy_from_logits if from_logits \
            else binary_crossentropy
        self._from_logits = from_logits
        self.ignore_label = ignore_label

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate the loss."""
        base_loss = self._base_loss_fn(y_true, y_pred)
        loss = tf.where(y_true == self.ignore_label, 0., base_loss)
        num_non_ignore = tf.reduce_sum(
            tf.where(y_true == self.ignore_label, 0, 1), axis=-1)
        loss = tf.divide(
            tf.reduce_sum(loss, axis=-1), tf.cast(num_non_ignore, loss.dtype))
        return tf.reduce_mean(loss)

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    def get_config(self) -> Dict[str, Union[List, Dict, int, float, str]]:
        """Get a loss configuration object."""
        base_config = super().get_config()
        base_config.pop("reduction")
        return {
            **base_config,
            "ignore_label": self.ignore_label,
            "from_logits": self._from_logits
        }
