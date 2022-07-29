# -*- coding: utf-8 -*-

"""A module for custom metrics."""

from typing import Dict, Optional, Union, List

import tensorflow as tf
from tensorflow.keras.metrics import Precision
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient as MCCoef


class IgnorantBinaryMCC(MCCoef):
    """A class for the Matthews Correlation coefficient
    for binary classification with an ignored label.
    """

    def __init__(
            self,
            ignore_label: int,
            name: str = "ignorant_binary_mcc",
            **kwargs) -> None:
        """Initialize the metric. Do not specify the number of classes.
        This metric is intended for binary classification only.

        Args:
            ignore_label: The label that will be ignored.
            name: Name of the metric. Defaults to `'ignorant_binary_mcc'`.
            **kwargs: Additional parent class keyword arguments.
        """
        super(IgnorantBinaryMCC, self).__init__(
            num_classes=2, name=name, **kwargs)
        self.ignore_label = ignore_label

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """Update the metric state.

        Args:
            y_true: The ground truth labels, an integer tensor of shape
                `(batch_size, sequence_length)`.
            y_pred: The prediction logits, a float tensor of shape
                `(batch_size, sequence_length)`.
            sample_weight: Sample weights for the examples that are not
                to be ignored. Optional.
        """
        y_pred = tf.boolean_mask(
            tf.squeeze(y_pred), y_true != self.ignore_label)
        y_true = tf.boolean_mask(y_true, y_true != self.ignore_label)
        y_true = tf.one_hot(
            y_true, depth=self.num_classes, dtype=y_pred.dtype)
        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.stack([1 - y_pred, y_pred], axis=-1)
        super(IgnorantBinaryMCC, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Union[List, Dict, int, float, str]]:
        """Return a configuration dictionary."""
        base_config = super(IgnorantBinaryMCC, self).get_config()
        base_config.pop("num_classes")
        return {**base_config, "ignore_label": self.ignore_label}


class IgnorantPrecision(Precision):
    """A class for precision with an ignored label."""

    def __init__(
            self,
            ignore_label: int,
            name: str = "ignorant_precision",
            **kwargs
    ) -> None:
        """Initialize the metric.

        Args:
            ignore_label: The label that will be ignored.
            name: Name of the metric. Defaults to `'ignorant_precision'`.
            **kwargs: Additional parent class keyword arguments.
        """
        super(IgnorantPrecision, self).__init__(name=name, **kwargs)
        self.ignore_label = ignore_label

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """Update the metric state.

        Args:
            y_true: The ground truth labels, an integer tensor of shape
                `(batch_size, sequence_length)`.
            y_pred: The prediction logits, a float tensor of shape
                `(batch_size, sequence_length)`.
            sample_weight: Sample weights for the examples that are not
                to be ignored. Optional.
        """
        y_pred = tf.boolean_mask(
            tf.squeeze(y_pred), y_true != self.ignore_label)
        y_true = tf.boolean_mask(y_true, y_true != self.ignore_label)
        y_pred = tf.sigmoid(y_pred)
        super(IgnorantPrecision, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)
