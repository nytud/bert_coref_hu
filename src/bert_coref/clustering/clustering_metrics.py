# -*- coding: utf-8 -*-

"""A module for metrics that measure the quality of clustering."""

from collections import Counter
from typing import Tuple

import numpy as np
from sklearn.metrics import rand_score, normalized_mutual_info_score


def filter_labels_for_eval(clusters: np.array, classes: np.array) -> Tuple[np.array, np.array]:
    """A helper function to filter labels for better validation.
    More specifically, remove the `ith` element of both `clusters` and `classes`
    if `classes[i] < 0` and `clusters[i] <= 0`.
    """
    indices = np.any([clusters > 0, classes >= 0], axis=0)
    return clusters[indices], classes[indices]


def compute_purity(clusters: np.array, classes: np.array) -> float:
    """Calculate the classification purity based on class labels.
    The `ith` element of both `clusters` and `classes` will be ignored
    if `classes[i] < 0` and `clusters[i] <= 0` as these values indicate
    meta labels.

    Args:
        clusters: An array of shape `(n_tokens)`, the clustering labels.
        classes: An array of shape `(n_tokens)`, the class labels.

    Returns:
        The clustering purity, a real number `x` such that `0 < x <= 1`.

    Raises:
        `AssertionError` if the shapes of `clusters` and `classes` are
        not equal.
    """
    assert classes.shape == clusters.shape, \
        f"The shapes of the cluster labels ({clusters.shape}) and the " \
        f"class labels ({classes.shape}) must be equal."
    clusters, classes = filter_labels_for_eval(clusters, classes)
    unique_clusters = {label for label in clusters if label >= 0}
    nominator, denominator = 0, 0
    for cluster_label in unique_clusters:
        indices = clusters == cluster_label
        classes_in_cluster = Counter(classes[indices])
        _, dominant_class_freq = classes_in_cluster.most_common(1)[0]
        nominator += dominant_class_freq
        denominator += np.sum(indices)
    if denominator == 0:
        result = 1.
    else:
        result = nominator / denominator
    return result


def compute_nmi(clusters: np.array, classes: np.array) -> float:
    """Compute the Normalized Mutual Information. This function wraps
    `sklearn.metrics.normalized_mutual_info_score`.
    The `ith` element of both `clusters` and `classes` will be ignored
    if `classes[i] < 0` and `clusters[i] <= 0` as these values indicate
    meta labels.


    Args:
        clusters: An array of shape `(n_tokens)`, the clustering labels.
        classes: An array of shape `(n_tokens)`, the class labels.

    Returns:
        The NMI score of the clustering, a real number `x` such that `0 < x < 1`.

    Raises:
        `AssertionError` if the shapes of `clusters` and `classes` are
        not equal.
    """
    assert classes.shape == clusters.shape, \
        f"The shapes of the cluster labels ({clusters.shape}) and the " \
        f"ground truth (class) labels ({classes.shape}) must be equal."
    clusters, classes = filter_labels_for_eval(clusters, classes)
    return normalized_mutual_info_score(classes, clusters)


def compute_rand_score(clusters: np.array, classes: np.array) -> float:
    """Compute the rand index. This function wraps `sklearn.metrics.rand_score`.
    The `ith` element of both `clusters` and `classes` will be ignored
    if `classes[i] < 0` and `clusters[i] <= 0` as these values indicate
    meta labels.

    Args:
        clusters: An array of shape `(n_tokens)`, the clustering labels.
        classes: An array of shape `(n_tokens)`, the class labels.

    Returns:
        The rand index of the clustering, a real number `x` such that `0 <= x <= 1`.

    Raises:
        `AssertionError` if the shapes of `clusters` and `classes` are
        not equal.
    """
    assert classes.shape == clusters.shape, \
        f"The shapes of the cluster labels ({clusters.shape}) and the " \
        f"ground truth (class) labels ({classes.shape}) must be equal."
    clusters, classes = filter_labels_for_eval(clusters, classes)
    return rand_score(classes, clusters)
