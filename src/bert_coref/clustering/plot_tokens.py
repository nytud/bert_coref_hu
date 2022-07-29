# -*- coding: utf-8 -*-

"""A module for tools that help plot embeddings with t-SNE."""

from typing import Optional, Sequence, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS
from sklearn.manifold import TSNE


def apply_tsne(matrix: np.array, **kwargs) -> np.array:
    """Reduce the embedding dimension with t-SNE to 2.
    If the number of columns in the input matrix is already 2,
    the function simply return the input.

    Args:
        matrix: The input matrix of size `(num_tokens, embedding_size)`, where
            `embedding_size >= 2`.
        kwargs: Keyword arguments for the t-SNE initialization,
            apart from `n_components`.

    Returns:
        A matrix of shape `(num_tokens, 2)`.

    Raises:
        AssertionError if the input shape is invalid.
    """
    input_shape = matrix.shape
    assert len(input_shape) == 2 and input_shape[-1] >= 2, \
        f"A matrix is expected, got an array of shape {input_shape}."
    if input_shape[-1] == 2:
        res = matrix
    else:
        tsne = TSNE(n_components=2, **kwargs)
        res = tsne.fit_transform(matrix)
    return res


def _map_labels_to_colors(
    labels: Sequence[int],
    black_label: Optional[int] = None
) -> List[str]:
    """A helper function to map an array of labels to colors."""
    label2color = {}
    colors = list(BASE_COLORS.keys())
    unique_labels = set(labels)
    if len(unique_labels) > len(colors):
        raise ValueError(f"There are too many unique labels "
                         f"({len(unique_labels)}), cannot map them "
                         f"to {len(colors)} colors.")
    if black_label is not None:
        colors.remove("k")
        unique_labels.remove(black_label)
        label2color[black_label] = "k"
    label2color.update({label: color for label, color
                        in zip(unique_labels, colors)})
    return [label2color[label] for label in labels]


def plot_embeddings(
        embeddings: np.array,
        img_output: str,
        labels: Optional[Sequence[int]] = None,
        black_label: Optional[int] = None,
) -> None:
    """Plot a matrix of token embeddings.

    Args:
        embeddings: A matrix of shape `(num_tokens, 2)`.
        img_output: Path to the file where the plot will be saved.
        labels: A sequence of `num_tokens` integers. Optional.
        black_label: The label that will be marked black. Relevant only if
            `labels` is specified. Optional.

    Raises:
        AssertionError if the number of labels is not equal to
            the number of tokens.
        ValueError if there are more unique labels than available colors.
    """
    xs, ys = embeddings.T.tolist()
    if labels is not None:
        assert len(xs) == len(labels), \
            f"The number of labels ({len(labels)}) " \
            f"does not equal the number of tokens ({len(xs)})."
        label_colors = _map_labels_to_colors(labels, black_label)
    else:
        label_colors = None
    plt.scatter(xs, ys, c=label_colors)
    plt.savefig(img_output)
