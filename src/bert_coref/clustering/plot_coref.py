#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the predictions of an encoder model.

The input is a `jsonlines` dataset with the fields `text` and `labels`.
`-1` will be interpreted as the label of tokens that do not
belong to any coreference cluster.

The visualized model predictions will be saved.
"""

from argparse import ArgumentParser, Namespace
from os import mkdir
from os.path import exists, join as os_path_join
from typing import Tuple

import numpy as np
from transformers import TFAutoModel, PreTrainedTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from bert_coref import (
    check_output_dir,
    check_positive_int,
    check_positive_float,
    get_jsonl_dataset,
    check_model_path,
    get_tokenizer,
    prepare_dataset_split
)
from bert_coref.clustering import plot_embeddings


def get_embedding_plot_args() -> Namespace:
    """Get command line arguments to plot embeddings."""
    parser = ArgumentParser(description="Arguments for plotting encoder embeddings.")
    parser.add_argument("dataset", type=get_jsonl_dataset,
                        help="Path to the input dataset.")
    parser.add_argument("output_dir", type=check_output_dir,
                        help="Path to the output directory. If it does not exist, it will be created.")
    parser.add_argument("--tokenizer", required=True, type=get_tokenizer,
                        help="Path to the trained tokenizer.")
    parser.add_argument("--model", required=True, type=check_model_path,
                        help="Path to the trained encoder.")
    parser.add_argument("--pca-components", dest="pca_components", type=check_positive_int,
                        help="The number of PCA components. If specified, PCA will be applied "
                             "directly to the model outputs. PCA will not be used otherwise. Optional.")
    parser.add_argument("--perplexity", type=check_positive_float, default=30.,
                        help="The perplexity parameter of the t-SNE "
                             "algorithm. Defaults to `30.0`.")
    parser.add_argument("--batch-size", dest="batch_size", type=check_positive_int, default=32,
                        help="The batch size used for inference. Defaults to `32`.")
    parser.add_argument("--array-save-dir", dest="array_save_dir", type=check_output_dir,
                        help="Path to a directory where the `(input_ids, labels, embeddings)` "
                             "triplets can be saved (one file per example, after t-SNE). "
                             "If not specified, these arrays will not be saved. Optional.")
    args = parser.parse_args()
    assert args.pca_components is None or args.pca_components > 2, \
        "You need to specify more than 2 PCA components."
    return args


def filter_subwords(
        input_ids: np.array,
        labels: np.array,
        embeddings: np.array,
        tokenizer: PreTrainedTokenizer,
        subword_prefix: str = "##"
) -> Tuple[np.array, np.array, np.array]:
    """Filter out subword tokens in non-starting positions.

    Args:
        input_ids: The input token IDs, an array of shape `(sequence_length)`.
        labels: The token labels, an array of shape `(sequence_length)`.
        embeddings: The token embeddings, an array of shape `(sequence_length, embedding_size)`.
        tokenizer: A tokenizer that can convert token IDs to tokens.
        subword_prefix: A prefix indicating that a token is a non-starting subword. Defaults to `'##'`.

    Returns:
        The input IDs, labels and embeddings with the non-starting subwords removed.
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    start_positions = [i for i, token in enumerate(tokens) if not token.startswith(subword_prefix)]
    return input_ids[start_positions], labels[start_positions], embeddings[start_positions, :]


def filter_ignored_elements(
        embedding_matrix: np.array,
        labels: np.array,
        ignore_label: int
) -> Tuple[np.array, np.array]:
    """Remove ignored labels and the corresponding embeddings from the data.

    Args:
        embedding_matrix: A matrix of token embeddings of shape `(num_tokens, embedding_size)`.
        labels: A label vector of shape `(num_tokens)`.
        ignore_label: The label with which ignored tokens are marked.

    Returns:
        An embedding matrix and label vector with the ignored labels removed.
    """
    selection = labels != ignore_label
    return embedding_matrix[selection], labels[selection]


def replace_dangling_labels(labels: np.array, ignore_label: int) -> np.array:
    """Replace labels occurring only once with an `ignore_label`."""
    num_occurrences = np.array([np.count_nonzero(labels == label) for label in labels])
    return np.where(num_occurrences == 1, ignore_label, labels)


def main() -> None:
    """Main function."""
    args = get_embedding_plot_args()
    tokenizer = args.tokenizer
    output_dir = args.output_dir
    ignore_label = -1
    if not exists(output_dir):
        mkdir(output_dir)
    array_save_dir = args.array_save_dir
    if array_save_dir is not None and not exists(array_save_dir):
        mkdir(array_save_dir)

    dataset, _ = prepare_dataset_split(
        dataset=args.dataset,
        tokenizer=tokenizer,
        ignore_label=ignore_label,
        batch_size=args.batch_size,
        text_column="text",
        label_column="labels",
    )
    model = TFAutoModel.from_pretrained(args.model)
    pca = PCA(n_components=args.pca_components) if args.pca_components is not None else None
    tsne = TSNE(perplexity=args.perplexity, learning_rate="auto", init="pca")

    for i, (inputs, labels) in enumerate(dataset):
        embeddings = model.predict_on_batch(inputs).last_hidden_state
        input_ids, _ = inputs
        for j, (id_vector, label_vector, embedding_matrix) in enumerate(
                zip(input_ids.numpy(), labels.numpy(), embeddings)):
            __, label_vector, embedding_matrix = filter_subwords(
                input_ids=id_vector,
                labels=label_vector,
                embeddings=embedding_matrix,
                tokenizer=tokenizer
            )
            if pca is not None:
                embedding_matrix = pca.fit_transform(embedding_matrix)
            embedding_matrix = tsne.fit_transform(embedding_matrix)
            # Save the arrays
            if array_save_dir is not None:
                np.savez(os_path_join(array_save_dir, f"batch_{i}_sample_{j}.npz"),
                         input_ids=id_vector, labels=label_vector, embeddings=embedding_matrix)

            embedding_matrix, label_vector = filter_ignored_elements(
                embedding_matrix=embedding_matrix,
                labels=label_vector,
                ignore_label=ignore_label
            )
            label_vector = replace_dangling_labels(label_vector, ignore_label)
            plot_embeddings(
                embeddings=embedding_matrix,
                img_output=os_path_join(output_dir, f"batch_{i}_sample_{j}.png"),
                labels=label_vector,
                black_label=ignore_label
            )


if __name__ == "__main__":
    main()
