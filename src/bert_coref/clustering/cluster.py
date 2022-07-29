#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script that implements a simple clustering algorithm.

Sequences of tokens are embedded by an encoder model. Some of the
tokens can be ignored by providing a mask label (using `-1` as the mask).
A group of tokens will be identified as a cluster if for any token pair
`(x, y)` in the group, the distance between their corresponding embeddings
are smaller than a predefined threshold.

"""

from sys import stdout
from collections import Counter
from dataclasses import fields, dataclass, field
from argparse import ArgumentParser, Namespace, FileType
from typing import (
    Union, Iterable, Generator, Tuple, Dict, Optional,
    Any, Set, FrozenSet, MutableSequence
)

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from transformers import TFAutoModel

from bert_coref import (
    get_jsonl_dataset,
    get_tokenizer,
    check_model_path,
    check_positive_int,
    check_positive_float,
    prepare_dataset_split,
    XtsvToken
)
from bert_coref.clustering import filter_ignored_elements


def get_clustering_args() -> Namespace:
    """Get command line arguments for clustering."""
    parser = ArgumentParser(description="Specify clustering arguments")
    parser.add_argument("dataset", type=get_jsonl_dataset,
                        help="Path to the input dataset.")
    parser.add_argument("output_file", type=FileType("r", encoding="utf-8"),
                        default=stdout, nargs="?",
                        help="Output file or `stdout`.")
    parser.add_argument("--tokenizer", required=True, type=get_tokenizer,
                        help="Path to the trained tokenizer.")
    parser.add_argument("--model", required=True, type=check_model_path,
                        help="Path to the trained encoder.")
    parser.add_argument("--batch-size", dest="batch_size", type=check_positive_int, default=32,
                        help="The batch size used for inference. Defaults to `32`.")
    parser.add_argument("--threshold", type=check_positive_float, default=8.,
                        help="Clustering threshold value. Defaults to `8.0`.")
    return parser.parse_args()


@dataclass
class ClusteredXtsvToken(XtsvToken):
    """A dataclass that adds an `koref_cluster` field to `XtsvToken`."""
    koref_cluster: Optional[int] = field(
        default=None,
        metadata={"help": "A coreference cluster label assigned to the token."}
    )

    def __post_init__(self) -> None:
        """Convert the coreference cluster label to integer."""
        super().__post_init__()
        if isinstance(self.koref_cluster, str):
            self.koref_cluster = int(self.koref_cluster)


def cluster_tokens(embedding_matrix: np.array, threshold: float) -> np.array:
    """Cluster tokens using their embeddings.

    Args:
        embedding_matrix: A token embedding matrix of shape `(num_tokens, embedding_size)`.
        threshold: The clustering threshold, the minimum required distance between the
            the embeddings of the tokens belonging to the same cluster and any other
            embedding.

    Returns:
        A cluster label for each token, a vector of shape `(num_tokens,)`.
    """
    binary_distances = euclidean_distances(embedding_matrix) < threshold
    cluster_labels = []
    for row1 in binary_distances:
        for i, row2 in enumerate(binary_distances, start=1):
            if np.array_equal(row1, row2):
                cluster_labels.append(i)
                break
    return np.array(cluster_labels)


def get_token_cluster_tuples(
        tokens: Iterable[str],
        orig_labels: np.array,
        filtered_labels: np.array,
        ignore_label: int,
) -> Generator[Tuple[str, int], None, None]:
    """Assign cluster labels and linguistic annotation to tokens.

    Args:
        tokens: The input token sequence of length `num_tokens`.
        orig_labels: A label vector of shape (`num_tokens`,). The values
            corresponding to non-ignored tokens in this vector are irrelevant.
        filtered_labels: The final cluster labels whose length is less or equal
            than `num_tokens`.
        ignore_label: The label that identifies irrelevant tokens.

    Returns:
        A generator that yields `(token, cluster_label)` pairs.

    Raises:
        `AssertionError` if the number of non-ignored labels in `orig_labels`
        is not equal to the number of elements in `filtered_labels`.
    """
    num_non_ignored = orig_labels[orig_labels != ignore_label].shape[0]
    num_filtered = filtered_labels.shape[0]
    assert num_non_ignored == num_filtered, \
        f"{num_non_ignored} filtered labels were expected, got {num_filtered}."
    filtered_labels = iter(filtered_labels)
    for token, orig_label in zip(tokens, orig_labels):
        cluster_label = next(filtered_labels) if orig_label != ignore_label else ignore_label
        yield token, cluster_label


def restore_xtsv_tokens(
        form_label_pairs: Iterable[Tuple[str, int]],
        annotation: Dict[str, Any],
        subword_prefix: str = "##",
        meta_tokens: Union[Set[str], FrozenSet[str]] = frozenset({"[CLS]", "[SEP]"}),
        ignore_label: int = -1
) -> Generator[ClusteredXtsvToken, None, None]:
    """Restore `xtsv` tokens from forms, labels and linguistic annotation.

    Args:
         form_label_pairs: An iterable of pairs each of which contains a token form and a label.
         annotation: A `dict` whose keys are linguistic tags and the values are sequences of tags
            (one per token). It must also contain a `text` key whose value is a sequence of token forms.
         subword_prefix: A prefix that identifies subword tokens in non-starting positions.
            The forms of these tokens will be appended to the forms of the preceding tokens.
         meta_tokens: token forms that should be ignored. Defaults to `frozenset({'[CLS]', '[SEP]'})`.
         ignore_label: The label assigned to ignored tokens. Defaults to `1`.

    Returns:
        A generator that yields annotated `xtsv` tokens.
    """
    target_forms = enumerate(annotation.pop("text"))
    i, target_form = next(target_forms)
    prev_form = ""
    act_label = ignore_label
    for form, label in form_label_pairs:
        if form in meta_tokens:
            continue
        if prev_form == "":
            act_label = label
        if form.startswith(subword_prefix):
            form = form[len(subword_prefix):]
        prev_form += form
        if prev_form == target_form:
            tags = {key: value[i] for key, value in annotation.items()}
            yield ClusteredXtsvToken(prev_form, koref_cluster=act_label, **tags)
            prev_form = ""
            i, target_form = next(target_forms, (None, None))


def relabel_deprel(tokens: Iterable[ClusteredXtsvToken]) -> Generator[ClusteredXtsvToken, None, None]:
    """Add sentence index to the `id` and `deprel` labels.
    This function modifies the inputs (`XtsvToken` is mutable) and returns
    a generator that yields the modified tokens.
    It is assumed that the `id` and `deprel` fields of the tokens are specified.

    Args:
        tokens: The annotated input tokens.

    Returns:
        A generator that yields the tokens with the modified `id`  and `deprel` fields.
    """
    sent_id = 0
    for token in tokens:
        if token.id == 1:
            sent_id += 1
        token.id = f"s{sent_id}_{token.id}"
        token.deprel = f"s{sent_id}_{token.deprel}"
        yield token


def filter_coref_clusters(
        tokens: MutableSequence[ClusteredXtsvToken],
        ignore_label: int = -1,
        new_cluster_label: int = 0
) -> None:
    """Filter coreference clusters based on dependency relations.
    If the set of the dependency parents of the tokens in a cluster
    does not contain at least two tokens not present in the cluster,
    than remove the cluster (change the `koref` labels of its tokens).

    Args:
        tokens: A sequence of tokens that may comprise multiple coreference clusters.
        ignore_label: The label that indicates tokens out of clusters.
        new_cluster_label: The new label of the tokens whose clusters were removed.
            Defaults to `0`.
    """
    clusters = {token.koref for token in tokens}
    if ignore_label in clusters:
        clusters.remove(ignore_label)
    for cluster in clusters:
        tokens_in_cluster = [tok for tok in tokens if tok.koref_cluster == cluster]
        if len(tokens_in_cluster) < 2:
            continue
        ids, deprels = set(), set()
        for token_in_cluster in tokens_in_cluster:
            ids.add(token_in_cluster.id)
            deprels.add(token_in_cluster.deprel)
        outer_parents = deprels.difference(ids)
        if len(outer_parents) < 2:
            for token_in_cluster in tokens_in_cluster:
                token_in_cluster.koref_cluster = new_cluster_label


def relabel_singletons(
        tokens: MutableSequence[ClusteredXtsvToken],
        new_cluster_label: int = 0,
        new_class_label: int = -1
) -> None:
    """Set the `koref_cluster` attribute of token `t` to `new_cluster_label` if
    `t` is the only element in its cluster. Set the `koref` attribute to
    `new_class_label` if `t` is the only element in its class.
    """
    cluster_counts = Counter(token.koref_cluster for token in tokens)
    class_counts = Counter(token.koref for token in tokens)
    for token in tokens:
        if cluster_counts[token.koref_cluster] == 1:
            token.koref_cluster = new_cluster_label
        if class_counts[token.koref] == 1:
            token.koref = new_class_label


def main() -> None:
    """Main function."""
    args = get_clustering_args()
    ignore_label = -1
    tokenizer = args.tokenizer
    output_file = args.output_file
    orig_dataset = args.dataset
    text_col, label_col = "text", "labels"
    dataset, _ = prepare_dataset_split(
        dataset=orig_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        ignore_label=ignore_label,
        text_column=text_col,
        label_column=label_col,
    )
    model = TFAutoModel.from_pretrained(args.model)
    data_idx = 0
    print("\t".join(token_field.name for token_field
                    in fields(ClusteredXtsvToken)), file=output_file)  # Print header
    for inputs, labels in dataset:
        embeddings = model.predict_on_batch(inputs).last_hidden_state
        input_ids, _ = inputs
        for id_vector, label_vector, embedding_matrix in zip(
                input_ids.numpy(), labels.numpy(), embeddings):
            embedding_matrix, filtered_labels = filter_ignored_elements(
                embedding_matrix=embedding_matrix,
                labels=label_vector,
                ignore_label=ignore_label
            )
            filtered_labels = cluster_tokens(embedding_matrix, args.threshold)
            pairs = get_token_cluster_tuples(
                tokens=tokenizer.convert_ids_to_tokens(id_vector),
                orig_labels=label_vector,
                filtered_labels=filtered_labels,
                ignore_label=ignore_label
            )
            annotation = orig_dataset[data_idx]
            annotation["koref"] = annotation.pop(label_col)
            annotation[text_col] = annotation[text_col].split()
            restored_tokens = restore_xtsv_tokens(pairs, annotation)
            restored_tokens = list(relabel_deprel(restored_tokens))
            filter_coref_clusters(restored_tokens)
            relabel_singletons(restored_tokens)
            for restored_token in restored_tokens:
                print(restored_token.to_string(), file=output_file)
            print(file=output_file)
            data_idx += 1


if __name__ == "__main__":
    main()
