#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script to evaluate clustering.

The input is an `xtsv`-style document with a header, but blank lines should separate
documents rather than sentences. Each non-empty line is expected to contain 12 fields
separated by `\\t` characters. The first field must be the token form, the second -
a coreference class label and the last one - the coreference cluster label. The
annotation contained in other fields will not be used.

The outputs are real numbers representing the clustering scores (one per document).
"""

from sys import stdin
from types import MappingProxyType
from collections import defaultdict
from argparse import Namespace, ArgumentParser, FileType
from typing import Iterable, Tuple, Generator, List

import numpy as np

from bert_coref.clustering import ClusteredXtsvToken, compute_purity, compute_rand_score, compute_nmi


def get_eval_args() -> Namespace:
    """Get command line arguments to evaluate clustering."""
    parser = ArgumentParser(
        description="Specify arguments to calculate a clustering metric.")
    parser.add_argument("input_file", type=FileType("r", encoding="utf-8"),
                        default=stdin, nargs="?", help="Input file or `stdin`.")
    parser.add_argument("--metrics", choices=["purity", "rand_index", "NMI"],
                        default=["rand_index"], nargs="+",
                        help="The metric types to use. Defaults to `['rand_index']`.")
    args = parser.parse_args()
    args.metrics = set(args.metrics)
    return args


SCORE2FUNC = MappingProxyType({
    "purity": compute_purity,
    "rand_index": compute_rand_score,
    "NMI": compute_nmi
})


def collect_documents(
        lines: Iterable[str],
        sep: str = "\t"
) -> Generator[List[ClusteredXtsvToken], None, None]:
    """Collect documents from the input.

    Args:
        lines: The `xtsv` style input lines.
        sep: The input field separator. Defaults to `'\\t'`.

    Returns:
        A generator that yields documents as lists of tokens.
    """
    document = []
    for line in map(str.strip, lines):
        if len(line) == 0:
            if len(document) != 0:
                yield document
                document = []
        else:
            document.append(ClusteredXtsvToken(*line.split(sep)))
    if len(document) != 0:
        yield document


def collect_class_cluster_labels(tokens: Iterable[ClusteredXtsvToken]) -> Tuple[np.array, np.array]:
    """Iterate over `tokens` and get an array of class labels
    and an array of cluster labels.

    Args:
        tokens: The tokens of the input document.

    Returns:
        Two 1D arrays of the same shape, one for class labels and one for cluster labels.
    """
    classes, clusters = [], []
    for token in tokens:
        classes.append(token.koref)
        clusters.append(token.koref_cluster)
    return np.array(classes), np.array(clusters)


def main() -> None:
    """Main function."""
    args = get_eval_args()
    input_file = args.input_file
    next(input_file)  # Skip header
    documents = collect_documents(input_file)
    scores = defaultdict(list)
    funcs = {k: v for k, v in SCORE2FUNC.items() if k in args.metrics}
    for document in documents:
        classes, clusters = collect_class_cluster_labels(document)
        for score_name, score_func in funcs.items():
            scores[score_name].append(score_func(clusters, classes))
    for score_name, score_values in scores.items():
        print(f"Average {score_name}: {sum(score_values) / len(score_values)}")


if __name__ == "__main__":
    main()
