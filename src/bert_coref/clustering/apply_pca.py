#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Apply PCA to a matrix of token embeddings.

`csv` input format is required. There should be `n` lines
in the input with `m` values in each line. `n` is the number
of tokens, `m` is the embedding dimensionality.

The output will have the same format, but with reduced embeddings.
"""

import csv
from argparse import Namespace
from typing import IO

import numpy as np
from sklearn.decomposition import PCA

from bert_coref import get_io_parser, check_positive_int


def get_pca_args() -> Namespace:
    """Get command line arguments for PCA."""
    parser = get_io_parser("PCA arguments")
    parser.add_argument("--num-components", dest="n_components",
                        default=100, type=check_positive_int,
                        help="The number of components (dimensions) to keep. "
                             "Defaults to `100`.")
    return parser.parse_args()


def read_input_matrix(csv_lines: IO) -> np.array:
    """Read the input lines into a numpy array."""
    reader = csv.reader(csv_lines, quoting=csv.QUOTE_NONNUMERIC)
    return np.array([line for line in reader])


def write_output_matrix(matrix: np.array, output_file: IO) -> None:
    """Write a matrix to the output file or `stdout`."""
    assert len(matrix.shape) == 2, f"The input must be a matrix. " \
                                   f"Got an array of shape {matrix.shape}."
    writer = csv.writer(output_file)
    writer.writerows(matrix.tolist())


def main() -> None:
    """Main function."""
    args = get_pca_args()
    embedding_matrix = read_input_matrix(args.input_file)
    pca_model = PCA(n_components=args.n_components)
    reduced_matrix = pca_model.fit_transform(embedding_matrix)
    write_output_matrix(reduced_matrix, args.output_file)


if __name__ == "__main__":
    main()
