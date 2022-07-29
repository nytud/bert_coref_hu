#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot a single coreference relation.

The input is a file that contains 3 `numpy` arrays:
    * input_ids: an integer array of shape `(num_subwords,)`. It contains subword token IDs.
    * labels: an integer array of shape `(num_tokens)`. It contains coreference labels.
    * embeddings: an embedding matrix of shape `(num_tokens, 2)`. It contains token embeddings.

The output is a image where only tokens with a specific label are plotted.
"""

from argparse import ArgumentParser, Namespace
from types import MappingProxyType

import numpy as np
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizer

from bert_coref import check_output_dir, get_tokenizer, check_npz, check_positive_int


COLORS = MappingProxyType({
    "blue": "b",
    "green": "g",
    "red": "r",
    "cyan": "c",
    "magenta": "m",
    "yellow": "y",
    "black": "k",
    "white": "w"
})


def get_relation_plot_args() -> Namespace:
    """Get command line arguments for plotting."""
    parser = ArgumentParser(
        description="Specify arguments to plot a single coreference relation.")
    parser.add_argument("input_arrays", type=check_npz,
                        help="Path to a file that contains 3 arrays: "
                             "`input_ids`, `labels` and `embeddings`.")
    parser.add_argument("output_image", type=check_output_dir,
                        help="Path to the output image.")
    parser.add_argument("--tokenizer", type=get_tokenizer, default="SZTAKI-HLT/hubert-base-cc",
                        help="The tokenizer which detokenizes the input IDs. "
                             "Defaults to `'SZTAKI-HLT/hubert-base-cc'`.")
    parser.add_argument("--label", type=check_positive_int, required=True,
                        help="The label to plot.")
    parser.add_argument("--color", choices=list(COLORS.keys()), default="red",
                        help="The plot color. Defaults to `'red'`.")
    return parser.parse_args()


def detokenize(
        input_ids: np.array,
        tokenizer: PreTrainedTokenizer,
        subword_prefix: str = "##"
) -> np.array:
    """Detokenize a sequence of input IDs."""
    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    tokens = []
    for subword_token in subword_tokens:
        if subword_token.startswith(subword_prefix):
            tokens[-1] += subword_token[len(subword_prefix):]
        else:
            tokens.append(subword_token)
    return np.array(tokens)


def main() -> None:
    """Main function."""
    args = get_relation_plot_args()
    tokens = detokenize(args.input_arrays["input_ids"], args.tokenizer)
    labels = args.input_arrays["labels"]
    embeddings = args.input_arrays["embeddings"]
    assert tokens.shape == labels.shape

    indices = labels == args.label
    embeddings = embeddings[indices]
    tokens = tokens[indices]

    xs, ys = embeddings.T.tolist()
    plt.scatter(xs, ys, c=COLORS[args.color])
    for i, token in enumerate(tokens):
        plt.annotate(token, (xs[i], ys[i]))
    plt.tight_layout()
    plt.savefig(args.output_image)


if __name__ == "__main__":
    main()
