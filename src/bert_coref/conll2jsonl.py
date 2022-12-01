#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Convert a CONLL-style file into jsonlines.

The input is a `tsv` file without header and comments.
Each line should correspond to a token annotated with 2 fields:
word form and label. Data points (= new lines in the output) are
expected to be separated by DOUBLE blank lines!

The output is a file in jsonlines format. Each row will consist of
2 fields: tokens (a string where tokens are separated by spaces) and
labels (an array where the ith label corresponds to the ith label).
"""

from argparse import Namespace
from collections import Counter
from itertools import chain
from dataclasses import fields
from copy import deepcopy
from typing import Sequence, Iterable, Generator, List, Union, Dict

from jsonlines import Writer as JsonlWriter

from bert_coref import get_io_parser, XtsvToken


def get_conll2jsonl_args() -> Namespace:
    """Get command line arguments for data handling or augmentation."""
    parser = get_io_parser("Specify arguments to convert a CONLL file "
                           "into `jsonlines`.")
    parser.add_argument("--create-triplets", dest="create_triplets",
                        action="store_true",
                        help="Create `anchor - positive - negative` triplets "
                             "from the input data.")
    parser.add_argument("--input-ignore-label", dest="input_ignore_label",
                        default="_",
                        help="The label for tokens out of co-reference "
                             "clusters in the input. Defaults to `_`.")
    parser.add_argument("--output-ignore-label", dest="output_ignore_label",
                        type=int, default=-1,
                        help="The label for tokens out of co-reference "
                             "clusters in the output. Defaults to `-1`.")
    return parser.parse_args()


def create_triplets(
        tokens: Sequence[XtsvToken],
        ignore_label: int = -1
) -> Generator[List[XtsvToken], None, None]:
    """Split a single data point into triplets.

    A triplet is a sequence of tokens where:
        1. one token `t` is selected as anchor and has the label `2`,
        2. tokens that belong to the same co-reference cluster as `t`
            have the label `1`,
        3. tokens that belong to another co-reference cluster
            have the label `0`,
        4. `ignore_label` is assigned to all other tokens.

    Args:
        tokens: A sequence of tokens that make up a document.
        ignore_label: A label that is assigned to tokens out of
            co-reference clusters. It cannot be `0`, `1` or `2`.
            It is recommended to use a negative number. Defaults to `-1`.

    Returns:
        A generator of data points as lists of annotated tokens.
    """
    assert ignore_label not in {0, 1, 2}, f"`ignore_label` cannot be 0, 1 or 2."
    label_anchor, label_pos, label_neg = 2, 1, 0
    label_counter = Counter(token.koref for token in tokens)
    for i, anchor_label in enumerate(anchor.koref for anchor in tokens):
        if anchor_label == ignore_label or label_counter[anchor_label] == 1:
            continue
        triplet_data = []
        for j, token in enumerate(tokens):
            label = token.koref
            new_token = deepcopy(token)
            if label == ignore_label:
                pass
            elif label != anchor_label:
                new_token.koref = label_neg
            elif j == i:
                new_token.koref = label_anchor
            else:
                new_token.koref = label_pos
            triplet_data.append(new_token)
        yield triplet_data


def collect_data_points(
        lines: Iterable[str],
        input_ignore_label: str = "_",
        output_ignore_label: int = -1,
        sep: str = "\t"
) -> Generator[List[XtsvToken], None, None]:
    """Collect data points (subsequences of tokens)
    from an iterable of tsv lines.

    Args:
        lines: An iterable of lines with `'\n'` at the end.
        input_ignore_label: The label for tokens out of co-reference
            clusters in the input. Defaults to `_`.
        output_ignore_label: The label for tokens out of co-reference
            clusters in the output. Defaults to `-1`.
        sep: The field separator character in the input lines.
            Defaults to `'\t'`.

    Returns:
        A generator of sequences of token objects.
    """
    is_last_line_empty = False
    document = []
    for line in map(str.strip, lines):
        if len(line) != 0:
            wordform, label, *ana = line.split(sep)
            label = output_ignore_label if label == input_ignore_label \
                else int(label)
            document.append(XtsvToken(wordform, label, *ana))
            is_last_line_empty = False
        else:
            if is_last_line_empty and len(document) != 0:
                yield document
                document = []
            else:
                is_last_line_empty = True
    if len(document) != 0:
        yield document


def tokens2dict(tokens: Iterable[XtsvToken]) -> Dict[str, Union[str, List[int]]]:
    """Create a dict with fields `text` and `labels` from
    an iterable of tokens.
    """
    collected_fields = {data_field.name: [] for data_field in fields(XtsvToken)}
    for token in tokens:
        for k, v in token.to_dict().items():
            collected_fields[k].append(v)
    collected_fields["labels"] = collected_fields.pop("koref")
    collected_fields["text"] = " ".join(collected_fields.pop("form"))
    return collected_fields


def main() -> None:
    """Main function"""
    args = get_conll2jsonl_args()
    token_seqs = collect_data_points(
        lines=args.input_file,
        input_ignore_label=args.input_ignore_label,
        output_ignore_label=args.output_ignore_label
    )
    if args.create_triplets:
        token_seqs = chain(
            *(create_triplets(token_seq, args.output_ignore_label)
              for token_seq in token_seqs)
        )
    writer = JsonlWriter(args.output_file)
    for token_seq in token_seqs:
        writer.write(tokens2dict(token_seq))


if __name__ == "__main__":
    main()
