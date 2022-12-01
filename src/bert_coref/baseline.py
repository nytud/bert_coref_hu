#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script to call the baseline model.

The input is a file with a header and xtsv-style token annotation
that can contain multiple documents separated by DOUBLE blank lines.

The output is a fully annotated xtsv file (each token has 12 annotation
fields with possible empty values). The `koref` field will be set by
the baseline.
"""

from argparse import Namespace
from dataclasses import fields
from collections import Counter
from typing import Iterable, Generator, List, IO, Tuple

from bert_coref import XtsvToken, get_io_parser


def get_baseline_args() -> Namespace:
    """Get command line arguments to use the baseline model."""
    parser = get_io_parser(description="A script to call the baseline model.")
    return parser.parse_args()


def get_annot_field_names(header: str, delimiter: str = "\t") -> Tuple[str, ...]:
    """Get the order of the annotation field names from a header line.

    Args:
        header: A single xtsv header line.
        delimiter: The field delimiter. Defaults to `'\t'`.

    Returns:
        A tuple of token annotation field names.
    """
    return tuple(header.strip().lower().split(delimiter))


def read_docs(
        corp: Iterable[str],
        annot_fields: Tuple[str, ...],
        delimiter: str = "\t"
) -> Generator[List[XtsvToken], None, None]:
    """Read the input line by line. Collect the tokens within each document
    into lists.

    Args:
        corp: The input as an iterable of xtsv lines.
        annot_fields: The names of the input fields. They must be valid xtsv
            token annotation field names.
        delimiter: The field delimiter int the input. Defaults to `'\t'`,

    Returns:
        A generator of lists of tokens. Each list represents the
        tokens of a document.
    """
    assert set(annot_fields).issubset({f.name for f in fields(XtsvToken)}), \
        f"Invalid annotation field names: {annot_fields}"
    is_prev_blank = False
    doc_tokens = []

    for line in map(str.strip, corp):
        if len(line) == 0:
            if is_prev_blank:
                yield doc_tokens
                doc_tokens = []
            else:
                doc_tokens.append(XtsvToken(""))
                is_prev_blank = True
        else:
            annot_values = line.split(delimiter)
            assert len(annot_values) == len(annot_fields), f"Invalid annotation in {line}"
            doc_tokens.append(XtsvToken(
                **{annot_field: annot_value for annot_field, annot_value
                   in zip(annot_fields, annot_values)}
            ))
            is_prev_blank = False

    if len(doc_tokens) != 0:
        yield doc_tokens


def _find_coref(doc: List[XtsvToken]) -> None:
    """Apply the rule-based coreference resolution baseline
    to a document. This is a SIDE EFFECT function! The `koref`
    field of the input tokens will be modified.

    Args:
        doc: The input document represented as a list of
            xtsv tokens. It may contain tokens whose form
            field stores an empty string. Such tokens will be
            interpreted as blank lines separating sentences.
    """
    noun_tags = {"NOUN", "PROPN"}
    coref_groups = {}
    coref_group_counter = Counter()

    for token in doc:
        if token.upostag in noun_tags:
            coref_group_counter[token.lemma] += 1
            if token.lemma not in coref_groups.keys():
                coref_groups[token.lemma] = len(coref_groups.keys()) + 1
            token.koref = coref_groups[token.lemma]
        else:
            token.koref = "_"

    # set `koref` to 0 for singletons
    for token in doc:
        if token.koref != "_" and coref_group_counter[token.lemma] == 1:
            token.koref = 0

def print_full_header(target: IO, delimiter: str = "\t") -> None:
    """Print a header that indicates the `XtsvToken` fields.

    Args:
        target: The file where the header will be printed.
        delimiter: Field delimiter in the output. Defaults to `'\t'.`
    """
    print(delimiter.join(f.name for f in fields(XtsvToken)), file=target)


def print_tokens(tokens: Iterable[XtsvToken], target: IO) -> None:
    """Print xtsv tokens.

    Args:
        tokens: The input xtsv tokens.
        target: The file where the tokens will be printed.
    """
    for token in tokens:
        line = token.to_string() if token.form != "" else ""
        print(line, file=target)


def main() -> None:
    """Main function."""
    args = get_baseline_args()
    input_file, output_file = args.input_file, args.output_file
    annot_fields = get_annot_field_names(next(input_file))

    print_full_header(target=output_file)
    for doc in read_docs(args.input_file, annot_fields=annot_fields):
        _find_coref(doc)
        print_tokens(doc, target=output_file)
        print(file=output_file)


if __name__ == "__main__":
    main()
