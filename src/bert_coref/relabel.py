#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A script to relabel inputs based on morphological and dependency features.

A token will be labelled non-coreferent iff one of the following conditions hold:
    1. Its POS tag is an element of a stop list;
    2. It was originally labelled as non-coreferent and it depends on a coreferent token.

Tokens that do not fulfill the requirements above but were originally labelled as
non-coreferent will be considered elements of single-token coreference clusters and
cluster IDs will be assigned to them.

It is expected that the input is in `xtsv` format, with blank lines separating
sentences and double blank lines separating documents.
"""

from argparse import Namespace, FileType
from dataclasses import dataclass, field
from typing import (
    Union, List, Sequence,
    Iterable, Generator, Set, FrozenSet
)

from bert_coref import get_io_parser, XtsvToken


def get_relabelling_args() -> Namespace:
    """Get command line arguments to relabel the input."""
    parser = get_io_parser("Specify arguments to relabel your data")
    parser.add_argument("--ignore-label", dest="ignore_label", default="_",
                        help="The label that indicates non-coreferent tokens in the input. "
                             "Defaults to `_`.")
    parser.add_argument("--tag-stop-list", dest="tag_stop_list",
                        type=FileType("r", encoding="utf-8"),
                        help="A file that lists POS tags with which coreferent tokens "
                             "are never tagged (one tag per line). Optional.")
    parser.add_argument("--subtag-stop-list", dest="subtag_stop_list",
                        type=FileType("r", encoding="utf-8"),
                        help="A file that lists POS tag substrings. A token annotated with a tag "
                             "containing any of these substrings will never be considered coreferent. "
                             "Each line should contain one subtag. Optional.")
    parser.add_argument("--skip-header", dest="skip_header", action="store_true",
                        help="Specify this flag if the input starts with a header line.")
    parser.add_argument("--exclude-descendants", dest="exclude_descendants",
                        action="store_true",
                        help="Specify this flag if you do not want any of the non-coreferent tokens "
                             "depending on a coreferent head to get a unique coref label.")
    return parser.parse_args()


@dataclass
class LabelValue:
    """A dataclass that contains a single integer value."""
    value: int = field(
        default=-1,
        metadata={"help": "An integer value that can be interpreted as a label."}
    )


def collect_documents(lines: Iterable[str]) -> Generator[List[str], None, None]:
    """Collect documents as lists of strings.

    Args:
        lines: The input text lines where double blank lines
            are used to separate documents.

    Returns:
        A generator the yields documents as lists of text lines.
    """
    document = []
    is_prev_blank = False
    for line in lines:
        is_blank = len(line.strip()) == 0
        if all((is_prev_blank, is_blank, len(document) != 0)):
            yield document
            document = []
        elif not (is_blank and is_prev_blank):
            document.append(line)
        is_prev_blank = is_blank
    if len(document) != 0:
        yield document


def collect_sentences(lines: Iterable[str], sep: str = "\t") -> Generator[List[XtsvToken], None, None]:
    """Collect sentences as lists of tokens from the input.

    Args:
        lines: The input strings.
        sep: Field separator in the input. Defaults to `'\\t'`.

    Returns:
        A generator that yields lists of annotated tokens.
    """
    sentence = []
    for line in map(str.strip, lines):
        if len(line) != 0:
            sentence.append(XtsvToken(*line.split(sep)))
        elif len(sentence) != 0:
            yield sentence
            sentence = []
    if len(sentence) != 0:
        yield sentence


def update_max_label(
        token: XtsvToken,
        max_label: LabelValue
) -> XtsvToken:
    """Update the maximal `koref` label value.

    Args:
        token: The annotated input token.
        max_label: Ab object that stores the actual maximal label.
            It will be updated by this function.

    Returns:
         The unaltered input token.
    """
    koref_label = token.koref
    if isinstance(koref_label, int) and koref_label > max_label.value:
        max_label.value = koref_label
    return token


def add_single_labels(
        tokens: Iterable[XtsvToken],
        min_new_label: LabelValue,
        ignore_label: int
) -> Generator[XtsvToken, None, None]:
    """Replace the `koref` labels of non-coreferent tokens with unique integer labels.

    Args:
        tokens: The annotated input tokens.
        min_new_label: The minimal value that can be used as a new label.
        ignore_label: The original label in the input that will be replaced.

    Returns:
        A generator that yields the input tokens whose `koref` labels can be modified.
    """
    for token in tokens:
        if token.koref == ignore_label:
            token.koref = min_new_label.value
            min_new_label.value += 1
        yield token


def exclude_stop_tags(
        tokens: Iterable[XtsvToken],
        replacement_label: str,
        stop_tags: Union[FrozenSet[str], Set[str]],
        stop_subtags: Sequence[str]
) -> Generator[XtsvToken, None, None]:
    """Change the `koref` label of a token (or leave it unaltered) based on its POS tag.

    Args:
        tokens: The annotated input tokens.
        replacement_label: The label with which the `koref` label
            of the selected tokens will be replaced.
        stop_tags: A set of POS tags. If any of these tags is assigned to a token,
            the `koref` label of the token will be set to `replacement_label`.
        stop_subtags: A sequence of POS tag substrings. If a tag containing any of
            these substrings is assigned to a token, the `koref` label of the token
            will be set to `replacement_label`.

    Returns:
        A generator that yields the input tokens with modified or unaltered `koref` labels.
    """
    for token in tokens:
        pos_label = token.xpostag
        if pos_label in stop_tags or any((stop_subtag in pos_label) for stop_subtag in stop_subtags):
            token.koref = replacement_label
        yield token


def exclude_descendants(
        tokens: Iterable[XtsvToken],
        ignore_label: str,
        replacement_label: str,
        coreferent_ids: Union[Set[int], FrozenSet[int]]
) -> Generator[XtsvToken, None, None]:
    """Modify the `koref` labels based on dependency relations:

    If the value of the `koref` field of a token `t` is `ignore_label` and
    The parent of the token in the dependency tree is another token whose
    `koref` value is not `ignore_label`, than change the `koref` value of `t`
    to `replacement_label`.

    Args:
        tokens: The input tokens whose `koref` labels can be modified.
        ignore_label: The original `koref` label of non-coreferent tokens.
        replacement_label: The new `koref` label of the tokens selected
            by the criteria given above.
        coreferent_ids: The IDs of the coreferent input tokens. This means that
            their `koref` values are not equal to `ignore_label` or `replacement_label`.

    Returns:
        A generator that yields all the tokens (both modified and unaltered).
    """
    for token in tokens:
        if token.koref == ignore_label and token.head in coreferent_ids:
            token.koref = replacement_label
        yield token


def main() -> None:
    """Main function."""
    args = get_relabelling_args()
    ignore_label = args.ignore_label
    replacement_label = "-" if ignore_label == "_" else "_"
    input_file, output_file = args.input_file, args.output_file
    stop_tags = frozenset(stop_tag.strip() for stop_tag in args.tag_stop_list) \
        if args.tag_stop_list is not None else frozenset()
    stop_subtags = tuple(stop_subtag.strip() for stop_subtag in args.subtag_stop_list) \
        if args.subtag_stop_list is not None else tuple()

    if args.skip_header:
        print(next(input_file), file=output_file, end="")
    for document in collect_documents(input_file):
        min_new_label = LabelValue()
        sentences = []

        for tokens in collect_sentences(document):
            coreferent_ids = frozenset(token.id for token in tokens if token.koref != ignore_label)
            tokens = exclude_stop_tags(
                tokens=tokens,
                replacement_label=replacement_label,
                stop_tags=stop_tags,
                stop_subtags=stop_subtags
            )
            if args.exclude_descendants:
                tokens = exclude_descendants(
                    tokens=tokens,
                    ignore_label=ignore_label,
                    replacement_label=replacement_label,
                    coreferent_ids=coreferent_ids
                )
            sentences.append(list(update_max_label(token, min_new_label) for token in tokens))
        min_new_label.value += 1
        document.clear()

        for sentence in sentences:
            sentence = add_single_labels(sentence, min_new_label, ignore_label=ignore_label)
            for token in sentence:
                if token.koref == replacement_label:
                    token.koref = ignore_label
                print(token.to_string(), file=output_file)
            print(file=output_file)
        print(file=output_file)


if __name__ == "__main__":
    main()
