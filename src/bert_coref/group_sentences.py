#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rearrange sentences within a document.

The input is a `tsv` document without a header, where each line
corresponds to a token and sentences are separated by blank lines.

The output depends on the `max_length` parameter. This is the
maximal length of a document in BERT subword tokens. If a document
is longer than `max_length`, then it will be split into sub-documents
separated by double blank lines. Each sub-document in the output has
the same format as the input.

"""

from argparse import Namespace
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Deque, Generator, IO, Iterable, Optional, Union, Dict

from transformers import PreTrainedTokenizer

from bert_coref import (
    check_positive_int,
    check_char,
    get_tokenizer,
    get_io_parser
)


def get_sent_group_args() -> Namespace:
    """Get command line arguments."""
    parser = get_io_parser(
        description="Specify arguments to split documents into sub-documents.")
    parser.add_argument("--tokenizer", default="SZTAKI-HLT/hubert-base-cc",
                        type=get_tokenizer,
                        help="Tokenizer name or path. "
                             "Defaults to `'SZTAKI-HLT/hubert-base-cc'`.")
    parser.add_argument("--max-length", dest="max_length",
                        type=check_positive_int, default=510,
                        help="The maximal document length in subword tokens. "
                             "Defaults to `510`.")
    parser.add_argument("--subst-label", dest="subst_label", default="_",
                        help="The label with which dangling co-reference "
                             "labels will be substituted. This takes effect only if "
                             "the `--remove-dangling` flag is used. Defaults to `'_'`.")
    parser.add_argument("--remove-dangling", dest="remove_dangling",
                        action="store_true",
                        help="Specify this flag to remove coreference labels "
                             "each of which belongs to one token only.")
    parser.add_argument("--input-separator", dest="input_separator",
                        type=check_char, default="\t",
                        help="The input field separator character. "
                             "Defaults to `'\\t'`.")
    parser.add_argument("--output-separator", dest="output_separator",
                        default="\t", type=check_char,
                        help="The output field separator character. "
                             "Defaults to `'\\t'`.")
    return parser.parse_args()


@dataclass
class XtsvToken:
    """A class for tokens annotated with the `xtsv` fields."""
    form: str = field(metadata={"help": "The surface (word) form."})
    koref: Optional[Union[str, int]] = field(default=None, metadata={"help": "A coreference label."})
    anas: Optional[Union[str, List[Dict[str, str]]]] = field(
        default=None,
        metadata={"help": "Morphemic and POS features."}
    )
    lemma: Optional[str] = field(default=None, metadata={"help": "The normal form of the token."})
    xpostag: Optional[str] = field(default=None, metadata={"help": "The `emmorph` POS tag."})
    upostag: Optional[str] = field(default=None, metadata={"help": "The `UD` POS tag."})
    feats: Optional[str] = field(default=None, metadata={"help": "Morphological features."})
    id: Optional[Union[str, int]] = field(default=None, metadata={"help": "The token ID in the sentence."})
    deprel: Optional[str] = field(default=None, metadata={"help": "Dependency relation tag."})
    head: Optional[Union[str, int]] = field(
        default=None,
        metadata={"help": "The ID of the token on which the current token depends."}
    )
    cons: Optional[str] = field(default=None, metadata={"help": "Syntactic constituent tag."})

    def __post_init__(self) -> None:
        """Convert the field values to the types corresponding to the fields."""
        if isinstance(self.koref, str):
            try:
                self.koref = int(self.koref)
            except ValueError:
                pass
        try:
            if isinstance(self.id, str):
                self.id = int(self.id)
            if isinstance(self.head, str):
                self.head = int(self.head)
        except ValueError:
            pass

    def to_string(self, sep: str = "\t") -> str:
        """Get the `xtsv`-style string representation of the token."""
        return sep.join(str(value) for value in self.__dict__.values())

    def to_dict(self) -> Dict[str, Union[int, str, None]]:
        """Return the data as a `dict`."""
        return self.__dict__


class SubDocumentCollector:
    """A class that collects sentences until
    the maximal document length is reached.

    Sentences are lists of `XtsvToken` instances. A `SubDocumentCollector`
    uses a tokenizer to monitor the total length of the text data stored.
    A new sentence can only be added if the total text length does not
    exceed the predefined maximal length.

    Before returning a group of sentences, dangling token labels can be
    (optionally) replaced with a substitution label. A dangling label is a
    co-reference label that is unmatched within a group of sentences.
    """

    def __init__(
            self,
            max_length: int,
            tokenizer: PreTrainedTokenizer,
            init_data: Optional[Iterable[List[XtsvToken]]] = None,
            subst_label: str = "_"
    ) -> None:
        """Initialize the object.

        Args:
            max_length: The maximal text length in subword tokens.
            tokenizer: The pre-trained tokenizer to use.
            init_data: Optional. Data that will be immediately added
                to the stack, an iterable of sentences. Sentences are
                expected to be represented as lists of `XtsvTokens`.
            subst_label: Substitution label with which dangling labels in the
                stack can be replaced. Defaults to `'_'`.
        """
        self.max_length = max_length
        self.subst_label = subst_label
        self._tokenizer = tokenizer
        if init_data is None:
            init_data = []
        self._data = deque(init_data)
        self._length = sum(self._calculate_sentence_length(sentence)
                           for sentence in self._data)
        if self._length > self.max_length:
            raise OverflowError(
                f"The stack size ({self._length}) is larger than the allowed "
                f"maximal length ({self.max_length}).")

    def append(self, sentence: List[XtsvToken]) -> bool:
        """Append a new sentence to the text data stack.
        This will not be performed if the length of the full text
        data with the added new sentence exceeds the maximal length.

        Args:
            sentence: A sentence as a list of tokens.

        Returns:
            A Boolean value that indicates whether adding the new sentence
            was successful.
        """
        new_length = self._length + self._calculate_sentence_length(sentence)
        is_appendable = new_length <= self.max_length
        if is_appendable:
            self._data.append(sentence)
            self._length = new_length
        return is_appendable

    def _calculate_sentence_length(self, sentence: Iterable[XtsvToken]) -> int:
        """Get the length of a sentence in subword tokens."""
        sentence_str = " ".join(token.form for token in sentence)
        return len(self._tokenizer.encode(
            sentence_str, add_special_tokens=False))

    def pop_all(self, remove_dangling: bool = False) -> Deque[List[XtsvToken]]:
        """Return the sentences and clear the stack.

        Args:
            remove_dangling: Replace dangling labels with the substitution
                label before returning the data. Defaults to `False`.
        """
        if remove_dangling:
            self._remove_dangling_labels()
        sentences = self._data
        self._data = deque()
        self._length = 0
        return sentences

    def _remove_dangling_labels(self) -> None:
        """Remove token labels that are not matched within a
        group of sentences.
        """
        label_counts = Counter(
            token.coref for sentence in self._data for token in sentence)
        for sentence in self._data:
            for token in sentence:
                if label_counts[token.coref] == 1:
                    token.coref = self.subst_label

    def __len__(self) -> int:
        """Get the total text length in subword tokens."""
        return self._length

    def __repr__(self) -> str:
        """Get object representation."""
        return f"{SubDocumentCollector.__name__}(data={self._data}, " \
               f"length={self._length})"


def collect_sentences(
        lines: Iterable[str],
        sep: str = "\t"
) -> Generator[List[XtsvToken], None, None]:
    """Collect tokens into sentences.

    Args:
        lines: Lines of a `tsv` file, where each line represents a token and
            sentences are separated by blank lines.
        sep: Field separator in the `tsv` lines. Defaults to `'\t'`.

    Returns:
        A generator that yields sentences as lists of tokens.
    """
    tokens = []
    for line in map(str.strip, lines):
        if line:
            tokens.append(XtsvToken(*line.split(sep)))
        elif len(tokens) != 0:
            yield tokens
            tokens = []
    if len(tokens) != 0:
        yield tokens


def print_sentences(
        sentences: Iterable[List[XtsvToken]],
        target: IO,
        sep: str = "\t"
) -> None:
    """Print a group of sentences.

    Args:
        sentences: A group of sentences as an iterable of lists of tokens.
        target: The output file where the sentences will be written.
        sep: The output field separator. Defaults to `'\t'`.
    """
    newline = "\n"
    for sentence in sentences:
        lines = (token.to_string(sep=sep) + newline for token in sentence)
        target.writelines(lines)
        target.write(newline)
    target.write(newline)


def main() -> None:
    """Main function."""
    args = get_sent_group_args()
    output_file = args.output_file
    output_separator = args.output_separator
    remove_dangling = args.remove_dangling
    doc_collector = SubDocumentCollector(
        args.max_length, args.tokenizer, subst_label=args.subst_label)
    for sentence in collect_sentences(args.input_file, args.input_separator):
        if not doc_collector.append(sentence) and len(doc_collector) != 0:
            print_sentences(
                sentences=doc_collector.pop_all(remove_dangling=remove_dangling),
                target=output_file,
                sep=output_separator
            )
            doc_collector.append(sentence)
    if len(doc_collector) != 0:
        print_sentences(
            sentences=doc_collector.pop_all(remove_dangling=remove_dangling),
            target=output_file,
            sep=output_separator
        )


if __name__ == "__main__":
    main()
