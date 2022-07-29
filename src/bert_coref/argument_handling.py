# -*- coding: utf-8 -*-

"""A module for argument handling utils."""

from sys import stdin, stdout
from argparse import ArgumentTypeError, ArgumentParser, FileType
from typing import Union, Dict
from pathlib import Path
from requests.exceptions import HTTPError

import numpy as np
from datasets import Dataset as HgfDataset, load_dataset
from transformers import AutoTokenizer
from huggingface_hub import list_repo_files


def get_io_parser(description: str) -> ArgumentParser:
    """Create an `ArgumentParser` with two positional arguments:
    input (file or `stdin`) and output (file or `stdout`).
    """
    parser = ArgumentParser(description=description)
    parser.add_argument("input_file", type=FileType("r", encoding="utf-8"),
                        nargs="?", default=stdin,
                        help="Path to the input file or `stdin`.")
    parser.add_argument("output_file", type=FileType("w", encoding="utf-8"),
                        nargs="?", default=stdout,
                        help="Path to the output file or `stdout`.")
    return parser


def get_deepl_parser(description: str) -> ArgumentParser:
    """Create an `ArgumentParser` with common deep learning arguments."""
    parser = ArgumentParser(description=description)
    parser.add_argument("train_dataset", type=get_jsonl_dataset,
                        help="Path to the `jsonlines` training dataset.")
    parser.add_argument("val_dataset", type=get_jsonl_dataset,
                        help="Path to the `jsonlines` validation dataset.")
    parser.add_argument("--model", required=True, type=check_model_path,
                        help="The name of the pretrained model "
                             "or a path to it.")
    parser.add_argument("--tokenizer", required=True, type=get_tokenizer,
                        help="The name of the pretrained tokenizer "
                             "or a path to it.")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir",
                        required=True, type=check_output_dir,
                        help="Path to the checkpoint root directory. "
                             "If it does not exist, it will be created.")
    parser.add_argument("--log-update-freq", dest="log_update_freq",
                        default=100, type=check_positive_int,
                        help="The number of training steps after which logs "
                             "will be reported. Defaults to `100`.")
    parser.add_argument("--save-freq", dest="save_freq",
                        type=check_positive_int,
                        help="The number of training steps after which a "
                             "checkpoint will be created. If not specified, "
                             "checkpoints will be created after each epoch.")
    parser.add_argument("--learning-rate", dest="learning_rate",
                        default=1e-5, type=check_positive_float,
                        help="The initial learning rate. Defaults to `1e-5`.")
    parser.add_argument("--decay-alpha", dest="decay_alpha",
                        default=0., type=check_non_negative_float,
                        help="Minimum learning rate value as a fraction of the "
                             "initial learning_rate. Defaults to `0.`.")
    parser.add_argument("--beta1", default=0.9, type=check_positive_float,
                        help="The `beta1` parameter of the `AdamW` optimizer. "
                             "Defaults to `0.9`.")
    parser.add_argument("--beta2", default=0.999, type=check_positive_float,
                        help="The `beta2` parameter of the `AdamW` optimizer. "
                             "Defaults to `0.999`.")
    parser.add_argument("--weight-decay", dest="weight_decay",
                        default=1e-5, type=check_non_negative_float,
                        help="The weight decay parameter of the `AdamW` "
                             "optimizer. Defaults to `1e-5`.")
    parser.add_argument("--batch-size", dest="batch_size",
                        default=32, type=check_positive_int,
                        help="The training and validation batch size. "
                             "Defaults to `32`.")
    parser.add_argument("--epochs", default=1, type=check_positive_int,
                        help="The number of training epochs. Defaults to `1`.")
    parser.add_argument("--shuffling-buffer-size", dest="shuffling_buffer_size",
                        type=check_positive_int, help="Buffer size for data "
                                                      "shuffling. Optional.")
    parser.add_argument("--project", help="WandB project name. Optional.")
    parser.add_argument("--run-name", dest="run_name",
                        help="WandB run name. Optional.")
    return parser


def check_positive_int(maybe_positive: Union[str, int, float]) -> int:
    """Check if the input is a positive integer."""
    maybe_positive = int(maybe_positive)
    if maybe_positive <= 0:
        raise ArgumentTypeError(
            f"Expected a positive integer, got {maybe_positive}.")
    return maybe_positive


def check_positive_float(maybe_positive: Union[str, int, float]) -> float:
    """Check if the input is a positive float."""
    maybe_positive = float(maybe_positive)
    if maybe_positive <= 0:
        raise ArgumentTypeError(
            f"Expected a positive float, got {maybe_positive}.")
    return maybe_positive


def check_non_negative_float(
        maybe_non_negative: Union[str, int, float]) -> float:
    """Check if the input is a positive float."""
    maybe_non_negative = float(maybe_non_negative)
    if maybe_non_negative < 0:
        raise ArgumentTypeError(
            f"Expected a non-negative float, got {maybe_non_negative}.")
    return maybe_non_negative


def check_char(maybe_char: str) -> str:
    """Check if the input is a single character."""
    if len(maybe_char) != 1:
        raise ArgumentTypeError(
            f"Expected a single character, got {maybe_char}.")
    return maybe_char


def check_output_dir(maybe_output_dir: str) -> str:
    """Check if the input is a correct output path,
    i.e. if the parent directory exists.
    """
    maybe_output_path = Path(maybe_output_dir)
    if not maybe_output_path.parent.exists():
        raise ArgumentTypeError(f"Invalid path: {maybe_output_dir}. The parent "
                                f"directory does not exist.")
    return maybe_output_dir


def check_model_path(model_path: str) -> str:
    """Check if the input is a valid model name or path."""
    if not Path(model_path).exists():
        try:
            _ = list_repo_files(model_path)
        except HTTPError:
            raise ArgumentTypeError(
                f"{model_path} is not a valid model path or name.")
    return model_path


def get_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    """Load a tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except EnvironmentError:
        raise ArgumentTypeError(
            f"{tokenizer_path} is not a valid tokenizer path or name.")
    return tokenizer


def get_jsonl_dataset(dataset_path) -> HgfDataset:
    """Load a dataset in jsonlines format."""
    if not Path(dataset_path).is_file():
        raise ArgumentTypeError(f"{dataset_path} is not a path to a file.")
    # `'train'` is a dummy value for `split`
    return load_dataset("json", data_files=[dataset_path], split="train")


def check_npz(maybe_npz: str) -> Dict[str, np.array]:
    """Check if the input is an `npz` file. If it is, load the arrays to a dict."""
    error_msg = f"{maybe_npz} is not a valid `npz` file."
    try:
        arrays = np.load(maybe_npz)
    except ValueError:
        raise ArgumentTypeError(error_msg)
    if not isinstance(arrays, np.lib.npyio.NpzFile):
        raise ArgumentTypeError(error_msg)
    return {key: arrays[key] for key in arrays}
