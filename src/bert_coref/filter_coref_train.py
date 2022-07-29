#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a model for coreference labeling. This is understood as a binary
classification task, where the label `0` or `1` is assigned to each token.

`0` means that the given token is not part of any coreference cluster.
`1` means that the given token is the part of a coreference cluster.

"""

from argparse import Namespace
from typing import Dict, Union, List

from datasets import Dataset as HgfDataset
from transformers import TFBertForTokenClassification

from bert_coref import (
    get_deepl_parser,
    prepare_dataset_split,
    run_training,
    wandb_setup,
    IgnorantBinaryCrossentropy,
    IgnorantBinaryMCC,
    IgnorantPrecision,
)


def get_classification_args() -> Namespace:
    """Get command line arguments for binary classification."""
    parser = get_deepl_parser(
        description="Train BERT to classify tokens into 2 classes based on "
                    "the fact if they are parts of coreference clusters.")
    parser.add_argument("--freeze-bert", dest="freeze_bert",
                        action="store_true", help="Freeze the Transformer "
                                                  "layers.")
    return parser.parse_args()


def relabel_to_binary(
        dataset: HgfDataset,
        label_column: str,
        zero_label: int
) -> HgfDataset:
    """Transform a dataset: turn every `zero_label` to `0`,
    replace all other labels with `1`.

    Args:
        dataset: A dataset that is to be transformed.
        label_column: The name of the dataset column (key) that
            contains the labels.
        zero_label: The label that will be replaced with `0`.

    Returns:
        The transformed dataset.
    """

    def relabel_func(
            example: Dict[str, Union[str, List[int]]]
    ) -> Dict[str, Union[str, List[int]]]:
        example[label_column] = [int(label != zero_label)
                                 for label in example[label_column]]
        return example

    return dataset.map(relabel_func)


def main() -> None:
    """Main function."""
    # Prepare the data
    args = get_classification_args()
    label_column, ignore_label = "labels", -1
    data_processing_kwargs = {
        "batch_size": args.batch_size,
        "shuffling_buffer_size": args.shuffling_buffer_size,
        "tokenizer": args.tokenizer,
        "label_column": label_column,
        "ignore_label": ignore_label,
        "text_column": "text"
    }
    train_dataset = relabel_to_binary(
        dataset=args.train_dataset,
        label_column=label_column,
        zero_label=ignore_label
    )
    train_dataset, train_length = prepare_dataset_split(
        dataset=train_dataset, **data_processing_kwargs)
    train_dataset = train_dataset.prefetch(2)
    val_dataset = relabel_to_binary(
        dataset=args.val_dataset,
        label_column=label_column,
        zero_label=ignore_label
    )
    val_dataset, _ = prepare_dataset_split(
        dataset=val_dataset, **data_processing_kwargs)

    # Train the model
    wandb_setup(vars(args))
    _ = run_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_length=train_length,
        model_init_fn=lambda: TFBertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=args.model, num_labels=1),
        loss_init_fn=lambda: IgnorantBinaryCrossentropy(
            ignore_label=ignore_label),
        log_update_freq=args.log_update_freq,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        decay_alpha=args.decay_alpha,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epochs=args.epochs,
        metric_init_fn=lambda: [IgnorantBinaryMCC(ignore_label),
                                IgnorantPrecision(ignore_label)],
        freeze_main_layer=args.freeze_bert
    )


if __name__ == "__main__":
    main()
