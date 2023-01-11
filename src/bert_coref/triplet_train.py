#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a model for coreference resolution with cluster-based triplet loss.

This is achieved through the following steps:
    1. Tokenize a dataset
    2. Load a pre-trained model
    3. Train the model using the triplet loss.
"""

from argparse import Namespace
from math import ceil
from typing import Dict, List, Union, Optional, Tuple, Generator, Callable

import tensorflow as tf
import wandb
from datasets import Dataset as HgfDataset
from tensorflow.data import Dataset as TFDataset
from tensorflow.keras import Model as KModel
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow_addons.optimizers import AdamW
from transformers import PreTrainedTokenizer, TFBertModel
from wandb.keras import WandbCallback

from bert_coref import (
    get_deepl_parser,
    check_non_negative_float,
    CheckpointCallback,
    LrLogCallback,
    FreezeCallback,
    ClusterTripletLoss
)


def get_training_args() -> Namespace:
    """Get command line arguments for training."""
    parser = get_deepl_parser(
        description="Train a BERT model for coreference resolution")
    parser.add_argument("--margin", type=check_non_negative_float, default=0.5,
                        help="Triplet loss margin. Defaults to `0.5`.")
    return parser.parse_args()


def tokenize_dataset(
        dataset: HgfDataset,
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        label_column: str,
        ignore_label: int,
        sep: str = " "
) -> HgfDataset:
    """Tokenize a dataset.

    Args:
        dataset: The dataset that is to be tokenized.
        tokenizer: A pre-trained tokenizer.
        text_column: The dataset column (or key) that contains the text data,
            e.g. `'text'`.
        label_column: The dataset column (or key) that contains the labels,
            e.g. `'labels'`.
        ignore_label: The label that will be assigned to the subword tokens in
            non-starting positions.
        sep: Token separator in the text. Defaults to `' '`.

    Returns:
        The tokenized dataset.
    """
    input_ids_name, attn_mask_name = "input_ids", "attention_mask"
    assert len({text_column, label_column}.intersection(
        {input_ids_name, attn_mask_name})) == 0, \
        f"The input dataset must not contain columns named {input_ids_name} " \
        f"or {attn_mask_name}."

    def tok_func(
            example: Dict[str, Union[str, List[int]]]
    ) -> Dict[str, List[int]]:
        tokens = [tokenizer.cls_token_id]
        labels = [ignore_label]
        for word, label in zip(
                example[text_column].split(sep), example[label_column]):
            subwords = tokenizer.encode(word, add_special_tokens=False)
            tokens.extend(subwords)
            labels.extend([label] + [ignore_label] * (len(subwords) - 1))
        tokens.append(tokenizer.sep_token_id)
        labels.append(ignore_label)
        example[label_column] = labels
        return {input_ids_name: tokens, attn_mask_name: [1] * len(tokens)}

    return dataset.map(tok_func, remove_columns=[text_column])


TensorDataStruct = Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]


def dataset2tf(dataset: HgfDataset, label_column: str) -> Tuple[TFDataset, int]:
    """Convert a tokenized `datasets.Dataset` object into
    a `TensorFlow` dataset.

    Args:
        dataset: A tokenized dataset. It is required to have columns
            `'input_ids'` and `'attention_mask'`.
        label_column: The dataset column (key) that contains the labels.

    Returns:
        The `TensorFlow` dataset and the dataset length.
    """
    input_features = ("input_ids", "attention_mask")
    dataset_columns = dataset.features.keys()
    assert set(input_features).issubset(dataset_columns), \
        "The dataset must contain columns `'input_ids'` and `'attention_mask'`."
    dataset_length = len(dataset)
    tensor_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32)

    def data_generator() -> Generator[TensorDataStruct, None, None]:
        for example in dataset:
            inputs = tuple(tf.convert_to_tensor(example[feature])
                           for feature in input_features)
            labels = tf.convert_to_tensor(example[label_column])
            yield inputs, labels

    tf_dataset = TFDataset.from_generator(
        data_generator,
        output_signature=((tensor_spec, tensor_spec), tensor_spec)
    )
    return tf_dataset, dataset_length


def prepare_dataset_split(
        dataset: HgfDataset,
        batch_size: int,
        shuffling_buffer_size: Optional[int] = None,
        **kwargs
) -> Tuple[TFDataset, int]:
    """Prepare a dataset:
    tokenize and convert to TensorFlow format.

    Args:
        dataset: A dataset split.
        batch_size: The global batch size.
        shuffling_buffer_size: The buffer size for data shuffling. Optional.
        **kwargs: Tokenization arguments (for the `tokenize_dataset` function,
        apart from the `dataset` argument).

    Returns:
        The dataset split and its size (number of batches per epoch).
    """
    pad_id = kwargs["tokenizer"].pad_token_id
    label_column, ignore_label = kwargs["label_column"], kwargs["ignore_label"]
    dataset = tokenize_dataset(dataset=dataset, **kwargs)
    dataset, dataset_length = dataset2tf(dataset, label_column=label_column)
    if shuffling_buffer_size is not None:
        dataset = dataset.shuffle(shuffling_buffer_size)
    dataset = dataset.padded_batch(
        batch_size, padding_values=((pad_id, pad_id), ignore_label))
    dataset_length = ceil(dataset_length / batch_size)
    return dataset, dataset_length


def select_strategy() -> tf.distribute.Strategy:
    """Select distribution strategy based on the available GPUs."""
    num_gpus = len(tf.config.list_physical_devices("GPU"))
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    return strategy


def wandb_setup(config: Dict[str, Union[str, int, float]]) -> None:
    """Configure the WandB run. The project and run names are expected to be
    comprised in `config`.
    """
    project = config.pop("project", None)
    run_name = config.pop("run_name", None)
    wandb.init(config=config, name=run_name, project=project)


class TFCorefBert(TFBertModel):
    """A model class to correctly handle the input data."""

    def train_step(
            self,
            data: TensorDataStruct
    ) -> Dict[str, Union[tf.Tensor, float]]:
        """A custom training step implementation."""
        inputs, y_true = data
        with tf.GradientTape() as tape:
            # noinspection PyCallingNonCallable
            y_pred = self(inputs, training=True).last_hidden_state
            loss = self.compiled_loss(
                y_true, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(
        self,
        data: TensorDataStruct
    ) -> Dict[str, Union[tf.Tensor, float]]:
        """Customize the test step."""
        inputs, y_true = data
        # noinspection PyCallingNonCallable
        y_pred = self(inputs, training=False).last_hidden_state
        _ = self.compiled_loss(
            y_true, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}


def run_training(
        train_dataset: TFDataset,
        val_dataset: TFDataset, *,
        train_length: int,
        model_init_fn: Callable[[], KModel],
        loss_init_fn: Callable[[], Loss],
        checkpoint_dir: str,
        save_freq: int,
        log_update_freq: int,
        epochs: int,
        learning_rate: float,
        decay_alpha: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        metric_init_fn: Optional[Callable[[], List[Metric]]] = None,
        freeze_main_layer: bool = False
) -> tf.keras.callbacks.History:
    """A high level function to start training.
    It is assumed that `wandb.init()` has already been called.
    """
    # Define the callbacks.
    callbacks = [
        CheckpointCallback(
            checkpoint_root=checkpoint_dir,
            save_freq=save_freq
        ),
        LrLogCallback(log_update_freq=log_update_freq),
        WandbCallback(
            save_model=False,
            predictions=100,
            log_batch_frequency=log_update_freq
        ),
    ]
    if freeze_main_layer:
        callbacks.insert(0, FreezeCallback())

    # Train the model
    strategy = select_strategy()
    num_steps = epochs * train_length
    with strategy.scope():
        scheduler = CosineDecay(
            initial_learning_rate=learning_rate,
            alpha=decay_alpha,
            decay_steps=num_steps
        )
        optimizer = AdamW(
            learning_rate=scheduler,
            weight_decay=weight_decay,
            beta_1=beta1,
            beta_2=beta2
        )
        model = model_init_fn()
        metrics = metric_init_fn() if metric_init_fn is not None else None
        model.compile(
            loss=loss_init_fn(),
            optimizer=optimizer,
            metrics=metrics
        )
    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def main() -> None:
    """Main function."""
    args = get_training_args()
    wandb_setup(vars(args))
    tf.random.set_seed(args.seed)  # Set global TensorFlow seed

    # Prepare the dataset splits
    data_processing_kwargs = {
        "batch_size": args.batch_size,
        "shuffling_buffer_size": args.shuffling_buffer_size,
        "tokenizer": args.tokenizer,
        "text_column": "text",
        "label_column": "labels",
        "ignore_label": -1
    }
    train_dataset, train_length = prepare_dataset_split(
        dataset=args.train_dataset, **data_processing_kwargs)
    val_dataset, _ = prepare_dataset_split(
        dataset=args.val_dataset, **data_processing_kwargs)
    train_dataset = train_dataset.prefetch(2)

    # Train the model
    _ = run_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_length=train_length,
        model_init_fn=lambda: TFCorefBert.from_pretrained(args.model),
        loss_init_fn=lambda: ClusterTripletLoss(margin=args.margin),
        log_update_freq=args.log_update_freq,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        decay_alpha=args.decay_alpha,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
