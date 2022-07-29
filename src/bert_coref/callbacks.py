# -*- coding: utf-8 -*-

"""A module for custom Keras callbacks. These are:
    1. A checkpoint callback,
    2. A callback for monitoring the learning rate.

"""

import pickle
from datetime import datetime
from os import mkdir
from os.path import exists, join as os_path_join
from typing import Optional, Dict, Any

import wandb
from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras import backend as keras_backend


class CheckpointCallback(Callback):
    """A callback to make model and optimizer checkpoints.
    This callback can only be used with a `TFPreTrainedModel`!
    """

    def __init__(
            self,
            checkpoint_root: str,
            save_freq: Optional[int] = None
    ) -> None:
        """Initialize the callback.

        Args:
            checkpoint_root: Path to the checkpoint root directory.
            save_freq: Specifies how often to make a checkpoint: if an integer
                is given, it means the number of iterations. Passing `None`
                means making a checkpoint after each epoch.
        """
        super(CheckpointCallback, self).__init__()
        if not exists(checkpoint_root):
            mkdir(checkpoint_root)
        subdir = f"run_{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}"
        self.checkpoint_dir = os_path_join(checkpoint_root, subdir)
        mkdir(self.checkpoint_dir)
        self.save_freq = save_freq

    def on_train_batch_end(
            self,
            batch: int,
            logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save the model after `save_freq` iterations, if `save_freq`
        is an integer.
        """
        if self.save_freq is not None:
            iteration = self.model.optimizer.iterations
            if iteration % self.save_freq == 0 and iteration != 0:
                step_dir = os_path_join(
                    self.checkpoint_dir, f"step_{iteration}")
                self.model.save_pretrained(step_dir)
                self._save_optimizer(step_dir)

    def on_epoch_end(
            self,
            epoch: int,
            logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save the model after each epoch if `save_freq` is
        specified as `'epoch'`.
        """
        if self.save_freq is None:
            epoch_dir = os_path_join(self.checkpoint_dir, f"epoch_{epoch}")
            self.model.save_pretrained(epoch_dir)
            self._save_optimizer(epoch_dir)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Create a checkpoint when the training is complete."""
        if self.save_freq is not None and \
                self.model.optimizer.iterations % self.save_freq != 0:
            end_dir = os_path_join(self.checkpoint_dir, "train_end")
            self.model.save_pretrained(end_dir)
            self._save_optimizer(end_dir)

    def _save_optimizer(self, dir_name: str) -> None:
        """A method to save the optimizer states."""
        symbolic_weights = getattr(self.model.optimizer, "weights")
        weight_values = keras_backend.batch_get_value(symbolic_weights)
        with open(os_path_join(dir_name, "optimizer.pkl"), "wb") as optim_file:
            pickle.dump(weight_values, optim_file)


class LrLogCallback(Callback):
    """A callback to monitor the learning rate controlled by a scheduler."""

    def __init__(self, log_update_freq: int) -> None:
        """Initialize the callback.

        Args:
            log_update_freq: Specifies how often to report the learning rate
                to `WandB` (number of iterations).
        """
        super(LrLogCallback, self).__init__()
        self.log_update_freq = log_update_freq

    def on_train_batch_end(
            self,
            batch: int,
            logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Get the actual learning rate and report it after the steps
        specified by `log_update_freq`.
        """
        iteration = self.model.optimizer.iterations
        learning_rate = keras_backend.get_value(
            self.model.optimizer.lr(iteration))
        if iteration % self.log_update_freq == 0:
            wandb.log({"learning_rate": learning_rate}, commit=False)


class FreezeCallback(Callback):
    """A callback that freezes the transformer layers.
    This works only with `HuggingFace` TensorFlow models.
    """

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Freeze the transformer layers."""
        self.model.bert.trainable = False
