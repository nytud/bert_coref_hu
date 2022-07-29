#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test data loading for triplet training."""

import tensorflow as tf
from datasets import Dataset as HgfDataset
from transformers import BertTokenizer

from bert_coref import tokenize_dataset, dataset2tf, prepare_dataset_split


class TripletDataTest(tf.test.TestCase):
    """A test case to check data loading for triplet training."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fixture setup: get a tokenizer and a raw dataset."""
        super().setUpClass()
        cls._tokenizer = BertTokenizer.from_pretrained(
            "SZTAKI-HLT/hubert-base-cc")
        cls._text_column, cls._label_column = "text", "label"
        cls._ignore_label = -1
        cls._dataset = HgfDataset.from_dict({
            cls._text_column: ["A kutya , amelyik most ivott vizet , ugat .",
                               "A macska , amelyik most ivott vizet , nyávog .",
                               "A kutya a fára kergette a macskát , "
                               "aki onnan figyeli az ebet . "],
            cls._label_column: [[-1, 2, -1, 1, -1, -1, 0, -1, -1, -1],
                                [-1, 2, -1, 1, -1, -1, 0, -1, -1, -1],
                                [-1, 2, -1, -1, -1, -1, 0, -1, 0, -1,
                                 -1, -1, 1, -1]]
        })

    def test_tokenize_dataset(self) -> None:
        """Test dataset tokenization."""
        dataset = tokenize_dataset(
            dataset=self._dataset,
            tokenizer=self._tokenizer,
            text_column=self._text_column,
            label_column=self._label_column,
            ignore_label=self._ignore_label
        )
        num_tokens = len(self._dataset[self._text_column][0].split(" "))
        data_point = dataset[0]
        subwords = data_point["input_ids"]
        num_subwords = len(subwords)
        print(f"subwords: {subwords}")
        print(f"detokenized subwords: "
              f"{self._tokenizer.convert_ids_to_tokens(subwords)}")
        self.assertGreaterEqual(num_subwords, num_tokens)
        for ds_col in ("attention_mask", self._label_column):
            ds_col_value = data_point[ds_col]
            print(f"{ds_col}: {ds_col_value}")
            self.assertEqual(num_subwords, len(ds_col_value))

    def test_dataset2tf(self) -> None:
        """Test dataset conversion to `TensorFlow` format."""
        dummy_ds = HgfDataset.from_dict({
            "input_ids": [[2, 9293, 3567, 448, 3],
                          [2, 1122, 675, 3]],
            "attention_mask": [[1] * 5, [1] * 4],
            self._label_column: [[0, 1, 0, 1, 0], [1, 0, 1, 0]]
        })
        dummy_tf, ds_length = dataset2tf(
            dummy_ds, label_column=self._label_column)
        data_point = next(iter(dummy_tf))
        print(f"A data point converted to `TensorFlow`: {data_point}")
        (input_ids, attn_mask), labels = data_point
        self.assertEqual(ds_length, len(dummy_ds))
        self.assertShapeEqual(input_ids, attn_mask)
        self.assertShapeEqual(input_ids, labels)

    def test_prepare_dataset_split(self) -> None:
        """Test the full data pipeline, including tokenization,
        TensorFlow conversion and batching.
        """
        processed_ds, num_batches = prepare_dataset_split(
            dataset=self._dataset,
            batch_size=2,
            shuffling_buffer_size=2,
            tokenizer=self._tokenizer,
            text_column=self._text_column,
            label_column=self._label_column,
            ignore_label=self._ignore_label
        )
        processed_ds = processed_ds.prefetch(2)
        i = sum(1 for _ in processed_ds)
        self.assertEqual(num_batches, i)
        batch = next(iter(processed_ds))
        print(f"A data batch: {batch}")
        (input_ids, attn_mask), labels = batch
        self.assertShapeEqual(input_ids, attn_mask)
        self.assertShapeEqual(input_ids, labels)


if __name__ == "__main__":
    tf.test.main()
