# Coreference Resolution on Hungarian Data with BERT

This repository contains the source code of a corefernce resolver trained on Hungarian data.

## Installation

We use [Poetry](https://python-poetry.org/) for dependency management and packaging. [Install](https://python-poetry.org/docs/#installation) it, clone this repository, then run

```bash
cd bert_coref_hu
poetry install
```

## Training BERT

We briefly describe below how we fine-tuned a pre-trained BERT model to support coreference resolution.

To see the full argument lists of our scripts, run

```bash
python3 <script_name>.py -h
```

### Data conversion

Our original data was contained in TSV-style document with the following fields:
* form
* koref
* anas
* lemma
* xpostag
* upostag
* feats
* id
* deprel
* head
* cons

Each row corresponded to a single token and sentences were separated by blank lines. The values of the `koref` field were gold standard coreference labels. The rest of the linguistic annotation was generated with the [emagyar](https://e-magyar.hu/en) text processing system.

The following scripts were applied to adjust the data to our task:
1. `src/bert_coref/group_sentences.py`: This splits the input into parts each of which meets the maximum sequence length constraint of a specific model (in our case, [`huBERT`](https://huggingface.co/SZTAKI-HLT/hubert-base-cc)). The resulting subsequences are separated by double blank lines.
2. `src/bert_coref/relabel.py`: It relabels the input based on POS-based filter rules.
3. `src/bert_coref/conll2jsonl.py`: It converts the TSV-style input to `jsonlines` (one line per document, documents separated by double blank lines in the input). `jsonlines` was the data format that we used to fine-tune `huBERT`. Spcify the `--create-triplets` flag to relabel the data for fine-tuning with the triplet loss objective.

### Fine-tuning the model

The training script is `src/bert_coref/triplet_train.py`. Note that it uses [WandB](https://wandb.ai/site) for logging, which means that running it requires a WandB account.

The triplet loss implementation can be found in `src/bert_coref/losses.py`.

## Clustering

Fine-tuned contextualized embeddings can be used as the inputs of a clustering algorithm (and clustering is expected to be equivalent to coreference resolution). This can be done by running `src/bert_coref/clustering/cluster.py`. This module requires `jsonlines`-style input (see above) and a fine-tuned encoder model. The key `labels` for gold standard corefence labels must be provided in the input, but these labels will not be used. For inference, it is sufficient to specify arbitrary values as coreference labels. The output is a TSV-style file with a new `koref_cluster` field added to the annotation. This field includes the inferred coreference labels.

Clustering outputs can be evaluated against gold standard labels with the `src/bert_coref/clustering/evaluate_clustering.py` script.

You can use the `src/bert_coref/clustering/plot_coref.py` script to visualize the clustering input.

## Download

Due to anonymity requirements, we can provide a download link to the weights of our fine-tuned BERT model only after receiving the reviews of our paper.
