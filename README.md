# LLM Experiment Framework

This repository provides a minimal but flexible setup for experimenting with Transformer language models. The code is inspired by projects such as NanoGPT and minGPT.

## Setup

1. Install Python dependencies (PyTorch, datasets, PyYAML):

```bash
pip install torch datasets pyyaml
```

2. (Optional) Prepare the PIQA dataset:

```bash
python data/download.py
python training/train_tokenizer
```

## Training

Edit `config.yaml` to adjust model and training parameters. Then run:

```bash
python training/train.py
```

The script will save checkpoints to the path specified in the configuration.
Sequences longer than `model.max_len` in `config.yaml` are automatically
truncated during batching.

## Testing

Basic unit tests are located in `tests/`. Run them with:

```bash
python -m unittest discover tests
```

