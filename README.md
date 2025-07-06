# LLM Experiment Framework

This repository provides a minimal but flexible setup for experimenting with Transformer language models. The code is inspired by projects such as NanoGPT and minGPT.

## Setup

1. Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate LLM
```

If you prefer using ``pip`` directly, install ``torch``, ``datasets`` and ``pyyaml`` manually.

2. (Optional) Prepare the Tiny Shakespeare dataset:

```bash
python data/download.py
python training/train_tokenizer
```

## Training

Edit `config.yaml` to adjust model and training parameters. Then run:

```bash
python training/train.py
```

The script will save checkpoints to the path specified in the configuration.  If
the file already exists, a version suffix (``_vN``) is appended so older
checkpoints are preserved.
Sequences longer than `model.max_len` in `config.yaml` are automatically
truncated during batching.

### Pretraining

If `pretrain_file` is set in `config.yaml`, `training/train.py` will first run a
pretraining stage on that corpus for the number of epochs specified by
`pretrain_epochs`. A small sample corpus is provided in
`data/processed/general_sample.txt` as an example.

## Testing

Basic unit tests are located in `tests/`. Run them with:

```bash
python -m unittest discover tests
```


## Evaluating a trained model

After training finishes, you can measure perplexity on the validation set using `evaluation/eval_perplexity.py`:

```bash
python evaluation/eval_perplexity.py
```

The script loads the most recent checkpoint matching the path in
`config.yaml` and prints the validation perplexity.

## Chatting with the model

You can run a simple interactive demo that generates a short sentence given your
prompt. Start the script and type your text:

```bash
python evaluation/chat.py
```

Type `quit` to exit. The demo now uses top-k sampling (k=3) instead of greedy
decoding to produce a short sentence. Token IDs are derived using a stable MD5
hash so that the mapping is consistent across runs. The script automatically
loads the latest checkpoint.
