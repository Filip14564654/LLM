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


## Evaluating a trained model

After training finishes, you can measure perplexity on the validation set using `evaluation/eval_perplexity.py`:

```bash
python evaluation/eval_perplexity.py
```

The script loads the checkpoint and validation file specified in `config.yaml` and prints the validation perplexity.

## Chatting with the model

You can run a simple interactive demo that generates a short sentence given your
prompt. Start the script and type your text:

```bash
python evaluation/chat.py
```

Type `quit` to exit. The demo prints the model's greedy completion of your
prompt as plain text.
