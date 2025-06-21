import torch
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerModel
from utils.functions import load_config

CONFIG = load_config()

def build_id_lookup(vocab_file, vocab_size):
    """Return a mapping from hashed token ID to a representative token.

    Because multiple tokens can map to the same hashed ID, we pick the most
    frequent token for each ID based on the counts stored in ``vocab_file``.
    ``vocab_file`` is expected to contain a JSON object mapping tokens to
    frequency counts as produced by ``BPETokenizer``.
    """

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Sort tokens by descending frequency to choose the most common token for
    # each ID.  ``vocab`` maps token -> count.
    sorted_tokens = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)

    table = {}
    for token, _ in sorted_tokens:
        tid = hash(token) % vocab_size
        if tid not in table:
            table[tid] = token
    return table

def load_model(device):
    cfg = CONFIG.get("model", {})
    model = TransformerModel(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=cfg.get("embed_dim", 128),
        num_heads=cfg.get("num_heads", 4),
        ff_dim=cfg.get("ff_dim", 256),
        num_layers=cfg.get("num_layers", 4),
        max_len=cfg.get("max_len", 512),
        dropout=cfg.get("dropout", 0.0),
        positional_encoding=cfg.get("positional_encoding", "learned"),
    ).to(device)
    model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location=device))
    model.eval()
    return model

def predict_next(model, ids, device):
    inp = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(inp)
        next_id = torch.argmax(logits[0, -1]).item()
    return next_id

def detokenize(tokens):
    words = []
    for t in tokens:
        if t.endswith("</w>"):
            words.append(t[:-4])
        else:
            words.append(t)
    return " ".join(words)

def generate_sentence(model, ids, device, id_lookup, max_new_tokens=20):
    generated = []
    current = ids[:]
    for _ in range(max_new_tokens):
        nxt = predict_next(model, current, device)
        token = id_lookup.get(nxt, "<UNK>")
        generated.append(token)
        current.append(nxt)
        if token.endswith(".</w>") or token.endswith("?</w>") or token.endswith("!</w>"):
            break
    return detokenize(generated)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer()
    tokenizer.load_vocab(CONFIG["vocab_file"])
    id_lookup = build_id_lookup(CONFIG["vocab_file"], CONFIG["vocab_size"])
    model = load_model(device)

    print("Type 'quit' to exit.")
    while True:
        text = input("You: ")
        if text.lower() in {"quit", "exit"}:
            break
        tokens = tokenizer.tokenize(text)
        flat = [t for sub in tokens for t in sub]
        ids = [hash(t) % CONFIG["vocab_size"] for t in flat]
        if not ids:
            print("Model: <no input>")
            continue
        sentence = generate_sentence(model, ids, device, id_lookup)
        print(f"Model: {sentence}")

if __name__ == "__main__":
    main()
