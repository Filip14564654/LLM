import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerModel
from utils.functions import load_config

# === Load config and model ===
CONFIG = load_config()  # same config used in training

model_cfg = CONFIG["model"]
model = TransformerModel(
    vocab_size=CONFIG["vocab_size"],
    embed_dim=model_cfg.get("embed_dim", 128),
    num_heads=model_cfg.get("num_heads", 4),
    ff_dim=model_cfg.get("ff_dim", 256),
    num_layers=model_cfg.get("num_layers", 4),
    max_len=model_cfg.get("max_len", 512),
    dropout=model_cfg.get("dropout", 0.0),
    positional_encoding=model_cfg.get("positional_encoding", "learned"),
)
model.load_state_dict(torch.load("model_checkpoint.pt", map_location="cpu"))
model.eval()

tokenizer = BPETokenizer()
tokenizer.load_vocab(CONFIG["vocab_file"])

# Example input
text = "Why do birds"
tokens = tokenizer.tokenize(text)
flat_tokens = [tok for sublist in tokens for tok in sublist]
token_ids = [hash(t) % CONFIG["vocab_size"] for t in flat_tokens]
input_tensor = torch.tensor([token_ids], dtype=torch.long)

with torch.no_grad():
    logits = model(input_tensor)
    next_token_logits = logits[0, -1]  # last token's output
    predicted_token_id = torch.argmax(next_token_logits).item()

# You may want a reverse-vocab or decoding logic
print("Predicted next token ID:", predicted_token_id)
