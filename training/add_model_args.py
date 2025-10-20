import torch
import yaml
import os
import sys

# === Importy z projektu ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.test_train import TransformerLanguageModel
from tokenizer.char_tokenizer import CharTokenizer

# === Cesty ke konfiguračnímu souboru a modelu ===
config_path = "config.yaml"
ckpt_path = "trained_model.pt"

# === Načti konfiguraci ===
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# === Načti tokenizer ===
tokenizer = CharTokenizer()
tokenizer.load(config["vocab_file"])
vocab_size = tokenizer.vocab_size()

# === Načti checkpoint (předchozí model) ===
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

# === Extrakce dimenzí z checkpointu ===
token_emb_weight = state_dict["token_emb.weight"]
pos_emb_weight = state_dict["pos_emb.weight"]
vocab_size_from_ckpt = token_emb_weight.shape[0]
d_model = token_emb_weight.shape[1]
max_len = pos_emb_weight.shape[0]

# === Inicializace modelu podle checkpointu ===
model = TransformerLanguageModel(
    vocab_size=vocab_size_from_ckpt,
    d_model=d_model,
    nhead=config["model"]["num_heads"],
    num_layers=config["model"]["num_layers"],
    max_len=max_len,
    dropout=config["model"].get("dropout", 0.1)
)
model.load_state_dict(state_dict)

# === Vytvoř nový checkpoint s model_args ===
new_checkpoint = {
    "model": model.state_dict(),
    "model_args": {
        "vocab_size": vocab_size_from_ckpt,
        "d_model": d_model,
        "nhead": config["model"]["num_heads"],
        "num_layers": config["model"]["num_layers"],
        "max_len": max_len,
        "dropout": config["model"].get("dropout", 0.1)
    }
}
torch.save(new_checkpoint, ckpt_path)
print("✅ Checkpoint úspěšně doplněn o model_args.")
