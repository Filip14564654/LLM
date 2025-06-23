import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerModel
from utils.functions import load_config, stable_hash, latest_version_path

CONFIG = load_config()

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.lines[idx])
        flat = [tok for sublist in tokens for tok in sublist]
        ids = [stable_hash(t, CONFIG["vocab_size"]) for t in flat]
        return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):
    batch = [b for b in batch if len(b) > 1]
    max_len = CONFIG["model"]["max_len"]
    batch = [b[:max_len] for b in batch]
    padded = [torch.cat([b, torch.zeros(max_len - len(b), dtype=torch.long)]) for b in batch]
    return torch.stack(padded)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.contiguous().view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer()
    tokenizer.load_vocab(CONFIG["vocab_file"])
    data = TextDataset(CONFIG["val_file"], tokenizer)
    loader = DataLoader(data, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    model_cfg = CONFIG.get("model", {})
    model = TransformerModel(
        vocab_size=CONFIG["vocab_size"],
        embed_dim=model_cfg.get("embed_dim", 128),
        num_heads=model_cfg.get("num_heads", 4),
        ff_dim=model_cfg.get("ff_dim", 256),
        num_layers=model_cfg.get("num_layers", 4),
        max_len=model_cfg.get("max_len", 512),
        dropout=model_cfg.get("dropout", 0.0),
        positional_encoding=model_cfg.get("positional_encoding", "learned"),
    ).to(device)
    ckpt = latest_version_path(CONFIG["checkpoint_path"])
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    criterion = nn.CrossEntropyLoss()
    ppl = evaluate(model, loader, criterion, device)
    print(f"Validation perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()

