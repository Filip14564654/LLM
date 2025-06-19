import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerModel
import json
import yaml
from dataclasses import asdict

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# === Dataset ===
class PIQADataset(Dataset):
    def __init__(self, filepath, tokenizer):
        with open(filepath, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(self.lines[idx])
        flat = [tok for sublist in tokens for tok in sublist]
        token_ids = [hash(t) % CONFIG["vocab_size"] for t in flat]
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    batch = [item for item in batch if len(item) > 1]
    max_len = max(len(x) for x in batch)
    padded = [torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in batch]
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

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BPETokenizer()
    tokenizer.load_vocab(CONFIG["vocab_file"])

    train_data = PIQADataset(CONFIG["train_file"], tokenizer)
    val_data = PIQADataset(CONFIG["val_file"], tokenizer)

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*CONFIG["epochs"])

    for epoch in range(CONFIG["epochs"]):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.contiguous().view(-1)

                loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch} | Batch {i}] Loss: {loss.item():.4f}")

        # === Uložení checkpointu ===
        torch.save(model.state_dict(), CONFIG["checkpoint_path"])
        print(f"[INFO] Model uložen do {CONFIG['checkpoint_path']}")

        # === Validace ===
        ppl = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] Validace perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
