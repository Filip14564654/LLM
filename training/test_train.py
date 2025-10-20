import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import time
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 🔄 ZMĚNA: importujeme char-level tokenizer
from tokenizer.char_tokenizer import CharTokenizer  

# === 1. Nastavení ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === 2. Dataset pro znakovou tokenizaci ===
class CharTextDataset(Dataset):
    """Continuous character dataset with random slicing like nanoGPT."""

    def __init__(self, token_ids, block_size: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + 1 + self.block_size]
        return x, y

# === 3. Jednoduchý Transformer Encoder ===
class TransformerLanguageModel(nn.Module):
    """Minimal GPT-style model with positional embeddings and causal masking."""

    def __init__(self, vocab_size, d_model, nhead=4, num_layers=4, max_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_out.weight = self.token_emb.weight  # weight tying
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x).permute(1, 0, 2)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(input_ids.device)
        x = self.transformer_encoder(x, mask=mask)
        x = x.permute(1, 0, 2)
        logits = self.fc_out(x)
        return logits

# === 4. Trénovací smyčka ===
def train(model, dataloader, optimizer, criterion, scheduler=None, warmup_steps=500):
    model.train()
    total_loss = 0
    global_step = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)

        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
        loss.backward()

        # ✅ Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ✅ Warmup + Scheduler update
        if scheduler is not None:
            global_step += 1
            if global_step < warmup_steps:
                # lineární warmup (ručně)
                warmup_lr = scheduler.base_lrs[0] * global_step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
            else:
                scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# === 5. Hlavní běh ===
def main():
    config = load_config()


    print(f"Trénuji na zařízení: {DEVICE}")
    print("=== CUDA Diagnostika ===")
    print(f"CUDA dostupná: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Aktivní zařízení: {torch.cuda.current_device()}")
        print(f"Název zařízení: {torch.cuda.get_device_name(0)}")

    # === Načtení trénovacích dat jako kontinuální text ===
    with open(config["train_file"], "r", encoding="utf-8") as f:
        full_text = f.read()

    # === Tokenizer ===
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([full_text])
    sample = full_text.splitlines()[0]
    print("[DEBUG] Ukázka tokenizace:", tokenizer.encode(sample)[:20])
    print(f"Velikost slovníku: {tokenizer.vocab_size()}")

    # === Dataset a DataLoader ===
    tokens = tokenizer.encode(full_text)
    dataset = CharTextDataset(tokens, block_size=config["model"]["max_len"])
    print(f"[DEBUG] Počet trénovacích vzorků: {len(dataset)}")
    print("[DEBUG] První 3 vzorky (x→y):")
    for i in range(3):
        x, y = dataset[i]
        print("x:", tokenizer.decode(x.tolist()))
        print("y:", tokenizer.decode(y.tolist()))
        print("---")
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # === Model ===
    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size(),
        d_model=config["model"]["embed_dim"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        max_len=config["model"]["max_len"],
        dropout=config["model"].get("dropout", 0.1),
    ).to(DEVICE)

    # === Optimalizátor s weight decay ===
    optimizer = optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=0.01)

    # === Scheduler: Cosine annealing přes počet update kroků ===
    total_steps = config["epochs"] * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        start = time.time()
        loss = train(model, dataloader, optimizer, criterion, scheduler, warmup_steps=500)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss:.4f}, Time: {time.time() - start:.2f}s")

    torch.save(model.state_dict(), config["checkpoint_path"])
    print("Model uložen.")

if __name__ == "__main__":
    main()
