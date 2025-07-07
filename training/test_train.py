# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from tokenizer.test_bpe_tokenizer import BPETokenizer, CharTextDataset
# import time
# import yaml

# # === 1. Nastavení ===
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # BATCH_SIZE = 32
# # LEARNING_RATE = 3e-5
# # NUM_EPOCHS = 3
# # MODEL_DIM = 512
# def load_config(path="config.yaml"):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# # === 2. Vlastní Dataset ===
# class CustomTextDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_length=128):
#         self.tokenizer = tokenizer
#         self.texts = texts
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         tokens = self.tokenizer.encode(self.texts[idx])[:self.max_length]
#         if len(tokens) < self.max_length:
#             tokens += [0] * (self.max_length - len(tokens))

#         input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
#         target_ids = torch.tensor(tokens[1:], dtype=torch.long)
#         return input_ids, target_ids


# # === 3. Jednoduchý Transformer Encoder ===
# class TransformerLanguageModel(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead=4, num_layers=4):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def forward(self, input_ids, attention_mask=None):
#         x = self.embedding(input_ids)
#         x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
#         logits = self.fc_out(x)
#         return logits

# # === 4. Trénovací smyčka ===
# def train(model, dataloader, optimizer, criterion):
#     model.train()
#     total_loss = 0
#     for input_ids, target_ids in dataloader:
#         input_ids = input_ids.to(DEVICE)
#         target_ids = target_ids.to(DEVICE)

#         optimizer.zero_grad()
#         output = model(input_ids)  # model vrací logits
#         loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     return total_loss / len(dataloader)

# # === 5. Hlavní běh ===
# def main():
#     config = load_config()

#     print(f"Trénuji na zařízení: {DEVICE}")
#     print("torch.cuda.is_available(): " + torch.cuda.is_available().__str__())

#     print("=== CUDA Diagnostika ===")
#     print(f"CUDA dostupná: {torch.cuda.is_available()}")
#     print(f"Aktivní zařízení: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
#     print(f"Název zařízení: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

#     tokenizer = CharTokenizer()
#     tokenizer.load(config["vocab_file"])

#     with open(config["train_file"], "r", encoding="utf-8") as f:
#         texts = f.readlines()

#     dataset = CustomTextDataset(texts, tokenizer, max_length=config["model"]["max_len"])
#     dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

#     model = TransformerLanguageModel(
#         vocab_size=len(tokenizer.vocab),
#         d_model=config["model"]["embed_dim"],
#         nhead=config["model"]["num_heads"],
#         num_layers=config["model"]["num_layers"]
#     ).to(DEVICE)

#     learning_rate = float(config["lr"])
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(config["epochs"]):
#         start = time.time()
#         loss = train(model, dataloader, optimizer, criterion)
#         print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss:.4f}, Time: {time.time() - start:.2f}s")

#     torch.save(model.state_dict(), config["checkpoint_path"])
#     print("Model uložen.")


# if __name__ == "__main__":
#     main()

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
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for text in texts:
            tokens = tokenizer.encode(text.strip())
            if len(tokens) < 2:
                continue
            for i in range(0, len(tokens) - max_length):
                x = tokens[i:i + max_length]
                y = tokens[i + 1:i + 1 + max_length]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# === 3. Jednoduchý Transformer Encoder ===
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
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

    # === Načtení trénovacích dat ===
    with open(config["train_file"], "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # === Tokenizer ===
    tokenizer = CharTokenizer()  # 🔄 změna z BPETokenizer
    tokenizer.build_vocab(texts)
    print("[DEBUG] Tokenizace 5 řádků:")
    for i in range(5):
        line = texts[i].strip()
        tokens = tokenizer.encode(line)
        print(f"  {repr(line)} → {tokens} ({len(tokens)} znaků)")
    print(f"Velikost slovníku: {tokenizer.vocab_size()}")

    # === Dataset a DataLoader ===
    dataset = CharTextDataset(texts, tokenizer, max_length=config["model"]["max_len"])
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
        vocab_size=tokenizer.vocab_size(),  # 🔄
        d_model=config["model"]["embed_dim"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"]
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
