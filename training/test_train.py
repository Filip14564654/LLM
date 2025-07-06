import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.test_bpe_tokenizer import BPETokenizer
import time
import yaml

# === 1. Nastavení ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 32
# LEARNING_RATE = 3e-5
# NUM_EPOCHS = 3
# MODEL_DIM = 512
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === 2. Vlastní Dataset ===
class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])[:self.max_length]
        # padding
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return input_ids, input_ids  # dummy attention_mask


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
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for input_ids, attention_mask in dataloader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        shift_logits = output[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# === 5. Hlavní běh ===
def main():
    config = load_config()

    tokenizer = BPETokenizer()
    tokenizer.load(config["vocab_file"])

    with open(config["train_file"], "r", encoding="utf-8") as f:
        texts = f.readlines()

    dataset = CustomTextDataset(texts, tokenizer, max_length=config["model"]["max_len"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.vocab),
        d_model=config["model"]["embed_dim"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"]
    ).to(DEVICE)

    learning_rate = float(config["lr"])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        start = time.time()
        loss = train(model, dataloader, optimizer, criterion)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss:.4f}, Time: {time.time() - start:.2f}s")

    torch.save(model.state_dict(), config["checkpoint_path"])
    print("Model uložen.")


if __name__ == "__main__":
    main()
