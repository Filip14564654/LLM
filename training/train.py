import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch import nn, optim
import sys
import os
import hashlib
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerModel
from utils.functions import load_config, stable_hash, next_version_path, load_vocab_mapping, token_to_id

CONFIG = load_config()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

WANDB_API_KEY = os.environ.get('WANDB_API_KEY', None)
if WANDB_AVAILABLE and not WANDB_API_KEY:
    WANDB_API_KEY = 'd6bdcf5302d3db3d67a19ca99d00a47442313e71'
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    try:
        wandb.login(key=WANDB_API_KEY, relogin=True)
    except Exception as e:
        print(f"[WARN] wandb login selhal: {e}")

# === Dataset ===
class TextIterableDataset(IterableDataset):
    def __init__(self, filepath, tokenizer, vocab_mapping):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.vocab_mapping = vocab_mapping

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = self.tokenizer.tokenize(line)
                flat = [tok for sublist in tokens for tok in sublist]
                token_ids = [token_to_id(t, self.vocab_mapping, CONFIG["vocab_size"]) for t in flat]
                yield torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    batch = [item for item in batch if len(item) > 1]
    max_len = CONFIG["model"]["max_len"]
    batch = [x[:max_len] for x in batch]  # truncate
    padded = [torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in batch]
    return torch.stack(padded)


def evaluate(model, loader, criterion, device, max_batches=100):
    print("[DEBUG] Spouštím evaluate()")
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                print(f"[DEBUG] Překročen limit {max_batches} batchů, ukončuji evaluate().")
                break

            print(f"[DEBUG] Batch shape: {batch.shape}")
            batch = batch.to(device)
            if batch.size(1) < 2:
                continue

            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.contiguous().view(-1)

            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    if total_tokens == 0:
        return float('inf')

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()




def count_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, _ in enumerate(f, 1):
                if i > 10_000_000:
                    #print(f"[WARN] Soubor {filepath} má více než 10 milionů řádků, používám odhad.")
                    return 10_000_000
            return i
    except Exception as e:
        print(f"[WARN] Nelze spočítat řádky v {filepath}: {e}, používám odhad 10000.")
        return 10000


def main():
    print("[DEBUG] Spouštím evaluate()")
    print("\n=== [START] Trénink modelu ===\n")
    print(f"[INFO] Zařízení: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO] Konfigurace: batch_size={CONFIG['batch_size']}, epochs={CONFIG['epochs']}, lr={CONFIG['lr']}")
    print(f"[INFO] Trénovací soubor: {CONFIG['train_file']}")
    print(f"[INFO] Validační soubor: {CONFIG['val_file']}")
    print(f"[INFO] Vocab: {CONFIG['vocab_file']} (size={CONFIG['vocab_size']})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BPETokenizer()
    tokenizer.load_vocab(CONFIG["vocab_file"])
    vocab_mapping = load_vocab_mapping(CONFIG["vocab_file"])
    print("[INFO] Načítám trénovací data...")
    train_data = TextIterableDataset(CONFIG["train_file"], tokenizer, vocab_mapping)
    print("[INFO] Načítám validační data...")
    val_data = TextIterableDataset(CONFIG["val_file"], tokenizer, vocab_mapping)

    pretrain_file = CONFIG.get("pretrain_file")
    pretrain_epochs = CONFIG.get("pretrain_epochs", 0)

    # Spočítej počet batchů pro scheduler
    num_train_lines = count_lines(CONFIG["train_file"])
    batches_per_epoch = (num_train_lines + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
    T_max = batches_per_epoch * (CONFIG.get("epochs", 0) + pretrain_epochs)

    if pretrain_file and pretrain_epochs > 0:
        pretrain_data = TextIterableDataset(pretrain_file, tokenizer, vocab_mapping)
        pretrain_loader = DataLoader(
            pretrain_data,
            batch_size=CONFIG["batch_size"],
            shuffle=False,  # IterableDataset nesmí mít shuffle=True
            collate_fn=collate_fn,
        )
    else:
        pretrain_loader = None

    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
    )

    # Early stopping variables
    best_val_ppl = float('inf')
    patience = 3  # Počet epoch bez zlepšení
    patience_counter = 0

    if WANDB_AVAILABLE:
        wandb.init(
            project='llm-training',
            name='transformer-run',
            config=CONFIG,
            reinit=True,
            anonymous=None
        )

    def train_epoch(loader, epoch_desc="", total_batches=None):
        try:
            dataset_len = len(loader.dataset)
        except TypeError:
            dataset_len = 'N/A'
        print(f"[DEBUG] Spouštím train_epoch pro {epoch_desc} s {dataset_len} vzorky, batch_size={CONFIG['batch_size']}")
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        for i, batch in enumerate(loader):
#            print(f"[DEBUG] Batch {i} velikost: {batch.shape if hasattr(batch, 'shape') else type(batch)}")
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.contiguous().view(-1)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # pyright: ignore
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            if i % 10 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                elapsed = time.time() - start_time
                loader_len = total_batches if total_batches is not None else 'N/A'
                print(f"[{epoch_desc}Batch {i}/{loader_len}] Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, Elapsed: {elapsed:.1f}s")
                if WANDB_AVAILABLE:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/avg_loss': avg_loss,
                        'train/batch': i,
                        'train/epoch': epoch_desc
                    })
                total_loss = 0
                num_batches = 0
                start_time = time.time()
        print(f"[DEBUG] train_epoch pro {epoch_desc} dokončen.")

    # Optional pretraining stage
    if pretrain_loader is not None:
        for epoch in range(pretrain_epochs):
            print(f"\n[PRETRAIN] Epoch {epoch+1}/{pretrain_epochs}")
            train_epoch(pretrain_loader, epoch_desc=f"Pretrain {epoch} | ")

    # Fine-tuning on task data
    for epoch in range(CONFIG["epochs"]):
        print(f"\n[TRAIN] Epoch {epoch+1}/{CONFIG['epochs']}")
        epoch_start = time.time()
        train_epoch(train_loader, epoch_desc=f"Epoch {epoch} | ", total_batches=batches_per_epoch)
        print(f"[INFO] Epoch {epoch+1} trénink hotov za {time.time()-epoch_start:.1f}s")

        # === Validace ===
        ppl = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] Validace perplexity: {ppl:.2f}")
        if WANDB_AVAILABLE:
            wandb.log({'val/perplexity': ppl, 'val/epoch': epoch})
        
        # === Early stopping ===
        if ppl < best_val_ppl:
            best_val_ppl = ppl
            patience_counter = 0
            # Uložit nejlepší model
            best_model_path = CONFIG["checkpoint_path"].replace('.pt', '_best.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Nový nejlepší model uložen do {best_model_path}")
        else:
            patience_counter += 1
            print(f"[INFO] Žádné zlepšení po {patience_counter} epochách")
        
        # === Uložení checkpointu včetně verze ===
        save_path = next_version_path(CONFIG["checkpoint_path"])
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Model uložen do {save_path}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[STOP] Early stopping po {epoch + 1} epochách (žádné zlepšení po {patience} epochách)")
            break

    print("\n=== [KONEC] Trénink dokončen ===\n")

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()

