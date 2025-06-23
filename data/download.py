import os
import urllib.request

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_and_prepare_shakespeare():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_path = os.path.join(RAW_DIR, "shakespeare.txt")
    print("[INFO] Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, raw_path)

    print("[INFO] Splitting into train/validation...")
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    split_idx = int(len(lines) * 0.9)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    train_path = os.path.join(PROCESSED_DIR, "shakespeare_train.txt")
    val_path = os.path.join(PROCESSED_DIR, "shakespeare_validation.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print("[DONE] Tiny Shakespeare downloaded and processed.")


if __name__ == "__main__":
    download_and_prepare_shakespeare()
