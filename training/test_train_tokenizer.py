import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.test_bpe_tokenizer import BPETokenizer



TRAIN_FILE = "data/processed/shakespeare_train.txt"
VOCAB_FILE = "tokenizer/vocab.json"


def main():
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Trénovací soubor nenalezen: {TRAIN_FILE}")

    print("[INFO] Načítám trénovací korpus pro BPE...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"[DEBUG] Načteno {len(lines)} řádků z trénovacího souboru.")
        # lines = lines[:100]
        # lines = lines[:1000]

    # Instancuj a trénuj BPE tokenizer (viz principy v minGPT, NanoGPT, gpt2-from-scratch)
    tokenizer = BPETokenizer(vocab_size=5000)
    tokenizer.train(lines)

    print(f"[INFO] Ukládám slovník do {VOCAB_FILE}...")
    os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
    tokenizer.save(VOCAB_FILE)

    print("[DONE] Slovník úspěšně vytvořen a uložen.")

    encoded = tokenizer.encode("machine learning is powerful")
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)


if __name__ == "__main__":
    main()
