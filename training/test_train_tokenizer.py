# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from tokenizer.test_bpe_tokenizer import BPETokenizer



# TRAIN_FILE = "data/processed/shakespeare_train.txt"
# VOCAB_FILE = "tokenizer/vocab.json"


# def main():
#     if not os.path.exists(TRAIN_FILE):
#         raise FileNotFoundError(f"Trénovací soubor nenalezen: {TRAIN_FILE}")

#     print("[INFO] Načítám trénovací korpus pro BPE...")
#     with open(TRAIN_FILE, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         print(f"[DEBUG] Načteno {len(lines)} řádků z trénovacího souboru.")
#         # lines = lines[:100]
#         # lines = lines[:1000]

#     # Instancuj a trénuj BPE tokenizer (viz principy v minGPT, NanoGPT, gpt2-from-scratch)
#     tokenizer = BPETokenizer(vocab_size=5000)
#     tokenizer.train(lines)

#     print(f"[INFO] Ukládám slovník do {VOCAB_FILE}...")
#     os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
#     tokenizer.save(VOCAB_FILE)

#     print("[DONE] Slovník úspěšně vytvořen a uložen.")

#     encoded = tokenizer.encode("machine learning is powerful")
#     print("Encoded:", encoded)

#     decoded = tokenizer.decode(encoded)
#     print("Decoded:", decoded)


# if __name__ == "__main__":
#     main()
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.char_tokenizer import CharTokenizer  # nový tokenizer

TRAIN_FILE = "data/processed/shakespeare_train.txt"
VOCAB_FILE = "tokenizer/char_vocab.json"

def main():
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Trénovací soubor nenalezen: {TRAIN_FILE}")

    print("[INFO] Načítám trénovací korpus...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        print(f"[DEBUG] Načteno {len(lines)} neprázdných řádků.")

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(lines)

    print(f"[INFO] Velikost slovníku: {tokenizer.vocab_size()} znaků")
    print(f"[INFO] Ukládám slovník do {VOCAB_FILE}...")
    os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)

    # Ulož slovník jako JSON: znak -> ID
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(tokenizer.stoi, f, ensure_ascii=False, indent=2)

    print("[DONE] Slovník úspěšně vytvořen a uložen.")

    # Test
    test_str = "To be, or not to be."
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    print("[TEST] Testovací řetězec:", test_str)
    print("        Zakódováno:", encoded)
    print("        Dekódováno:", decoded)

if __name__ == "__main__":
    main()
