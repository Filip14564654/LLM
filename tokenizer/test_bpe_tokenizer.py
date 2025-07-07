import re
import json
from collections import defaultdict, Counter
from typing import List, Tuple
from torch.utils.data import Dataset
import torch

class BPETokenizer:
    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def get_stats(self, corpus):
        pairs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair: Tuple[str, str], corpus):
        new_corpus = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in corpus:
            new_word = pattern.sub(''.join(pair), word)
            new_corpus[new_word] = corpus[word]
        return new_corpus

    def train(self, texts: List[str]):
        corpus = Counter()
        for line in texts:
            if not line.strip():
                continue
            for word in line.strip().split():
                if not word:
                    continue
                chars = " ".join(list(word)) + " </w>"
                corpus[chars] += 1

        if not corpus:
            raise ValueError("[ERROR] Trénovací korpus je prázdný – zkontroluj vstupní data.")

        print(f"[INFO] Začíná BPE trénink s {len(corpus)} unikátními položkami.")
        num_merges = 0

        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(corpus)
            if not pairs:
                print(f"[WARNING] Žádné další páry k mergování. Trénink končí předčasně.")
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            corpus = self.merge_vocab(best, corpus)
            num_merges += 1

        # Vytvoření výsledného slovníku z tokenů
        tokens = set()
        for word in corpus:
            tokens.update(word.split())

        tokens.add("<unk>")  # Fallback token pro neznámé vstupy

        self.vocab = {token: idx for idx, token in enumerate(sorted(tokens))}

        print(f"[INFO] Trénink BPE dokončen. Merge kroků: {num_merges}")
        print(f"[INFO] Skutečná velikost slovníku: {len(self.vocab)}")


    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in text.strip().split():
            word = list(word) + ["</w>"]
            while True:
                pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
                merge = None
                for pair in self.merges:
                    if pair in pairs:
                        merge = pair
                        break
                if not merge:
                    break
                first, second = merge
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            tokens.extend(word)

        # Použij <unk> pro tokeny, které nejsou ve slovníku
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        words = [inv_vocab.get(i, "<unk>").replace("</w>", " ") for i in ids]
        return "".join(words).strip()

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab_file"]
        self.merges = [tuple(m) for m in data["merges"]]

        if "<unk>" not in self.vocab:
            self.vocab["<unk>"] = len(self.vocab)
            print("[INFO] Přidán chybějící token <unk> do slovníku.")
