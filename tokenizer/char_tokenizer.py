import json

class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def build_vocab(self, texts):
        chars = sorted(set("".join(texts)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join([self.itos.get(i, "?") for i in ids])

    def vocab_size(self):
        return len(self.stoi)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab(self):
        return self.stoi
