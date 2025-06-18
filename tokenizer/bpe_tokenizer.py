import re
import json
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.bpe_merges = []

    def get_vocab(self):
        return self.vocab

    def save_vocab(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def _get_stats(self, tokens):
        pairs = defaultdict(int)
        for word, freq in tokens.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair, tokens):
        pattern = re.escape(' '.join(pair))
        re_pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
        new_tokens = {}
        for word in tokens:
            new_word = re_pattern.sub(''.join(pair), word)
            new_tokens[new_word] = tokens[word]
        return new_tokens

    def train(self, corpus_lines):
        print("[INFO] Trénuji BPE na trénovacím korpusu...")
        tokens = Counter()
        for line in corpus_lines:
            line = line.strip()
            for word in line.split():
                word = ' '.join(list(word)) + ' </w>'
                tokens[word] += 1

        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(tokens)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            tokens = self._merge_vocab(best, tokens)
            self.bpe_merges.append(best)

        for word in tokens:
            self.vocab[word.replace(' ', '')] = tokens[word]
        print("[DONE] Slovník BPE vytvořen.")

    def encode(self, word):
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            candidate = None
            for merge in self.bpe_merges:
                if merge in pairs:
                    candidate = merge
                    break
            if candidate is None:
                break
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == candidate:
                    word = word[:i] + [''.join(candidate)] + word[i + 2:]
                    break
                i += 1
        return word

    def tokenize(self, text):
        return [self.encode(word) for word in text.strip().split()]

# Tato implementace bpe_tokenizer.py vychází z principů a struktury, které se objevují například v repozitářích:
#            minGPT – ručně řízený tokenizer s podporou subword tokenizace.
#            NanoGPT – jednoduché načtení vlastního slovníku a tokenizace.
#            gpt2-from-scratch – příkladové uložení a aplikace slovníku.