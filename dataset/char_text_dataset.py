import torch
from torch.utils.data import Dataset

class CharTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for text in texts:
            tokens = tokenizer.encode(text.strip())
            if len(tokens) < 2:
                continue

            # Hlavní případ: sliding window pro dlouhé sekvence
            if len(tokens) >= max_length + 1:
                for i in range(0, len(tokens) - max_length):
                    x = tokens[i:i + max_length]
                    y = tokens[i + 1:i + 1 + max_length]
                    self.data.append((x, y))
            else:
                # Fallback: i krátká sekvence jako jeden vzorek (s paddingem)
                x = tokens[:-1]
                y = tokens[1:]
                if len(x) == len(y) and len(x) > 0:
                    pad_len = max_length - len(x)
                    x += [0] * pad_len
                    y += [0] * pad_len
                    self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

