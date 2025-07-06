import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

from tokenizer.test_bpe_tokenizer import BPETokenizer
from training.test_train import TransformerLanguageModel  # přizpůsob cestu podle umístění modelu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "trained_model.pt"
VOCAB_PATH = "tokenizer/vocab.json"
MODEL_DIM = 256
MAX_NEW_TOKENS = 50
MAX_INPUT_LENGTH = 128

def load_model(tokenizer):
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.vocab),
        d_model=MODEL_DIM
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt)[:MAX_INPUT_LENGTH]

    if not input_ids:
        print("[ERROR] Tokenizer nerozpoznal žádné tokeny – prompt je mimo slovník.")
        return "<unk>"

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    temperature = 1.5
    k = 50
    repetition_counts = {}
    last_tokens = []

    for _ in range(max_new_tokens):
        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]

        # Penalizace opakujících se tokenů
        for token_id, count in repetition_counts.items():
            if count >= 3:
                next_token_logits[token_id] -= 1.0

        # Top-k ořezání + teplota
        filtered_logits = top_k_logits(next_token_logits.unsqueeze(0), k)
        probs = torch.softmax(filtered_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Historie pro penalizaci
        repetition_counts[next_token] = repetition_counts.get(next_token, 0) + 1
        last_tokens.append(next_token)
        if len(last_tokens) > 10:
            last_tokens.pop(0)
            if all(t == last_tokens[0] for t in last_tokens):
                print("[INFO] Ukončeno kvůli opakování tokenů.")
                break

        input_tensor = torch.cat(
            [input_tensor, torch.tensor([[next_token]], device=DEVICE)], dim=1
        )

    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)


def top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)

def main():
    print("[INFO] Načítám tokenizer a model...")
    tokenizer = BPETokenizer()
    print(f"[DEBUG] Skutečná velikost slovníku po trénování: {len(tokenizer.vocab)}")
    tokenizer.load(VOCAB_PATH)
    model = load_model(tokenizer)
    print(tokenizer.encode("Caius"))


    print("[READY] Zadej prompt (prázdný vstup ukončí program):")
    while True:
        prompt = input(">>> ").strip()
        if not prompt:
            print("Ukončuji...")
            break
        output = generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
        print("=== VÝSTUP ===")
        print(output)
        print("=" * 40)

if __name__ == "__main__":
    main()
