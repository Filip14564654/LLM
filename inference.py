import torch
import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerModel
from utils.functions import load_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(tokenizer, config):
    model_cfg = config["model"]
    model = TransformerModel(
        vocab_size=config["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        num_heads=model_cfg["num_heads"],
        ff_dim=model_cfg.get("ff_dim", 256),
        num_layers=model_cfg["num_layers"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg.get("dropout", 0.0),
        positional_encoding=model_cfg.get("positional_encoding", "learned")
    ).to(DEVICE)

    model_path = config["checkpoint_path"].replace('.pt', '_best.pt')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

@torch.no_grad()
def generate(model, tokenizer, prompt, config, max_new_tokens=50):
    from utils.functions import token_to_id, load_vocab_mapping
    vocab_mapping = load_vocab_mapping(config["vocab_file"])
    tokens = tokenizer.tokenize(prompt)
    flat_tokens = [tok for sublist in tokens for tok in sublist]
    input_ids = [token_to_id(t, vocab_mapping, config["vocab_size"]) for t in flat_tokens][:config["model"]["max_len"] - 1]

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

        # Top-k + teplota
        filtered_logits = top_k_logits(next_token_logits.unsqueeze(0), k)
        probs = torch.softmax(filtered_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

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

        if input_tensor.size(1) >= config["model"]["max_len"]:
            break

    output_ids = input_tensor[0].tolist()
    return tokenizer.detokenize(output_ids)


def top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)


def main():
    config = load_config("config.yaml")
    print("[INFO] Načítám tokenizer a model...")
    tokenizer = BPETokenizer()
    tokenizer.load_vocab(config["vocab_file"])
    print(f"[DEBUG] Velikost slovníku: {len(tokenizer.vocab)}")

    model = load_model(tokenizer, config)

    print("[READY] Zadej prompt (prázdný vstup ukončí program):")
    while True:
        prompt = input(">>> ").strip()
        if not prompt:
            print("Ukončuji...")
            break
        output = generate(model, tokenizer, prompt, config, max_new_tokens=config["model"]["max_len"] - 1)
        print("=== VÝSTUP ===")
        print(output)
        print("=" * 40)


if __name__ == "__main__":
    main()
