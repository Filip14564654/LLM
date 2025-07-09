import torch
import os
import sys
import yaml
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.char_tokenizer import CharTokenizer
from training.test_train import TransformerLanguageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config):
    # === Tokenizer ===
    tokenizer = CharTokenizer()
    tokenizer.load(config["vocab_file"])
    print(f"[DEBUG] Skutečná velikost slovníku po trénování: {tokenizer.vocab_size()}")

    # === Model ===
    checkpoint = torch.load(config["checkpoint_path"], map_location=DEVICE)
    model_args = checkpoint["model_args"]

    model = TransformerLanguageModel(
        vocab_size=model_args["vocab_size"],
        d_model=model_args["d_model"],
        nhead=model_args["nhead"],
        num_layers=model_args["num_layers"],
        max_len=model_args["max_len"],
        dropout=model_args.get("dropout", 0.1),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, tokenizer, model_args["max_len"]


@torch.no_grad()
def generate(model, tokenizer, prompt, max_len, max_new_tokens=100, temperature=1.0, top_k=50):
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[:max_len - 1]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    for _ in range(max_new_tokens):
        logits = model(input_tensor)
        logits = logits[0, -1, :] / temperature

        # Top-k sampling
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[-1]] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

        if input_tensor.size(1) >= max_len:
            break

    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)


def main():
    config = load_config()
    print("[INFO] Načítám tokenizer a model...")
    model, tokenizer, max_len = load_model_and_tokenizer(config)

    print("[READY] Zadej prompt (prázdný vstup ukončí program):")
    while True:
        prompt = input(">>> ").strip()
        if not prompt:
            print("Ukončuji...")
            break
        output = generate(model, tokenizer, prompt, max_len=max_len, max_new_tokens=max_len - len(prompt))
        print("=== VÝSTUP ===")
        print(output)
        print("=" * 40)


if __name__ == "__main__":
    main()
