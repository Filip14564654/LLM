import os
import json
from datasets import load_dataset

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def download_and_prepare_piqa():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n[INFO] Stahuji dataset PIQA z Hugging Face...")
    dataset = load_dataset("ybisk/piqa")

    print("[INFO] Ukládám surová data...")
    for split in ["train", "validation", "test"]:
        split_data = dataset[split]
        raw_path = os.path.join(RAW_DIR, f"piqa_{split}.jsonl")
        with open(raw_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[INFO] Základní preprocessing...")
    for split in ["train", "validation", "test"]:
        input_path = os.path.join(RAW_DIR, f"piqa_{split}.jsonl")
        output_path = os.path.join(PROCESSED_DIR, f"piqa_{split}_formatted.txt")

        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:

            for line in infile:
                data = json.loads(line)
                premise = data["goal"].strip()
                choice1 = data["sol1"].strip()
                choice2 = data["sol2"].strip()
                label = str(data.get("label", ""))  # prázdný pro test set

                example_text = f"Q: {premise}\nA) {choice1}\nB) {choice2}\nAnswer: {label}\n\n"
                outfile.write(example_text)

    print("[DONE] Dataset PIQA byl stažen a předzpracován.")


if __name__ == "__main__":
    download_and_prepare_piqa()
