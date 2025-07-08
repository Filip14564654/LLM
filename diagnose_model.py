#!/usr/bin/env python3
"""
Diagnostický skript pro kontrolu stavu modelu a dat
"""

import torch
import json
import sys
import os
sys.path.append(os.path.abspath("."))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerModel
from utils.functions import load_config, load_vocab_mapping, token_to_id

def check_cuda():
    print("=== CUDA DIAGNOSTIKA ===")
    print(f"CUDA dostupné: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Počet GPU: {torch.cuda.device_count()}")
        print(f"Aktuální GPU: {torch.cuda.current_device()}")
        print(f"Název GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU paměť: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA není dostupné - model se bude trénovat na CPU (velmi pomalé!)")
    print()

def check_data():
    print("=== DATA DIAGNOSTIKA ===")
    config = load_config()
    
    # Kontrola trénovacích dat
    train_file = config["train_file"]
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Trénovací soubor: {len(lines)} řádků")
        print(f"Ukázka: {lines[0][:100]}...")
    else:
        print(f"❌ Trénovací soubor neexistuje: {train_file}")
    
    # Kontrola slovníku
    vocab_file = config["vocab_file"]
    if os.path.exists(vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"Slovník: {len(vocab)} tokenů")
        print(f"Konfigurovaná velikost: {config['vocab_size']}")
        if len(vocab) > config['vocab_size']:
            print("⚠️  Slovník je větší než konfigurovaná velikost - budou kolize!")
    else:
        print(f"❌ Slovník neexistuje: {vocab_file}")
    print()

def check_model():
    print("=== MODEL DIAGNOSTIKA ===")
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_cfg = config.get("model", {})
    model = TransformerModel(
        vocab_size=config["vocab_size"],
        embed_dim=model_cfg.get("embed_dim", 128),
        num_heads=model_cfg.get("num_heads", 4),
        ff_dim=model_cfg.get("ff_dim", 256),
        num_layers=model_cfg.get("num_layers", 4),
        max_len=model_cfg.get("max_len", 512),
        dropout=model_cfg.get("dropout", 0.0),
        positional_encoding=model_cfg.get("positional_encoding", "learned"),
    ).to(device)
    
    # Počet parametrů
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Celkem parametrů: {total_params:,}")
    print(f"Trénovatelných parametrů: {trainable_params:,}")
    print(f"Model velikost: {total_params * 4 / 1e6:.1f} MB (float32)")
    
    # Test forward pass
    try:
        test_input = torch.randint(0, config["vocab_size"], (2, 10)).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Forward pass funguje: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass selhal: {e}")
    print()

def check_tokenization():
    print("=== TOKENIZACE DIAGNOSTIKA ===")
    config = load_config()
    
    try:
        tokenizer = BPETokenizer()
        tokenizer.load_vocab(config["vocab_file"])
        vocab_mapping = load_vocab_mapping(config["vocab_file"])
        
        test_text = "Hello world, this is a test."
        tokens = tokenizer.tokenize(test_text)
        flat = [t for sublist in tokens for t in sublist]
        ids = [token_to_id(t, vocab_mapping, config["vocab_size"]) for t in flat]
        
        print(f"Test text: {test_text}")
        print(f"Tokeny: {tokens}")
        print(f"Flat tokeny: {flat}")
        print(f"ID: {ids}")
        print("✅ Tokenizace funguje")
        
    except Exception as e:
        print(f"❌ Tokenizace selhala: {e}")
    print()

def main():
    print("🔍 DIAGNOSTIKA LLM PROJEKTU")
    print("=" * 50)
    
    check_cuda()
    check_data()
    check_model()
    check_tokenization()
    
    print("=== DOPORUČENÍ ===")
    if not torch.cuda.is_available():
        print("🚨 INSTALUJTE CUDA PRO RYCHLÉJŠÍ TRÉNOVÁNÍ!")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("✅ Diagnostika dokončena")

if __name__ == "__main__":
    main() 