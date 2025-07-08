#!/usr/bin/env python3
"""
Monitorovací skript pro sledování průběhu trénování
"""

import os
import time
import glob
from datetime import datetime

def get_latest_checkpoint():
    """Najde nejnovější checkpoint."""
    checkpoints = glob.glob("model_checkpoint*.pt")
    if not checkpoints:
        return None
    
    # Seřaď podle času modifikace
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]

def get_training_progress():
    """Získá informace o průběhu trénování."""
    latest_ckpt = get_latest_checkpoint()
    
    if latest_ckpt is None:
        return "Žádný checkpoint nenalezen"
    
    # Získej čas modifikace
    mtime = os.path.getmtime(latest_ckpt)
    mtime_str = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
    
    # Získej velikost souboru
    size_mb = os.path.getsize(latest_ckpt) / (1024 * 1024)
    
    return {
        'checkpoint': latest_ckpt,
        'last_modified': mtime_str,
        'size_mb': size_mb
    }

def monitor_training():
    """Monitoruje trénování v reálném čase."""
    print("📊 MONITOROVÁNÍ TRÉNOVÁNÍ")
    print("=" * 40)
    
    last_checkpoint = None
    last_time = None
    
    try:
        while True:
            progress = get_training_progress()
            
            if isinstance(progress, str):
                print(f"❌ {progress}")
                break
            
            current_checkpoint = progress['checkpoint']
            current_time = progress['last_modified']
            
            # Kontrola změn
            if current_checkpoint != last_checkpoint:
                print(f"\n🔄 Nový checkpoint: {current_checkpoint}")
                print(f"   Čas: {current_time}")
                print(f"   Velikost: {progress['size_mb']:.1f} MB")
                last_checkpoint = current_checkpoint
                last_time = current_time
            elif current_time != last_time:
                print(f"⏰ Aktualizace: {current_time} (velikost: {progress['size_mb']:.1f} MB)")
                last_time = current_time
            
            time.sleep(10)  # Kontrola každých 10 sekund
            
    except KeyboardInterrupt:
        print("\n🛑 Monitorování zastaveno")

def main():
    print("Vyberte akci:")
    print("1. Jednorázová kontrola")
    print("2. Průběžné monitorování")
    
    choice = input("Vaše volba (1/2): ").strip()
    
    if choice == "1":
        progress = get_training_progress()
        if isinstance(progress, str):
            print(f"❌ {progress}")
        else:
            print(f"✅ Poslední checkpoint: {progress['checkpoint']}")
            print(f"   Čas: {progress['last_modified']}")
            print(f"   Velikost: {progress['size_mb']:.1f} MB")
    
    elif choice == "2":
        monitor_training()
    
    else:
        print("❌ Neplatná volba")

if __name__ == "__main__":
    main() 