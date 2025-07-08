#!/usr/bin/env python3
"""
Skript pro bezpečné zastavení trénování
"""

import os
import signal
import psutil
import time

def find_training_process():
    """Najde proces trénování Python skriptu."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train.py' in cmdline:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def stop_training():
    """Bezpečně zastaví trénování."""
    print("🔍 Hledám proces trénování...")
    
    proc = find_training_process()
    if proc is None:
        print("❌ Nenalezen žádný proces trénování")
        return
    
    print(f"✅ Nalezen proces trénování (PID: {proc.pid})")
    print(f"   Příkaz: {' '.join(proc.cmdline())}")
    
    try:
        # Pokus o graceful shutdown
        print("🛑 Zastavuji trénování...")
        proc.terminate()
        
        # Počkej 5 sekund na graceful shutdown
        proc.wait(timeout=5)
        print("✅ Trénování bylo úspěšně zastaveno")
        
    except psutil.TimeoutExpired:
        print("⚠️  Proces se nezastavil, force kill...")
        proc.kill()
        print("✅ Proces byl force killed")
        
    except Exception as e:
        print(f"❌ Chyba při zastavování: {e}")

def main():
    print("🛑 ZASTAVENÍ TRÉNOVÁNÍ LLM")
    print("=" * 40)
    
    stop_training()
    
    print("\n💡 Tip: Pro restart s novými parametry spusťte:")
    print("   python training/train.py")

if __name__ == "__main__":
    main() 