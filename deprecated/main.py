import subprocess
import sys
import os

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("[main] Running training script...")
    result = subprocess.run([sys.executable, os.path.join(here, "train.py")], check=True)
    print("[main] Training complete. Running plotting script...")
    result = subprocess.run([sys.executable, os.path.join(here, "plotting.py")], check=True)
    print("[main] All done. Plots saved.") 