import subprocess
import sys

if __name__ == "__main__":
    print("[main] Running training script...")
    result = subprocess.run([sys.executable, "train.py"], check=True)
    print("[main] Training complete. Running plotting script...")
    result = subprocess.run([sys.executable, "plotting.py"], check=True)
    print("[main] All done. Plots saved.") 