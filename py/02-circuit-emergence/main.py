import subprocess
import sys
import os

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("[main] Training models...")
    subprocess.run([sys.executable, os.path.join(here, "train_models.py")], check=True)
    print("[main] Running probes...")
    subprocess.run([sys.executable, os.path.join(here, "run_probe.py")], check=True)
    print("[main] Plotting probe results...")
    subprocess.run([sys.executable, os.path.join(here, "plotting.py")], check=True)
    print("[main] All done.") 