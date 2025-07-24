# load_params.py
import yaml

def load_params(filepath="params.yaml"):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)