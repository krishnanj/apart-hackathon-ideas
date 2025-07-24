# 01-grokking-hessian

## Idea

This experiment explores how **curvature of the loss landscape (Hessian eigenvalues)** evolves during training on a grokking task, and how this behavior varies with **model depth**.

We analyze:
- The emergence (or absence) of grokking behavior.
- How top Hessian eigenvalues evolve during training.
- How **model depth affects gradient norm stability, curvature, and generalization**.

This directly probes whether **sharpness correlates with generalization** in small models trained on algorithmic tasks — a question with implications for **understanding emergent behavior in larger models**.

---

## References

- Grokking: "Learning Algorithmic Tasks by Tuning Gradient Descent"
  [NeurIPS 2022 Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)

- Base implementation: [https://github.com/teddykoker/grokking](https://github.com/teddykoker/grokking)
- Sharpness analysis inspiration: [https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability)
- Hessian top eigenvalue computation: custom autograd implementation (similar to PyHessian/BackPACK)

---

## How to Run

1. **Set up virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Train the models and generate plots:**

```bash
python main.py
```

This will train MLPs of depth 1–7 on a modular addition task (x + y mod p) and save:
- Train/test loss traces
- Top Hessian eigenvalue traces
- Gradient norm traces
- Accuracy curves

All plots will be saved in the `plots/` directory, and intermediate results in `results/`.

---

## Results

### Hessian Curvature vs. Training
- Top eigenvalues increase with model depth, suggesting deeper models enter sharper regions.
- However, deeper models (depth ≥ 4) do not necessarily grok — they show high curvature without generalization.

### Loss vs. Steps
- Shallower models (depth 2–3) successfully transition from memorization to generalization (classic grokking curve).
- Deeper models either overfit or plateau at high test loss.

### Gradient Norms
- Deeper models show noisy, unstable gradient behavior — possibly related to training instability near edge-of-stability regions.

### Test Accuracy
- Only depth 2 consistently reaches high test accuracy.
- Generalization does not monotonically improve with depth.

---

**Note:**
- You can run `python plotting.py` separately to regenerate plots from existing results.
- Adjust model/training parameters in `train.py` as needed for further experiments. 