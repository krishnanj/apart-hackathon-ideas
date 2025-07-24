# Training script for grokking Hessian experiment
from model import MLP
from dataset import create_mod_add_dataset
from hessian_utils import compute_top_hessian_eigval
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
p = 97
# Prepare data
train_x, train_y, test_x, test_y = create_mod_add_dataset(p=p)

train_x = train_x.float().to(device)
test_x = test_x.float().to(device)
train_y = train_y.to(device)
test_y = test_y.to(device)

depths = [1, 2, 3, 4, 5, 6, 7]

for n_layers in depths:
    # Init model and optimizer
    model = MLP(n_hidden_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = lambda logits, y: F.cross_entropy(logits, y)

    hess_trace = []
    train_loss_trace = []
    test_loss_trace = []
    test_acc_trace = []
    grad_norm_trace = []
    eigvecs = []
    grokking_step = None

    for step in range(10000):
        # Training step
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            # Log train/test loss and accuracy
            with torch.no_grad():
                train_loss = loss_fn(model(train_x), train_y).item()
                test_logits = model(test_x)
                test_loss = loss_fn(test_logits, test_y).item()
                test_pred = torch.argmax(test_logits, dim=1)
                test_acc = (test_pred == test_y).float().mean().item()
            train_loss_trace.append(train_loss)
            test_loss_trace.append(test_loss)
            test_acc_trace.append(test_acc)
            if grokking_step is None and test_acc >= 0.9:
                grokking_step = step

        if step % 1000 == 0:
            # Log Hessian eig, eigenvector, and grad norm
            eig, v = compute_top_hessian_eigval(model, loss_fn, train_x, train_y, return_vector=True)
            hess_trace.append(eig)
            eigvecs.append(v.numpy())
            model.zero_grad()
            logits = model(train_x)
            loss = loss_fn(logits, train_y)
            loss.backward()
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            grad_norm_trace.append(grad_norm)
            print(f"[{step}] depth={n_layers}, top-eig={eig:.2f}, train-loss={train_loss:.2f}, test-acc={test_acc:.2f}")

    # Save results
    torch.save({
        "train": train_loss_trace,
        "test": test_loss_trace,
        "test_acc": test_acc_trace,
        "hess": hess_trace,
        "grad_norm": grad_norm_trace,
        "eigvecs": eigvecs,
        "grokking_step": grokking_step
    }, f"results_depth{n_layers}.pt")