# Hessian eigenvalue utilities
import torch

def compute_top_hessian_eigval(model, loss_fn, inputs, targets, max_iters=100, return_vector=False):
    # Compute top Hessian eigenvalue using power iteration
    model.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad])

    def hessian_vector_product(v):
        # Compute Hessian-vector product
        grad_dot_v = torch.dot(grad_vec, v)
        hv = torch.autograd.grad(grad_dot_v, model.parameters(), retain_graph=True)
        hv_vec = torch.cat([g.contiguous().view(-1) for g in hv])
        return hv_vec

    v = torch.randn_like(grad_vec)
    v = v / v.norm()
    for _ in range(max_iters):
        hv = hessian_vector_product(v)
        hv_norm = hv.norm()
        v = hv / hv_norm
    eigval = torch.dot(v, hessian_vector_product(v)).item()
    if return_vector:
        return eigval, v.detach().cpu()
    return eigval


def compute_hessian_spectrum(model, loss_fn, inputs, targets, k=10):
    """
    Compute the top k eigenvalues of the Hessian using torch.autograd.functional.hessian (for small models).
    Returns eigenvalues sorted in descending order.
    """
    model.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    params = [p for p in model.parameters() if p.requires_grad]
    flat_params = torch.cat([p.contiguous().view(-1) for p in params])

    def closure(flat_params):
        idx = 0
        new_params = []
        for p in params:
            numel = p.numel()
            new_params.append(flat_params[idx:idx+numel].view_as(p))
            idx += numel
        # Temporarily assign new params
        with torch.no_grad():
            for p, new_p in zip(params, new_params):
                p.copy_(new_p)
        logits = model(inputs)
        return loss_fn(logits, targets)

    # Compute full Hessian (only feasible for small models)
    hess = torch.autograd.functional.hessian(closure, flat_params)
    hess = hess.detach().cpu().reshape(flat_params.numel(), flat_params.numel())
    eigvals = torch.linalg.eigvalsh(hess)
    eigvals = eigvals.numpy()[::-1]  # descending order
    return eigvals[:k]