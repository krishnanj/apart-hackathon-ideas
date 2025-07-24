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