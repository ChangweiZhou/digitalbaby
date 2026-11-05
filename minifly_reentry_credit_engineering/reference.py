"""Independent Torch reverse-mode diagnostic on the same graph."""
import numpy as np
import torch


def gradient(model, examples, window):
    theta = torch.tensor(model.theta, requires_grad=True)
    head = torch.tensor(model.head, requires_grad=True)
    h, memory = torch.tensor(model.h), torch.tensor(model.memory)
    U, R = theta[:, :128].reshape(8, 8, 16), theta[:, 128:192].reshape(8, 8, 8)
    b = theta[:, 192:200]
    losses = []
    for index, (x, y) in enumerate(examples):
        if index % window == 0:
            h, memory = h.detach(), memory.detach()
        sensory = torch.einsum('bij,bj->bi', U, torch.tensor(x)) + b
        for _ in range(3):
            q = h / torch.sqrt(1 + (h * h).sum(1, keepdim=True))
            r = torch.einsum('bij,bj->bi', memory, q)
            h = 0.75 * h + 0.25 * torch.tanh(sensory + torch.einsum('bij,bj->bi', R, h) + model.route * r)
        q = h / torch.sqrt(1 + (h * h).sum(1, keepdim=True))
        r = torch.einsum('bij,bj->bi', memory, q)
        features = torch.cat((h, r), 1).flatten()
        logp = torch.log_softmax(head[:, :128] @ features + head[:, 128], dim=0)
        losses.append(-logp[y])
        v = torch.tensor(model.values[y])
        alpha = 0.10 * torch.sigmoid((theta[:, 200:208] * h + theta[:, 208:216] * v).sum(1) + theta[:, 216])
        memory = memory + alpha[:, None, None] * (v - r)[:, :, None] * q[:, None, :]
        scale = torch.linalg.vector_norm(memory, dim=(1, 2)) / 4
        memory = memory / torch.where(scale > 1, scale, torch.ones_like(scale))[:, None, None]
    loss = torch.stack(losses).sum()
    gtheta, ghead = torch.autograd.grad(loss, (theta, head))
    return float(loss.detach()), gtheta.detach().numpy(), ghead.detach().numpy(), h.detach().numpy(), memory.detach().numpy()


def train_block(model, symbols):
    """Train a BPTT32 block or the terminal partial block."""
    if not 0 < len(symbols) <= 32 or model.sites % 32:
        raise ValueError("BPTT32 requires a nonempty aligned block")
    fe = model.fe.copy()
    examples = []
    for y in symbols:
        examples.append((model.sensory(), y))
        model.advance_sensor(y)
    end_fe = model.fe.copy()
    model.fe = fe
    loss, gt, go, h, memory = gradient(model, examples, 32)
    model.h, model.memory, model.fe = h, memory, end_fe
    model.gtheta, model.ghead = gt, go
    model.sites += len(symbols)
    if len(symbols) == 32:
        model.apply_update()
    return loss
