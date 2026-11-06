"""Independent reverse-time diagnostic for the selected fixed graph."""
import torch


def gradient(model, features, symbols, weights):
    cell = torch.tensor(model.cell, requires_grad=True)
    head = torch.tensor(model.head, requires_grad=True)
    h = torch.tensor(model.h)
    index = torch.tensor(model.incoming)
    x = torch.tensor(features)
    ys = torch.tensor(symbols, dtype=torch.long)
    w = torch.tensor(weights)
    xs = x @ cell[:, :, :24].transpose(-2, -1) + cell[:, None, :, -2]
    outputs = []
    for t in range(len(symbols)):
        outputs.append(head[:, :-1] @ h + head[:, -1])
        recurrent = (cell[:, :, 24:-2] * h[index][None]).sum(-1) + cell[:, :, -1]
        reset = torch.sigmoid(xs[0, t] + recurrent[0])
        update = torch.sigmoid(xs[1, t] + recurrent[1])
        candidate = torch.tanh(xs[2, t] + reset * recurrent[2])
        h = (1 - update) * candidate + update * h
    losses = torch.nn.functional.cross_entropy(torch.stack(outputs), ys, reduction='none')
    loss = (w * losses).sum()
    gc, gh = torch.autograd.grad(loss, (cell, head))
    return float(loss.detach()), gc.detach().numpy(), gh.detach().numpy(), h.detach().numpy()
