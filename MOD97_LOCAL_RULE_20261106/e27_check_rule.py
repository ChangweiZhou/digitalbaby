"""E27, follow-up item 5: is E27's two-bank sign path the rule E24's sweep births were made under (birthrule T, traceup all)?

Run with the births' own settings in the environment (E16_BIRTHRULE=T E16_TRACEUP=all, plus E16_LOSS / E16_LOSSW where the birth used them),
so that exp16_integrated installs exactly the forward its births used: E.forward(m, x, local=True, through_time forced True, keep forced
False), trace routines disabled. This script then reproduces update16's loss and backward pass, up to but NOT including the step; no
update rule is constructed and no weight changes. The gradient is compared with e25_alignment.grads(m, x, y, mask, True), E27's sign path.

  E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 E16_BIRTHRULE=T E16_TRACEUP=all [E16_LOSS=all [E16_LOSSW=balanced]] python3 e27_check_rule.py <file> <seed>"""
import sys, json, os
import torch as T
import e25_alignment as A, e27_measure as M27
G, E, X = A.G, A.E, A.X


def main(path, seed):
    T.set_num_threads(1)
    assert G.BIRTHRULE16 == "T" and G.TRACEUP16 == "all" and G.FEEDBACK16 == "sign" and G.HEADS16 == "top"
    m, rec = M27.net("G0", path, seed); birth, _ = M27.panels(seed)
    x, y, amask = X.batch_tensor(birth)
    every = y != E.BOS; mask = every if G.LOSS16 == "all" else amask
    # update16, lines 620-644, without the step
    m.zero_grad(); logits, it = E.forward(m, x, local=True, through_time=False, keep=True)
    if G.LOSSW16 == "balanced":
        other = mask & ~amask; loss = .5 * E.cross_entropy(logits[amask], y[amask]) + .5 * E.cross_entropy(logits[other], y[other])
    else:
        loss = E.cross_entropy(logits[mask], y[mask])
    loss.backward()
    births = {n: (p.grad.detach().clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}
    # E27's sign path, with the same weighting of the loss
    if G.LOSSW16 == "balanced":
        _, ga, _, _ = A.grads(m, x, y, amask, True); _, go, _, _ = A.grads(m, x, y, mask & ~amask, True)
        e27 = {n: .5 * ga[n] + .5 * go[n] for n in ga}
    else:
        _, e27, _, _ = A.grads(m, x, y, mask, True)
    rel = max(float((births[n] - e27[n]).abs().max()) / max(float(e27[n].abs().max()), 1e-12) for n in e27)
    # and the uniform every-byte loss E27 measured, against the balanced loss this birth actually took (cosine, all parameters)
    _, gu, _, _ = A.grads(m, x, y, every, True)
    v = lambda g: T.cat([t.flatten() for t in g.values()])
    out = dict(file=path, receipt=rec, loss=G.LOSS16, lossw=G.LOSSW16, birth_forward=f"{E.forward.__module__ if hasattr(E.forward, '__module__') else E.forward}",
               max_relative_difference_birth_rule_vs_e27_sign_path=rel, same=rel < 1e-5,
               cosine_birth_loss_vs_uniform_every_byte=A.cosine(v(births), v(gu)))
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
