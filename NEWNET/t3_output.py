"""Test 2 (roadmap step 2): the output end learns from feedback to give the next byte.

The tissue is net2.Tissue (regional winner-take-k, learned recurrence). The output end: one cell for each byte value, each with
FAN_OUT connections from random cells of the last region, strengths starting at 0. At every step the output cell with the largest
input gives its byte (none if every input is 0). When the next byte arrives, feedback reaches the output end:
  teach:  the feedback line of the right byte reaches that byte's output cell (the right answer is given, as the owner allows);
          the cell that gave a wrong byte lowers, and the right byte's cell raises, the connections that were active:
          dw = eta * pre * (target - gave), each cell from its own target line and its own output only;
  yesno:  only "yes" or "no" reaches the cell that gave the byte; it raises (yes) or lowers (no) its active connections;
          with chance EPS the output end gives a byte at random instead (it must try to find out).
Streams: 'cycle' (P distinct bytes repeated), 'random' (independent bytes: nothing to learn, the control).
norm 1: each output cell's total strength is held at 1 by subtracting the excess from all its connections alike.
Reported: the share of steps at which the byte given is the next byte, in successive quarters."""
import sys
import numpy as np
from net2 import Tissue

def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
T, P, mode, stream_kind = a.pop("T", 8000), a.pop("P", 7), a.pop("mode", "teach"), a.pop("stream", "cycle")
eta, eps, fan_out, seed = a.pop("eta", 0.1), a.pop("eps", 0.1), a.pop("fan_out", 64), a.get("seed", 1)
norm = a.pop("norm", 1)
net = Tissue(**a)
rng = np.random.default_rng(seed + 1000)
V = net.n_values
if stream_kind == "cycle":
    pat = rng.choice(V, P, replace=False)
    stream = np.array([pat[t % P] for t in range(T + 1)])
else:
    stream = rng.integers(0, V, T + 1)
last = np.arange((net.L - 1) * net.R, net.L * net.R)
osrc = rng.choice(last, (V, fan_out))
ow = np.zeros((V, fan_out))
right = np.zeros(T, bool)
for t in range(T):
    x = net.step(stream[t])
    pre = x[osrc]                                       # [V, fan_out]: the activity at each output cell's connections
    O = (ow * pre).sum(1)
    gave = int(np.argmax(O)) if O.max() > 0 else int(rng.integers(0, V))   # with nothing to go on, it babbles
    if mode == "yesno" and rng.random() < eps:
        gave = int(rng.integers(0, V))
    nxt = stream[t + 1]
    right[t] = gave == nxt
    if mode == "teach":
        tgt = np.zeros(V); tgt[nxt] = 1.0
        g = np.zeros(V)
        if gave >= 0: g[gave] = 1.0
        ow += eta * pre * (tgt - g)[:, None]
    else:
        if gave >= 0:
            ow[gave] += eta * pre[gave] * (1.0 if gave == nxt else -1.0)
    np.clip(ow, 0.0, 4.0, out=ow)
    if norm:                                            # each output cell's total incoming strength held at 1 (subtractive): inputs compete
        ow += (1.0 - ow.sum(1, keepdims=True)) / fan_out
        np.clip(ow, 0.0, 4.0, out=ow)
q = T // 4
print(f"{mode:5s} stream={stream_kind:6s} regions={net.L} P={P}: right share by quarter " + " ".join(f"{right[i*q:(i+1)*q].mean():.3f}" for i in range(4))
      + f" | chance {1/P if stream_kind=='cycle' else 1/V:.3f}")
