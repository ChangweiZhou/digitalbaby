"""The comparator on the stream of one life (roadmap, section 5, condition 5): a recurrent network trained by back-propagation to give the next
byte, reading the life's stream (PREP/life_stream.py, life(seed)) from its first byte in order, one item at a time.

    python3 comparator.py run <seed> [--bytes N] [--out DIR]      # reads the first N bytes of the life (all of it when N is omitted)

The network (all numbers are defaults of Config):
  - a table of 32 numbers for each of the 256 byte values (the code of a byte);
  - a GRU of 128 units (torch's nn.GRU, one layer), whose state is carried from one item to the next;
  - a linear readout from the 128 units to 256 classes, one for each byte value.

Reading. The stream is cut into items, an item ending at the end-mark byte 10 (a stretch without an end mark, such as a quiet stretch of
4,096 spaces, is one item). For an item that covers positions s to e, the network reads the bytes at positions s - 1 to e - 1 (at position
0 the byte 10, as if an item had just ended) and gives, at each position p, its output after reading the byte at p - 1. The byte given at p
is the most likely byte of that output; it is recorded before the update that uses the item, so that it is the network's answer before
the byte at p is shown. The loss is the mean cross entropy of the item's outputs against the bytes at s to e; one optimiser step follows
each item. The state at the end of an item is carried into the next item and detached, so that the gradient is cut at the item's end.

Optimiser: torch.optim.AdamW, rate 0.001, betas (0.9, 0.98), eps 1e-8, as in the project's record (exp26_candidate.make_opt: weight decay
0.25 on every weight but the byte table, 0 on the byte table). The weight decay is not named in the reviewer's message; it is the record's,
and Config.weight_decay changes it.

Counts. Items read; multiply-adds of the forward pass (input map, recurrent map and readout at each position) and of the backward pass,
counted as twice the forward's; the optimiser's work is counted separately as one update of each parameter per step.

Outputs in DIR: comparator_<seed>_<N>.npz with `given` (uint8, the byte given at every position), `shown` (the life's bytes), `prob`
(float32, the probability given to the byte shown), `loss_item` and `item_end` (per item), and comparator_<seed>_<N>.json with the
settings, the counts, the CPU time and the forecast for longer reads.
"""
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch as T
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import life_stream as LS          # noqa: E402

END = 10


@dataclass(frozen=True)
class Config:
    code: int = 32
    units: int = 128
    lr: float = 0.001
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-8
    weight_decay: float = 0.25
    torch_seed: int = 0


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.book = nn.Embedding(256, cfg.code)
        self.rnn = nn.GRU(cfg.code, cfg.units, num_layers=1, batch_first=True)
        self.out = nn.Linear(cfg.units, 256)

    def forward(self, x, h):
        y, h = self.rnn(self.book(x)[None], h)
        return self.out(y[0]), h


def macs_forward(cfg):
    """Multiply-adds of the forward pass for one position: the input and recurrent maps of the three gates, the readout."""
    return 3 * (cfg.code * cfg.units + cfg.units * cfg.units) + cfg.units * 256


def items_of(stream):
    ends = np.flatnonzero(stream == END)
    starts = np.concatenate([[0], ends + 1])
    stops = np.concatenate([ends, [len(stream) - 1]])
    return [(int(s), int(e)) for s, e in zip(starts, stops) if s <= e]


def run(seed, nbytes, out, cmd, cfg=Config()):
    T.manual_seed(cfg.torch_seed); T.set_num_threads(1)
    L = LS.life(seed)
    stream = L["bytes"] if nbytes is None else L["bytes"][:nbytes]
    n = len(stream)
    net = Net(cfg)
    decay = [p for name, p in net.named_parameters() if not name.startswith("book.")]
    opt = T.optim.AdamW([dict(params=decay, weight_decay=cfg.weight_decay), dict(params=list(net.book.parameters()), weight_decay=0.0)],
                        lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    nparams = sum(p.numel() for p in net.parameters())
    given = np.zeros(n, dtype=np.uint8); prob = np.zeros(n, dtype=np.float32)
    its = items_of(stream); loss_item = np.zeros(len(its), dtype=np.float32); item_end = np.array([e for _, e in its], dtype=np.int64)
    h = T.zeros(1, 1, cfg.units)
    s_all = T.from_numpy(stream.astype(np.int64))
    t0 = time.process_time(); w0 = time.time()
    for i, (s, e) in enumerate(its):
        x = T.cat([s_all[s - 1:s] if s > 0 else T.tensor([END]), s_all[s:e]])
        y = s_all[s:e + 1]
        logits, h_new = net(x, h)
        with T.no_grad():
            pr = T.softmax(logits, 1)
            given[s:e + 1] = pr.argmax(1).numpy().astype(np.uint8)
            prob[s:e + 1] = pr[T.arange(len(y)), y].numpy()
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_item[i] = float(loss)
        h = h_new.detach()
    cpu = time.process_time() - t0; wall = time.time() - w0
    mf = macs_forward(cfg)
    counts = dict(bytes=n, items=len(its), longest_item=max(e - s + 1 for s, e in its), parameters=nparams, macs_forward_a_byte=mf,
                  macs_forward=mf * n, macs_backward=2 * mf * n, macs_total=3 * mf * n, optimiser_steps=len(its), optimiser_parameter_updates=nparams * len(its))
    right = int((given == stream).sum())
    full = len(L["bytes"])
    per = cpu / n
    fc = {"whole life": dict(bytes=full, cpu_seconds=per * full, cpu_hours=per * full / 3600, macs_total=3 * mf * full),
          "first 262,144 bytes": dict(bytes=262144, cpu_seconds=per * 262144, cpu_hours=per * 262144 / 3600, macs_total=3 * mf * 262144)}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tag = f"comparator_{seed}_{n}"
    np.savez_compressed(out / f"{tag}.npz", given=given, shown=stream, prob=prob, loss_item=loss_item, item_end=item_end)
    rec = dict(kind="comparator", seed=seed, command=cmd, config=asdict(cfg), life_config=asdict(L["config"]) if "config" in L else None,
               counts=counts, right_first_attempt=right, right_share=right / n, mean_loss_first_100_items=float(loss_item[:100].mean()),
               mean_loss_last_100_items=float(loss_item[-100:].mean()), cpu_seconds=cpu, wall_seconds=wall, cpu_seconds_a_byte=per,
               forecast=fc, torch=T.__version__, threads=T.get_num_threads(), record=f"{tag}.npz")
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("life_config",)}, indent=1))
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = " ".join(["python3", "comparator.py", *a])
    if a and a[0] == "run":
        nb = int(a[a.index("--bytes") + 1]) if "--bytes" in a else None
        od = a[a.index("--out") + 1] if "--out" in a else str(HERE / "comparator_runs")
        run(int(a[1]), nb, od, cmd)
    else:
        raise SystemExit(__doc__)
