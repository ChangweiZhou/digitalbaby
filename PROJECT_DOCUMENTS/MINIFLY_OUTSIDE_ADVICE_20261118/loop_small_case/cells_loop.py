#!/usr/bin/env python3
"""The loop at the smallest size (PLAN.md beside this file): cells in between that change their own strengths from a mismatch
formed at the answering cell and relayed to them over return connections. numpy and the standard library only.

The text and the score are those of the reviewer's yesno/addtext.py, written again here: lines "a+b=c" with an end mark, a and b
from 0 to 9, 20 of the 100 pairs never shown (the same 20 as addtext.py draws for the seed), read one byte at a time; at every
byte but the end mark the system names the next byte before it arrives. One stream, no batch, nothing reset.

    python3 cells_loop.py check                      the comparator's change against central differences; the return connections
    python3 cells_loop.py run <label> arm=relay-learned seed=1 H=32 eta1=1/4 eta2=1/16 lam=1e-5 [lines=60000] [rule=D|Q] [order=0]
    python3 cells_loop.py dev                        the grid of PLAN.md section 5 on the development seed (res/dev_*.json)
    python3 cells_loop.py report <setting options>   the six arms on the five reported seeds (res/rep_*.json) and the table

Arms: none | fixed | relay-fixed | relay-learned | comparator.  rule=D: the mismatch from the byte that arrives; rule=Q: from yes
or no only (the byte named moves toward 1 or 0; on yes the others toward 0; with chance 1/16 while learning a byte seen so far is
named at random).
"""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n")
S = len(SYMS)                       # byte values in use; a line for each (the other 243 lines are never active)
SI = {c: i for i, c in enumerate(SYMS)}
END = SI[10]
D = 5                               # the present byte and four delay cells (text d1); set by configure() for text d2
NL = D * S
OFF = np.arange(D) * S
TEXT = "d1"                         # d1: lines "a+b=c", a and b 0 to 9 (PLAN.md); d2: "aa+bb=ccc", a and b 0 to 99 at fixed width (PLAN_2.md)


def configure(text):
    global D, NL, OFF, TEXT
    TEXT = text
    D = 5 if text == "d1" else 10    # d2: a line has ten bytes, so the present byte and nine delay cells
    NL = D * S
    OFF = np.arange(D) * S
SEEDS = (73129, 93683, 116089, 406364, 545353)
ARMS = ("none", "fixed", "relay-fixed", "relay-learned", "comparator")


def text_of(a, b):
    if TEXT == "d1":
        return [SI[c] for c in f"{a}+{b}={a + b}\n".encode()]
    return [SI[c] for c in f"{a:02d}+{b:02d}={a + b:03d}\n".encode()]


def pairs_of(seed, held=20):
    """The pairs never shown and shown, and the generator of the stream. Text d1: as addtext.py makes them. Text d2: of the
    10,000 pairs, 2,000 drawn by the seed are never shown."""
    if TEXT == "d1":
        rng = np.random.default_rng(seed)
        pairs = [(a, b) for a in range(10) for b in range(10)]
        rng.shuffle(pairs)
        return rng, pairs[:held], pairs[held:]
    rng = np.random.default_rng([seed, 99])
    order = rng.permutation(10000)
    pairs = [(int(q) // 100, int(q) % 100) for q in order]
    return rng, pairs[:2000], pairs[2000:]


class System:
    def __init__(self, arm, H, seed, eta1, eta2, lam, rule="D", fan=0, lam1=0.0, w2r=0):
        self.arm, self.H, self.eta1, self.eta2, self.lam, self.rule, self.lam1 = arm, H, eta1, eta2, lam, rule, lam1
        wr = np.random.default_rng([seed, 20261119])                 # the birth
        self.xr = np.random.default_rng([seed, 20261120])            # the random choices of rule Q
        nin = NL if arm == "none" else H
        self.W2 = np.zeros((S, nin)); self.b2 = np.zeros(S)          # the answering cell: one branch a byte value
        if arm != "none":
            self.W1 = wr.uniform(-0.5, 0.5, (H, NL)); self.b1 = np.zeros(H)
            self.R = wr.uniform(-1.0, 1.0, (H, S))                   # return connections, at their birth values
            self.mask = None
            if fan:                                                  # each cell in between reads `fan` of the 65 lines
                self.mask = np.zeros((H, NL), bool)
                for j in range(H):
                    self.mask[j, wr.choice(NL, fan, replace=False)] = True
                self.W1 *= self.mask
        if w2r and arm != "none":                                    # option: the answering cell's strengths start at the return connections' birth values
            self.W2 = self.R.T.copy()
        self.hist = [END] * D                                        # the present byte and the delay cells
        self.seen = np.zeros(S, bool)

    def read(self, byte):
        self.hist.insert(0, byte); self.hist.pop()
        self.seen[byte] = True

    def forward(self, hist):
        act = OFF + np.asarray(hist)
        if self.arm == "none":
            return act, None, self.W2[:, act].sum(1) + self.b2
        h = 1.0 / (1.0 + np.exp(-(self.W1[:, act].sum(1) + self.b1)))
        return act, h, self.W2 @ h + self.b2

    def attempt_and_learn(self, y):
        """The system names the next byte; the byte y arrives; the cells change. Returns whether the byte named was right."""
        act, h, a = self.forward(self.hist)
        e = int(a.argmax())
        if self.rule == "Q" and self.xr.random() < 1 / 16:
            sv = np.flatnonzero(self.seen); e = int(sv[self.xr.integers(len(sv))])
        ok = e == y
        if self.rule == "D":
            m = -a; m[y] += 1.0
        else:
            m = np.zeros(S); m[e] = float(ok) - a[e]
            if ok:
                m = -a; m[e] = 1.0 - a[e]
        if self.arm == "none":
            g = self.eta2 / (D + 1.0)
            self.W2[:, act] += (g * m)[:, None]; self.b2 += g * m
            if self.lam: self.W2 *= 1.0 - self.lam
            return ok
        g = self.eta2 / (float(h @ h) + 1.0)
        if self.arm in ("relay-fixed", "relay-learned", "comparator"):
            r = (self.W2.T if self.arm == "comparator" else self.R) @ m          # what reaches each cell in between
            d = self.eta1 * r * h * (1.0 - h)
            if self.mask is None:
                self.W1[:, act] += d[:, None]
            else:
                self.W1[:, act] += d[:, None] * self.mask[:, act]
            self.b1 += d
            if self.lam1:                                            # each cell in between lets all its strengths fade a little at every byte
                self.W1 *= 1.0 - self.lam1
        self.W2 += np.outer(g * m, h); self.b2 += g * m
        if self.arm == "relay-learned":
            self.R += np.outer(h, g * m)                                         # the same product as the forward strength
            if self.lam: self.R *= 1.0 - self.lam
        if self.lam: self.W2 *= 1.0 - self.lam
        return ok

    def score(self, pairs):
        """Every pair once, learning off, no random choice. Returns the share with the whole answer right (the bytes after '='
        up to the end mark), the same with the end mark included, and the pairs' own results."""
        whole, strict, per = 0, 0, {}
        for a, b in pairs:
            line = text_of(a, b); eq = line.index(SI[ord("=")]); hist = [END] * D
            w = s = True
            for t in range(len(line) - 1):
                hist.insert(0, line[t]); hist.pop()
                e = int(self.forward(hist)[2].argmax()); y = line[t + 1]
                if t >= eq:
                    s = s and e == y
                    if y != END: w = w and e == y
            whole += w; strict += s; per[f"{a}+{b}"] = bool(w)
        return whole / len(pairs), strict / len(pairs), per


def _by_place(self, pairs):
    """Text d2: the share of pairs with each byte of the answer named right (hundreds, tens, units, the end mark)."""
    right = np.zeros(4)
    for a, b in pairs:
        line = text_of(a, b); eq = line.index(SI[ord("=")]); hist = [END] * D
        for t in range(len(line) - 1):
            hist.insert(0, line[t]); hist.pop()
            if t >= eq:
                right[t - eq] += int(self.forward(hist)[2].argmax()) == line[t + 1]
    return [float(x) for x in right / len(pairs)]


System.by_place = _by_place


def order_of_digits(W1):
    """For each cell in between and each delay: the correlation of its ten strengths from the digit lines with the digits, and
    the slope. A cell is counted as ordered for a pair of delays when both correlations are at least 0.9 in size with one sign."""
    dig = np.arange(10.0); dc = dig - dig.mean()
    out = {}
    for k in range(D):
        w = W1[:, k * S: k * S + 10]
        wc = w - w.mean(1, keepdims=True)
        out[k] = ((wc @ dc) / (np.linalg.norm(wc, axis=1) * np.linalg.norm(dc) + 1e-30), (wc @ dc) / (dc @ dc))
    res = {}
    for name, (ka, kb) in (("first answer byte (first number 3 back, second 1 back)", (3, 1)), ("second answer byte (4 back, 2 back)", (4, 2))):
        ra, sa = out[ka]; rb, sb = out[kb]
        both = (np.abs(ra) >= 0.9) & (np.abs(rb) >= 0.9) & (np.sign(ra) == np.sign(rb))
        ratio = np.sort(sb[both] / sa[both]) if both.any() else np.array([])
        res[name] = dict(cells_ordered_at_both=int(both.sum()), slope_ratios=[round(float(x), 3) for x in ratio],
                         abs_r_first_sorted=[round(float(x), 3) for x in np.sort(np.abs(ra))[::-1][:8]],
                         abs_r_second_sorted=[round(float(x), 3) for x in np.sort(np.abs(rb))[::-1][:8]])
    return res


def run(kw):
    t0 = time.process_time()
    configure(kw.get("text", "d1"))
    arm = kw.get("arm", "relay-learned"); seed = int(kw.get("seed", 1)); H = int(kw.get("H", 32)); lines = int(kw.get("lines", 60000))
    eta1 = float(eval(str(kw.get("eta1", "1/4")))); eta2 = float(eval(str(kw.get("eta2", "1/16")))); lam = float(eval(str(kw.get("lam", "1e-5"))))
    rule = kw.get("rule", "D"); order = int(kw.get("order", 0)); fan = int(kw.get("fan", 0)); lam1 = float(eval(str(kw.get("lam1", "0"))))
    rng, held, shown = pairs_of(seed)
    if order:
        rng = np.random.default_rng([seed, order])
    sys_ = System(arm, H, seed, eta1, eta2, lam, rule, fan, lam1, int(kw.get("w2r", 0)))
    every = int(kw.get("every", 5000))
    if TEXT == "d2":                                                 # scored on 500 pairs of each kind, fixed by the seed
        sr = np.random.default_rng([seed, 500])
        shown_s = [shown[i] for i in sr.choice(len(shown), 500, replace=False)]; held_s = [held[i] for i in sr.choice(len(held), 500, replace=False)]
    else:
        shown_s, held_s = shown, held
    curve = []; right = n = 0
    for it in range(lines):
        a, b = shown[rng.integers(len(shown))]
        line = text_of(a, b)
        for t in range(len(line) - 1):
            sys_.read(line[t]); right += sys_.attempt_and_learn(line[t + 1]); n += 1
        sys_.read(line[-1])                                          # the end mark is followed by the next line's first byte: no attempt
        if (it + 1) % every == 0:
            ws, ss, _ = sys_.score(shown_s); wh, sh, _ = sys_.score(held_s)
            curve.append(dict(lines=it + 1, shown=ws, never_shown=wh, shown_with_end=ss, never_shown_with_end=sh, bytes_right_while_learning=right / n))
            right = n = 0
    ws, ss, _ = sys_.score(shown_s); wh, sh, per = sys_.score(held_s)
    shown_sums = {a + b for a, b in shown}
    places = sys_.by_place(held_s) if TEXT == "d2" else None
    out = dict(label=kw.get("label", ""), args={k: v for k, v in kw.items() if k != "label"}, arm=arm, seed=seed, H=H, rule=rule, lines=lines,
               shown=ws, never_shown=wh, shown_with_end=ss, never_shown_with_end=sh, curve=curve,
               never_shown_pairs=({k: dict(right=v, sum_shown=(eval(k) in shown_sums)) for k, v in per.items()} if TEXT == "d1" else None),
               never_shown_by_place=places, text=TEXT,
               strengths=(0 if arm == "none" else sys_.W1.size + sys_.b1.size + (sys_.R.size if arm in ("relay-fixed", "relay-learned") else 0)) + sys_.W2.size + sys_.b2.size,
               cpu_seconds=round(time.process_time() - t0, 1))
    if len(curve) >= 3:                                              # the reading of PLAN_2.md: the mean of the last three readings
        out["last3_shown"] = float(np.mean([c["shown"] for c in curve[-3:]])); out["last3_never_shown"] = float(np.mean([c["never_shown"] for c in curve[-3:]]))
    if arm != "none" and TEXT == "d1":
        out["order_of_digits"] = order_of_digits(sys_.W1)
    if arm != "none":
        if arm == "relay-learned":
            out["cosine_return_forward"] = float((sys_.R * sys_.W2.T).sum() / (np.linalg.norm(sys_.R) * np.linalg.norm(sys_.W2) + 1e-30))
    if kw.get("label") and kw.get("save") and arm != "none":       # the learned strengths, for look_inside.py
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
        np.savez_compressed(os.path.join(HERE, "res", kw["label"] + ".npz"), W1=sys_.W1.astype(np.float32), b1=sys_.b1.astype(np.float32),
                            W2=sys_.W2.astype(np.float32), b2=sys_.b2.astype(np.float32))
    if kw.get("label"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
        json.dump(out, open(os.path.join(HERE, "res", kw["label"] + ".json"), "w"), indent=1)
    return out


def check():
    """The comparator's change of a cell in between is minus the slope of the squared mismatch; and the learned return connections
    come to agree with the forward strengths."""
    s = System("comparator", 8, 5, 1.0, 1.0, 0.0)
    rng = np.random.default_rng(0)
    s.W2 = rng.normal(0, 0.5, s.W2.shape); s.b2 = rng.normal(0, 0.5, S)
    hist = [3, 10, 7, 12, 11]; y = 4
    def loss():
        _, h, a = s.forward(hist); t = np.zeros(S); t[y] = 1.0
        return 0.5 * ((t - a) ** 2).sum()
    act, h, a = s.forward(hist); m = -a.copy(); m[y] += 1.0
    ana = (s.W2.T @ m) * h * (1 - h)                                  # the comparator's change of each active strength at eta1 = 1
    worst = 0.0
    for j in range(8):
        for i in act[:2]:
            w0 = s.W1[j, i]; eps = 1e-6
            s.W1[j, i] = w0 + eps; lp = loss(); s.W1[j, i] = w0 - eps; lm = loss(); s.W1[j, i] = w0
            num = -(lp - lm) / (2 * eps)
            worst = max(worst, abs(num - ana[j]) / max(abs(num), abs(ana[j]), 1e-30))
    print(f"check 1: the comparator's change against central differences of half the squared mismatch, 16 strengths: largest relative difference {worst:.1e}")
    o = run(dict(arm="relay-learned", seed=1, H=16, eta1="1/4", eta2="1/16", lam="1e-4", lines=4000))
    print(f"check 2: after 4,000 lines with decay 1e-4 the cosine between the learned return connections and the forward strengths is {o['cosine_return_forward']:.4f}")


def table(rows, title):
    print(title)
    print("   arm, rule                      | whole answer right, sums shown: each seed; median | sums never shown: each seed; median | never shown, with the end mark, median | cells ordered (first; second answer byte), each seed")
    for (arm, rule), rs in rows.items():
        d2 = rs[0].get("text") == "d2"                               # text d2: each figure is the mean of the run's last three readings
        sh = [100 * r["last3_shown" if d2 else "shown"] for r in rs]; nv = [100 * r["last3_never_shown" if d2 else "never_shown"] for r in rs]; ne = [100 * r["never_shown_with_end"] for r in rs]
        od = "; ".join("/".join(str(v["cells_ordered_at_both"]) for v in r["order_of_digits"].values()) for r in rs) if "order_of_digits" in rs[0] else "-"
        if rs[0].get("never_shown_by_place"):
            od = "by place, median: " + " ".join(f"{100 * np.median([r['never_shown_by_place'][k] for r in rs]):.0f}" for k in range(4))
        print(f"   {arm + (', yes or no' if rule == 'Q' else ''):32s} | " + " ".join(f"{x:5.1f}" for x in sh) + f"; {np.median(sh):5.1f} | "
              + " ".join(f"{x:5.1f}" for x in nv) + f"; {np.median(nv):5.1f} | {np.median(ne):5.1f} | {od}")


def main():
    cmd = sys.argv[1]
    kw = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    if cmd == "check":
        check()
    elif cmd == "run":
        kw["label"] = sys.argv[2]
        o = run(kw)
        print(f"== {o['label']} {o['args']}: sums shown {100 * o['shown']:.1f}, never shown {100 * o['never_shown']:.1f} (with the end mark {100 * o['shown_with_end']:.1f}, {100 * o['never_shown_with_end']:.1f}); {o['cpu_seconds']} CPU-s")
        for c in o["curve"]:
            print(f"   {c['lines']:6d} lines: shown {100 * c['shown']:5.1f}  never shown {100 * c['never_shown']:5.1f}  bytes right while learning {100 * c['bytes_right_while_learning']:5.1f}")
    elif cmd == "dev":
        jobs = [dict(label=f"dev_H{H}_e{e1.replace('/', 'o')}_f{e2.replace('/', 'o')}_l{lam}", arm="relay-learned", seed="1", H=str(H), eta1=e1, eta2=e2, lam=lam)
                for H in (8, 16, 32, 64) for e1 in ("1/16", "1/4", "1") for e2 in ("1/16", "1/4") for lam in ("0", "1e-5", "1e-4")]
        with get_context("fork").Pool(4) as pool:
            res = pool.map(run, jobs)
        res.sort(key=lambda o: (-o["never_shown"], -o["shown"]))
        print("development seed 1, arm relay-learned: H, eta1, eta2, lambda: sums shown, never shown (whole answer right, %)")
        for o in res:
            a = o["args"]
            print(f"   H={a['H']:>2s} eta1={a['eta1']:>4s} eta2={a['eta2']:>4s} lam={a['lam']:>5s}: {100 * o['shown']:5.1f} {100 * o['never_shown']:5.1f}")
    elif cmd == "report":
        base = dict(H=kw.get("H", "32"), eta1=kw.get("eta1", "1/4"), eta2=kw.get("eta2", "1/16"), lam=kw.get("lam", "1e-5"), lines=kw.get("lines", "60000"),
                    text=kw.get("text", "d1"), every=kw.get("every", "5000"))
        base.update({k: kw[k] for k in ("w2r", "fan", "lam1") if k in kw})             # options passed through to every run
        tag = kw.get("tag", "rep")
        arms = [(a, "D") for a in ARMS] + [("relay-learned", "Q")]
        if "only" in kw:
            arms = [(a, r) for a, r in arms if a == kw["only"] and r == kw.get("rule", "D")]
        jobs = [dict(base, label=f"{tag}_{a}_{r}_{s}", arm=a, rule=r, seed=str(s)) for a, r in arms for s in SEEDS]
        with get_context("fork").Pool(4) as pool:
            res = pool.map(run, jobs)
        rows = {}
        for o in res:
            rows.setdefault((o["arm"], o["rule"]), []).append(o)
        table(rows, f"setting {base}; seeds {SEEDS}; whole answer right, %")
        ok = [o for o in res if o["arm"] == "relay-learned" and o["rule"] == "D"]
        if base["text"] != "d1":
            return
        miss = sorted({k for o in ok for k, v in o["never_shown_pairs"].items() if not v["right"] and not v["sum_shown"]})
        print(f"   relay-learned: never-shown pairs answered wrongly whose sum occurs in no pair shown: {miss or 'none'}; "
              f"never-shown pairs whose sum is shown, right: {sum(v['right'] for o in ok for v in o['never_shown_pairs'].values() if v['sum_shown'])} of "
              f"{sum(1 for o in ok for v in o['never_shown_pairs'].values() if v['sum_shown'])}")


if __name__ == "__main__":
    main()
