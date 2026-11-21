#!/usr/bin/env python3
"""nano/multi, PLAN_2.md: the system of multi.py (single lines and pair-connections, rewiring) on the test text of nano/data.
    python3 textm.py <label> [seed=1] [H=1024] [D=10] [nline=64|all] [npair=0] [rewire=4096] [pdraw=uniform|active] [adj=0] [ans=P|LS]
                             [eta1=1] [eta2=1] [rscale=1] [passes=8] [lam=1e-5] [fast=1] [bytes=N]
The text, the split, the stream and the scoring are those of nano/loop/loop.py (branch claude/nano-loop-8mmhfm, commit e0c864c), written again here:
the first nine tenths are the stream, read `passes` times one byte at a time, each next byte named before it arrives, nothing reset; the last tenth
is never read by the learner. After each reading, with learning off, every byte of the last tenth is scored from the D bytes before it.
  ans=P : the answering cell's branches compete: p = exp(a) / sum(exp(a)); mismatch = (1 for the byte that arrived, 0 for the others) - p (nano/loop's arm P).
  ans=LS: mismatch = the same indicator - a (least squares, as loop_small_case/prose.py); its loss is crude (outputs cut off below at 1e-3, made to sum to one).
bytes=N: only the first N bytes of the stream are read (for check_text.py)."""
import json, os, sys, time
import numpy as np
import multi
HERE = os.path.dirname(os.path.abspath(__file__)); TEXT = os.path.join(HERE, "..", "data", "tinyshakespeare.txt")

def load():
    raw = np.frombuffer(open(TEXT, "rb").read(), np.uint8); vals = np.unique(raw); idx = np.zeros(256, np.int32); idx[vals] = np.arange(len(vals))
    x = idx[raw]; cut = len(x) * 9 // 10; return x, cut, len(vals), int(idx[10])        # the whole text, where the never-read tenth starts, byte values, the line feed

def contexts(x, lo, hi, D):
    """For every position t in lo..hi-1: the D bytes before it, newest first (lo >= D)."""
    return np.stack([x[lo - 1 - d: hi - 1 - d] for d in range(D)], 1), x[lo:hi]

def score(s, x, lo, hi, soft, chunk=2000):
    hist, y = contexts(x, lo, hi, s.D); right = 0; nats = 0.0
    for i in range(0, len(y), chunk):
        a = s.out(hist[i:i + chunk]); yy = y[i:i + chunk]; right += int((a.argmax(1) == yy).sum())
        if soft: a = a - a.max(1, keepdims=True); lp = a[np.arange(len(yy)), yy] - np.log(np.exp(a).sum(1))
        else: p = np.maximum(a, 1e-3); lp = np.log(p[np.arange(len(yy)), yy] / p.sum(1))               # crude: cut off below at 1e-3, made to sum to one
        nats -= float(lp.sum())
    return dict(nats=nats / len(y), bits=nats / len(y) / np.log(2), right=100 * right / len(y))

def run(kw):
    t0 = time.process_time(); x, cut, S, pad = load(); D = int(kw.get("D", 10)); H = int(kw.get("H", 1024)); soft = int(kw.get("ans", "P") == "P"); passes = int(kw.get("passes", 8)); nl = kw.get("nline", "all")
    s = multi.Sys(int(kw.get("seed", 1)), S=S, D=D, H=H, nline=None if nl == "all" else int(nl), npair=int(kw.get("npair", 0)), rewire=int(kw.get("rewire", 0)), adj=int(kw.get("adj", 0)), comp=soft, pad=pad,
                  eta1=float(eval(kw.get("eta1", "1"))), eta2=float(eval(kw.get("eta2", "1"))), lam=float(kw.get("lam", 1e-5)), pdraw=kw.get("pdraw", "uniform"), pscale=float(kw.get("pscale", .5)))
    s.R *= float(kw.get("rscale", 1)); s.fast = int(kw.get("fast", 1)); nb = int(kw.get("bytes", cut)); xs = x[:nb - 1]; ys = x[1:nb]
    out = dict(label=kw["label"], args=kw, S=S, stream=nb, never_read=len(x) - cut, counts=s.counts(), rows=[]); lo, hi = (cut, len(x)) if "bytes" not in kw else (cut, cut + 2000)
    for p in range(passes):
        s.nats = 0.0; s.right = 0; s.feed(xs, ys); r = score(s, x, lo, hi, soft)
        if not (np.isfinite(s.W2).all() and np.isfinite(s.W1).all() and np.isfinite(s.PW).all()): r = dict(nats=float("inf"), bits=float("inf"), right=0.0)
        r.update(passes=p + 1, stream_nats=s.nats / len(xs), stream_right=100 * s.right / len(xs), made_lines=s.made, made_pairs=s.madep, cpu=round(time.process_time() - t0, 1)); out["rows"].append(r)
        if not kw.get("nowrite"): os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"t2_{kw['label']}.json"), "w"))
        if not np.isfinite(r["nats"]): break
    if s.P:
        dd = np.zeros((D, D), int); np.add.at(dd, (s.PA // S, s.PB // S), 1); out["pair_delays"] = dd.tolist(); out["pair_abs_mean"] = float(np.abs(s.PW).mean())
    if s.M is not None and s.nline: out["line_delays"] = s.M.reshape(H, D, S).sum((0, 2)).tolist(); out["line_abs_mean"] = float(np.abs(s.W1[s.M]).mean())
    out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"): json.dump(out, open(os.path.join(HERE, "res", f"t2_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], "| " + " ".join(f"{r['passes']}: {r['nats']:.4f}n {r['right']:.2f}%" for r in o["rows"]), "| made", o["rows"][-1]["made_lines"], o["rows"][-1]["made_pairs"], "| strengths a cell", o["counts"]["strengths_a_cell"], o["cpu"], "s", flush=True)
