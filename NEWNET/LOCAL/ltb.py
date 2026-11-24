"""Battery for local-tissue candidates (24 Nov 2026). Two goals only.
  life: streams with overlapping bytes read one after another, nothing reset: shak -> emma -> code -> moby -> shak (continued).
        After each phase, learning off, nats/byte on each stream's fixed held-out stretch (from its middle, never read).
        cl = mean nats over the four streams at the end of the life (lower is better); also each stream's nats right after its own
        phase and at the end (forgetting = end - after own phase).
  slope: moby; size factor s = 1, 2, 4 multiplies cells per column M and the bytes read (T0 * s); nats on the fixed held-out.
        slope = least-squares nats per doubling (more negative is better).
Usage: python3 ltb.py NAME 'opt=v ...' [T=16000 NT=4000 T0=16000 parts=life,slope]  -> one JSON line on stdout"""
import sys, json, time, os
import numpy as np
from lt import Net, DEF

D = "/Users/pencilbard/Downloads/digitalbaby_newnet/"
SRC = dict(shak=D + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt",
           emma=D + "minifly_v155_keys_dev_v1_20261026/data/austen-emma.txt",
           moby=D + "minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt",
           code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_stream.bin"))
S = {k: np.minimum(np.frombuffer(open(v, "rb").read(), np.uint8), 127).astype(np.uint8) for k, v in SRC.items()}


def held(k, nt): m = len(S[k]) // 2; return S[k][m:m + nt]


def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v


def main():
    name, opts = sys.argv[1], dict(a.split("=") for a in sys.argv[2].split()) if sys.argv[2].strip() else {}
    opts = {k: num(v) for k, v in opts.items()}
    ex = {k: num(v) for k, v in (a.split("=") for a in sys.argv[3:])}
    T, NT, T0 = ex.get("T", 16000), ex.get("NT", 4000), ex.get("T0", 16000)
    parts = str(ex.get("parts", "life,slope")).split(",")
    out = dict(name=name, opts=opts, T=T, NT=NT, T0=T0); t0 = time.time()
    if "life" in parts:
        net = Net(**opts); order = ["shak", "emma", "code", "moby", "shak"]; pos = {k: 0 for k in S}; mat = []
        tests = ["shak", "emma", "code", "moby"]
        for ph, k in enumerate(order):
            net.run(S[k][pos[k]:pos[k] + T]); pos[k] += T
            mat.append([round(net.test(held(j, NT))[0], 4) for j in tests])
        own = {k: mat[order.index(k)][tests.index(k)] for k in tests}
        end = dict(zip(tests, mat[-1]))
        out.update(cl=round(float(np.mean(mat[-1])), 4), mat=mat,
                   forget={k: round(end[k] - mat[[i for i, o in enumerate(order) if o == k][0]][tests.index(k)], 4) for k in ["emma", "code"]},
                   shak_first=mat[0][0], shak_before_return=mat[3][0], audit=net.audit(), life_s=round(time.time() - t0, 1))
    if "slope" in parts:
        t1 = time.time(); xs, ys = [], []
        for s in (1, 2, 4):
            o2 = dict(opts); o2["M"] = int(opts.get("M", DEF["M"])) * s
            net = Net(**o2); net.run(S["moby"][:T0 * s]); ys.append(net.test(held("moby", NT))[0]); xs.append(np.log2(s))
        out.update(slope=round(float(np.polyfit(xs, ys, 1)[0]), 4), ladder=[round(v, 4) for v in ys], slope_s=round(time.time() - t1, 1))
    out["secs"] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
