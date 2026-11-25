"""Battery v2, continual-learning part (25 Nov 2026): one life on the English + addition data (arith/make_life.py, design 2) and
the same life on its control file (answers permuted within each cell and block). After every block, learning off:
  English: nats/byte on the middle 4,000 bytes of Shakespeare and of Emma (never read);
  addition: for each test set (taught E1, e2 E2, heldpair / heldop / heldsum E3), up to NITEM items, each scored on its own line
           ("\n" + text) from the life's current state: mean nats over the answer bytes (word = the sum in words, the addition;
           word_first = its first word; digit = the sum in digits, which follows the word sum, i.e. translation).
E3 evidence = control minus real on held-out items (both lives have the same bytes everywhere except the answers).
Learners have new(seed) -> object with run(x), test(x) -> nats/byte, trace(x) -> nats of each byte.
Usage: python3 life.py LEARNER DATA_DIR OUT.json [seed]     LEARNER: lt7 | oracle (scoring check)"""
import sys, os, json, re, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = "/Users/pencilbard/Downloads/digitalbaby_newnet/"
ENG = dict(shak=D + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt",
           emma=D + "minifly_v155_keys_dev_v1_20261026/data/austen-emma.txt")
NITEM, NENG = 48, 4000
SETS = ["taught", "e2", "heldpair", "heldop", "heldsum"]


class LT7:
    def __init__(self, seed):
        sys.path.insert(0, HERE + "/../lt7"); from lt7 import Net7, test_trace
        self.n = Net7(dict(ntis=2048, ninh=512, nrly=512, seed=seed)); self._tt = test_trace
    def run(self, x): self.n.run(x)
    def test(self, x): return self.n.test(x)[0]
    def trace(self, x): return self._tt(self.n, x)


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
def words(n):
    if n == 1000: return "one thousand"
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))
W2N = {words(n): n for n in range(1001)}


class Oracle:
    """scoring check: predicts a byte with p = 0.99 when its own parser of the text before it knows the byte (the sum after
    'is' / 'equal to' / 'make', and after '='), else uniform over 256. It never sees the battery's spans."""
    def __init__(self, seed): pass
    def run(self, x): pass
    def test(self, x): return float(np.mean(self.trace(x)[1:]))
    def trace(self, x):
        s = bytes(x).decode("latin-1"); out = np.full(len(s), np.log(256.0), np.float32); out[0] = -1
        for t in range(1, len(s)):
            pre = s[:t]; line = pre.split("\n")[-1]; tgt = None
            m = re.match(r"^(.+?) (?:plus|and) (.+?)(?:, we say,|, as everyone knows,|, as every child in school knows,)? (?:is|equal to|make) (.*)$", line)
            if m and " - in equation" not in m.group(3) and m.group(1) in W2N and m.group(2) in W2N:
                ans = words(W2N[m.group(1)] + W2N[m.group(2)]); k = len(m.group(3))
                if ans.startswith(m.group(3)) and k < len(ans): tgt = ans[k]
            m2 = re.match(r"^.* - in equation (\d+)\+(\d+)=(\d*)$", line)
            if m2:
                ans = str(int(m2.group(1)) + int(m2.group(2))); k = len(m2.group(3))
                if ans.startswith(m2.group(3)) and k < len(ans): tgt = ans[k]
            if tgt is not None and s[t] == tgt: out[t] = -np.log(0.99)
        return out


def evaluate(L, items, eng):
    res = {k: round(L.test(v), 4) for k, v in eng.items()}
    for name, rows in items.items():
        sc = {"word": [], "word_first": [], "digit": []}
        for r in rows:
            x = np.frombuffer(("\n" + r["text"]).encode("latin-1"), np.uint8); per = L.trace(x)
            for k in sc: a, b = r["spans"][k]; sc[k].append(float(np.mean(per[a + 1:b + 1])))
        res[name] = {k: round(float(np.mean(v)), 4) for k, v in sc.items()}
    return res


def life(L, data_dir, fname, items, eng):
    lines = open(os.path.join(data_dir, fname), "rb").read().split(b"\n"); E = json.load(open(os.path.join(data_dir, "exposure.json")))
    pos, out = 0, []
    for b in E["blocks"]:
        chunk = b"\n".join(lines[pos:pos + b["lines"]]) + b"\n"; pos += b["lines"]
        L.run(np.frombuffer(chunk, np.uint8)); out.append(dict(block=b["block"], kind=b["kind"], arg=b["arg"], **evaluate(L, items, eng)))
        print(json.dumps(out[-1]), flush=True)
    return out


if __name__ == "__main__":
    lr, data_dir, outp = sys.argv[1], sys.argv[2], sys.argv[3]; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    mk = {"lt7": LT7, "oracle": Oracle}[lr]; rng = random.Random(1234)
    items = {}
    for nme in SETS:
        rows = [json.loads(l) for l in open(os.path.join(data_dir, f"test_{nme}.jsonl"))]; rng.shuffle(rows); items[nme] = rows[:NITEM]
    eng = {}
    for k, p in ENG.items():
        x = np.frombuffer(open(p, "rb").read(), np.uint8); m = len(x) // 2; eng[k] = x[m:m + NENG]
    R = dict(learner=lr, data=data_dir, seed=seed, real=life(mk(seed), data_dir, "life.txt", items, eng), control=life(mk(seed), data_dir, "control.txt", items, eng))
    json.dump(R, open(outp, "w"), indent=1)
    last_r, last_c = R["real"][-1], R["control"][-1]
    print("END  set: real word / control word (E3 evidence = control - real)")
    for nme in SETS: print(f"  {nme:9s} {last_r[nme]['word']:.3f} / {last_c[nme]['word']:.3f}   first word {last_r[nme]['word_first']:.3f} / {last_c[nme]['word_first']:.3f}   digit {last_r[nme]['digit']:.3f} / {last_c[nme]['digit']:.3f}")
    print(f"  English shak {last_r['shak']:.3f} / {last_c['shak']:.3f}   emma {last_r['emma']:.3f} / {last_c['emma']:.3f}")
