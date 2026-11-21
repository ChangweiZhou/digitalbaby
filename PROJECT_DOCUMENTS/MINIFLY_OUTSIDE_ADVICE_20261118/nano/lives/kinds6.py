#!/usr/bin/env python3
"""nano/lives, PLAN_6.md: six streams of bytes that are not text, and a control. The streams are built by the recipes of the plan; nothing in a system is changed.
    python3 kinds6.py build                     the streams' manifests (manifests6/<name>.json: recipe, length, sha256)
    python3 kinds6.py refs                      xz, bzip2, gzip on the same bytes and the same tenth (res/kinds6_ref__<name>.json)
    python3 kinds6.py run sys=ZA|ZAM32x512      each stream read alone by the system (res/mem__K6_<sys>_single_<name>.json)
    python3 kinds6.py report"""
import hashlib, itertools, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SIZE = 1500000; NAMES = ("elf_c", "elf_go", "fonts", "pyc", "magic", "pixels", "random")

def joined(top, suffix, size):
    paths = sorted(os.path.join(top, f) for f in os.listdir(top) if f.endswith(suffix) and os.path.isfile(os.path.join(top, f)) and not os.path.islink(os.path.join(top, f))); parts = []; n = 0
    for p in paths:
        b = open(p, "rb").read(); parts.append(b); n += len(b)
        if n >= size: break
    return b"".join(parts)[:size]

def raw_of(name):
    if name == "elf_c": return open("/usr/lib/x86_64-linux-gnu/libMagickCore-6.Q16.so.7.0.0", "rb").read()[:SIZE]
    if name == "elf_go":
        with open("/usr/bin/docker", "rb") as f: f.seek(4000000); return f.read(SIZE)
    if name == "fonts": return joined("/usr/share/fonts/truetype/crosextra", ".ttf", SIZE)
    if name == "pyc": return joined("/usr/lib/python3.12/__pycache__", ".pyc", SIZE)
    if name == "magic": return open("/usr/lib/file/magic.mgc", "rb").read()[:SIZE]
    if name == "pixels":
        from PIL import Image
        return Image.open("/opt/google-cloud-sdk/platform/gsutil/third_party/requests/ext/ss.png").convert("RGB").tobytes()[:SIZE]
    if name == "random": return np.random.default_rng(20261121).bytes(SIZE)
    raise KeyError(name)

def load(name):
    b = raw_of(name); assert len(b) == SIZE, (name, len(b)); m = os.path.join(HERE, "manifests6", name + ".json"); h = hashlib.sha256(b).hexdigest()
    if os.path.exists(m): assert json.load(open(m))["sha256"] == h, "the stream rebuilt is not the one of the manifest"
    else: os.makedirs(os.path.dirname(m), exist_ok=True); json.dump(dict(name=name, length=len(b), cut=int(0.9 * len(b)), sha256=h, distinct_byte_values=int(len(set(b)))), open(m, "w"), indent=1)
    return b, int(0.9 * len(b))

def refs(name):
    b, cut = load(name); n = len(b) - cut; run = lambda cmd, data: len(subprocess.run(cmd, input=data, stdout=subprocess.PIPE, check=True).stdout); nats = lambda cmd: (run(cmd, b) - run(cmd, b[:cut])) * 8 * np.log(2) / n
    out = dict(stream=name, xz=nats(["xz", "-9e", "-c"]), bzip2=nats(["bzip2", "-9", "-c"]), gzip=nats(["gzip", "-9", "-c"])); grid = {}
    for lc, lp, pb in itertools.product((0, 3, 4), (0, 1, 2), (0, 1, 2)):
        if lc + lp > 4: continue
        grid[f"lc{lc}lp{lp}pb{pb}"] = nats(["xz", "-c", f"--lzma2=preset=9e,lc={lc},lp={lp},pb={pb}"])
    out["xz_literal_settings"] = grid; out["xz_best_literal_setting"] = min(grid, key=grid.get); out["xz_best"] = min(min(grid.values()), out["xz"])
    if name.startswith("elf"): out["xz_x86_filter"] = nats(["xz", "-c", "--x86", "--lzma2=preset=9e"])
    json.dump(out, open(os.path.join(HERE, "res", f"kinds6_ref__{name}.json"), "w")); print(name, {k: round(v, 4) for k, v in out.items() if isinstance(v, float)}, out["xz_best_literal_setting"], flush=True)

def run(sysname):
    import lives as LV, mem as MM
    for name in NAMES:
        tag = f"K6_{sysname}_single_{name}"
        if os.path.exists(os.path.join(HERE, "res", f"mem__{tag}.json")): continue
        b, cut = load(name); LV._RAW[name] = (np.frombuffer(b, np.uint8).astype(np.int32), cut); out = MM.run_life(dict(order=name, sys=sysname, tag=tag)); t = out["stages"][0]["tests"][name]
        print(tag, "| nothing changing", round(t["nothing_changing"]["cell2"], 4), "| going on", round(t["going_on"]["cell2"], 4), "| cells", out["stages"][0]["sizes"]["cells"], flush=True)

def report():
    L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json"))); have = lambda n: os.path.exists(os.path.join(HERE, "res", n + ".json")); rows = []
    print("| stream | the small system (ZA) | with bounded kinds | xz -9e | xz, best of its literal settings | bzip2 -9 | gzip -9 | xz with its filter for machine code | nothing changing: ZA, with bounded kinds |\n|---|" + "---|" * 8)
    for name in NAMES:
        if not (have(f"kinds6_ref__{name}") and have(f"mem__K6_ZA_single_{name}") and have(f"mem__K6_ZAM32x512_single_{name}")): continue
        r = L(f"kinds6_ref__{name}"); a = L(f"mem__K6_ZA_single_{name}")["stages"][0]["tests"][name]; b = L(f"mem__K6_ZAM32x512_single_{name}")["stages"][0]["tests"][name]; rows.append((name, a, b, r))
        print(f"| {name} | {a['going_on']['cell2']:.4f} | {b['going_on']['cell2']:.4f} | {r['xz']:.4f} | {r['xz_best']:.4f} | {r['bzip2']:.4f} | {r['gzip']:.4f} | " + (f"{r['xz_x86_filter']:.4f}" if "xz_x86_filter" in r else "") + f" | {a['nothing_changing']['cell2']:.4f}, {b['nothing_changing']['cell2']:.4f} |")
    six = [q for q in rows if q[0] != "random"]
    if len(six) == 6:
        print(f"\nOf the six kinds, going on: ZA below xz -9e on {sum(a['going_on']['cell2'] < r['xz'] for _, a, b, r in six)}; with bounded kinds below xz -9e on {sum(b['going_on']['cell2'] < r['xz'] for _, a, b, r in six)}, below the best literal setting on {sum(b['going_on']['cell2'] < r['xz_best'] for _, a, b, r in six)}, below bzip2 on {sum(b['going_on']['cell2'] < r['bzip2'] for _, a, b, r in six)}; "
              f"bounded kinds lower by 0.001 or more on {sum(a['going_on']['cell2'] - b['going_on']['cell2'] >= 0.001 for _, a, b, r in six)}, largest harm {max(b['going_on']['cell2'] - a['going_on']['cell2'] for _, a, b, r in six):+.4f}; nothing changing: ZA above xz -9e on {sum(a['nothing_changing']['cell2'] > r['xz'] for _, a, b, r in six)}")

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); cmd = sys.argv[1]; kw = dict(a.split("=", 1) for a in sys.argv[2:])
    if cmd == "build":
        for n in NAMES: b, cut = load(n); print(n, len(b), "bytes, cut", cut, "distinct byte values", len(set(b)), hashlib.sha256(b).hexdigest()[:16])
    elif cmd == "refs":
        for n in NAMES: refs(n)
    elif cmd == "run": run(kw["sys"])
    elif cmd == "report": report()
