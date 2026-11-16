"""E35-A tables from the receipts (no readings).   python3 exp45/e35a/summarize.py stage0|stream"""
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
OPS = ("add89", "sub89", "max89", "min89")
FAM = {"addsub": ("add89", "sub89"), "maxmin": ("max89", "min89")}
ROWS = (16384, 20480, 24576)


def stage0():
    out = ["| fraction | seed | arm | final held mean | add | sub | max | min | curve of held mean (every 4,096) |", "|---|---|---|---|---|---|---|---|---|"]
    for f in (25, 10):
        for arm in ("ring", "table"):
            for s in (73129, 93683, 116089):
                p = D / "stage0" / f"e35a_parsed_{arm}_f{f}_{s}.json"
                if not p.exists():
                    out.append(f"| {f/100:.2f} | {s} | {arm} | NOT RUN | | | | | |"); continue
                r = json.loads(p.read_text()); c = r["curve"][-1]
                out.append(f"| {f/100:.2f} | {s} | {arm} | {c['held_mean_pct']:.1f} | " + " | ".join(f"{c['held'][o]['pct']:.1f}" for o in ("add", "sub", "max", "min"))
                           + " | " + ", ".join(f"{x['held_mean_pct']:.1f}" for x in r["curve"]) + " |")
    return "\n".join(out)


def pct(v):
    return 100 * v["exact"] / v["n"]


def stream_rows(r):
    rows = {c["step"]: c for c in r["curve"]}
    return [rows[s] for s in ROWS if s in rows]


def metric(rows, key, k, ops):
    vals = [sum(pct(c[key][o][str(k)] if str(k) in c[key][o] else c[key][o][k]) for o in ops) / len(ops) for c in rows if key in c]
    return (sum(vals) / len(vals) if vals else None), (vals[-1] if vals else None)


def load(arm, f, seed=73129):
    p = D / f"e35a_dev_{arm}_f{f}_{seed}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("complete") else None


def fmt(x):
    return "—" if x is None else f"{x:.1f}"


def stream():
    out = []
    cols = [("held12", "sealed", 12), ("held40", "sealed", 40), ("taught12", "taught", 12), ("off held12", "module_off", 12), ("off held40", "module_off", 40)]
    for label, ops in [("mean of four", OPS), ("add+sub", FAM["addsub"]), ("max+min", FAM["maxmin"])] + [(o, (o,)) for o in OPS]:
        out.append(f"\n**{label}** (last-three mean / final row)\n")
        out.append("| arm | fraction | " + " | ".join(c[0] for c in cols) + " |"); out.append("|---|---|" + "---|" * len(cols))
        for f in (10, 25):
            for arm in ("G", "M", "X", "F"):
                r = load(arm, f)
                if r is None:
                    continue
                rows = stream_rows(r)
                cells = []
                for _, key, k in cols:
                    a, b = metric(rows, key, k, ops)
                    cells.append(f"{fmt(a)} / {fmt(b)}")
                out.append(f"| {arm} | {f/100:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    print(stage0() if sys.argv[1] == "stage0" else stream())
