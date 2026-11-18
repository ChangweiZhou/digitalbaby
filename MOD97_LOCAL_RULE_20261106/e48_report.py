"""E48 report tables from the run receipts (exp45/e48/e48_<rule>_<seed>_r<rank>.json) and E40's lived receipts. Writes exp45/e48/E48_TABLE.json
and prints Markdown tables. Readings exactly as fixed in the brief (revision 2); nothing beyond them.
    python3 e48_report.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D, D40 = HERE / "exp45" / "e48", HERE / "exp45" / "e40"
SEEDS, RANKS, RULES = (73129, 93683, 116089), (4, 16, 64), ("S", "B")
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024)
E40ROWS = (0, 64, 128, 256, 512, 1024)
EARLIER6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73")
NET = 1124113
pct = lambda d: 100.0 * d["exact"] / d["n"]


def load(rule, seed, rank):
    return json.loads((D / f"e48_{rule}_{seed}_r{rank}.json").read_text())


def e40(rule, seed):
    return json.loads((D40 / f"e40_lived_{rule}_{seed}.json").read_text())


def rows(rec):
    return {r["updates"]: r for r in rec["rows"]}


def red(rec):
    return {u: r["scores"]["summary"]["reduction_score"] for u, r in rows(rec).items()}


def Rmean(d):
    return (d[128] + d[256] + d[512]) / 3


def verdict(diffs):
    if sum(d >= -10 for d in diffs) >= 2:
        return "no price"
    if sum(d <= -30 for d in diffs) >= 2:
        return "a price"
    return "some price"


def main():
    out = dict(curves={}, R={}, readings={}, tabled={}, earlier={}, forced_off={}, cosines={}, files={})
    md = []
    # curves
    md.append("## Curves: reduction score (E48 | E40) by row\n")
    md.append("| run | " + " | ".join(str(u) for u in ROWS) + " |")
    md.append("|---|" + "---|" * len(ROWS))
    for rule in RULES:
        for seed in SEEDS:
            r40 = red(e40(rule, seed))
            md.append(f"| E40 lived {rule} {seed} | " + " | ".join(f"{r40[u]:.1f}" if u in r40 else "–" for u in ROWS) + " |")
            for rank in RANKS:
                r48 = red(load(rule, seed, rank))
                out["curves"][f"{rule}_{seed}_r{rank}"] = r48
                md.append(f"| E48 {rule} {seed} r{rank} | " + " | ".join(f"{r48[u]:.1f}" for u in ROWS) + " |")
    # other summaries
    md.append("\n## Curves: held-out k12 add/sub mean, taught k12 add/sub mean, no-reduction score (E48), rows 0..1024\n")
    md.append("| run | measure | " + " | ".join(str(u) for u in ROWS) + " |")
    md.append("|---|---|" + "---|" * len(ROWS))
    for rule in RULES:
        for seed in SEEDS:
            for rank in RANKS:
                rr = rows(load(rule, seed, rank))
                for k, lab in (("held_add_sub_mean", "held"), ("taught_add_sub_mean", "taught"), ("no_reduction_score", "no-red")):
                    md.append(f"| {rule} {seed} r{rank} | {lab} | " + " | ".join(f"{rr[u]['scores']['summary'][k]:.1f}" for u in ROWS) + " |")
            rr = rows(e40(rule, seed))
            for k, lab in (("held_add_sub_mean", "held"), ("taught_add_sub_mean", "taught"), ("no_reduction_score", "no-red")):
                md.append(f"| E40 {rule} {seed} | {lab} | " + " | ".join(f"{rr[u]['scores']['summary'][k]:.1f}" if u in rr else "–" for u in ROWS) + " |")
    # reading
    md.append("\n## Reading: R = mean reduction score at rows 128, 256, 512; E48 − E40 per seed\n")
    md.append("| rule | rank | " + " | ".join(f"{s}: E48 / E40 / diff" for s in SEEDS) + " | reading |")
    md.append("|---|---|" + "---|" * len(SEEDS) + "---|")
    for rule in RULES:
        for rank in RANKS:
            diffs, cells = [], []
            for seed in SEEDS:
                a, b = Rmean(red(load(rule, seed, rank))), Rmean(red(e40(rule, seed)))
                diffs.append(a - b); cells.append(f"{a:.2f} / {b:.2f} / {a - b:+.2f}")
                out["R"][f"{rule}_{seed}_r{rank}"] = dict(e48=a, e40=b, diff=a - b)
            v = verdict(diffs)
            out["readings"][f"{rule}_r{rank}"] = dict(diffs=diffs, reading=v)
            md.append(f"| {rule} | {rank} | " + " | ".join(cells) + f" | **{v}** |")
    # tabled, not read: single rows 64, 256, 1024; taught; numbers
    md.append("\n## Tabled, not read: single rows of the reduction score (E48 / E40), taught panel mean at 256 and 1,024, numbers changed\n")
    md.append("| rule | seed | rank | row 64 | row 256 | row 1024 | taught 256 | taught 1024 | numbers changed | share of network |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for rule in RULES:
        for seed in SEEDS:
            r40 = red(e40(rule, seed)); t40 = rows(e40(rule, seed))
            for rank in RANKS:
                rec = load(rule, seed, rank); r48 = red(rec); rr = rows(rec)
                tg = lambda R_, u: R_[u]["scores"]["summary"]["taught_add_sub_mean"]
                md.append(f"| {rule} | {seed} | {rank} | {r48[64]:.1f} / {r40[64]:.1f} | {r48[256]:.1f} / {r40[256]:.1f} | {r48[1024]:.1f} / {r40[1024]:.1f} | "
                          f"{tg(rr, 256):.1f} / {tg(t40, 256):.1f} | {tg(rr, 1024):.1f} / {tg(t40, 1024):.1f} | {rec['numbers_that_may_change']:,} | {100 * rec['numbers_that_may_change'] / NET:.2f}% |")
    # earlier families: E40 paid, E48 by construction row 0
    md.append("\n## The other side: earlier-family exact counts, E40 rows 0 / 128 / 256 / 512 (E48 at every row = row 0, check 3)\n")
    md.append("| rule | seed | " + " | ".join(EARLIER6) + " |")
    md.append("|---|---|" + "---|" * 6)
    for rule in RULES:
        for seed in SEEDS:
            rr = rows(e40(rule, seed))
            cells = [" / ".join(str(rr[u]["scores"]["earlier_k12"][n]["exact"]) for u in (0, 128, 256, 512)) + f" (of {rr[0]['scores']['earlier_k12'][n]['n']})" for n in EARLIER6]
            out["earlier"][f"{rule}_{seed}"] = {n: {u: rr[u]["scores"]["earlier_k12"][n]["exact"] for u in (0, 128, 256, 512)} for n in EARLIER6}
            md.append(f"| {rule} | {seed} | " + " | ".join(cells) + " |")
    # gate forced off
    md.append("\n## Column: new family scored with the gate forced off (the code rows alone), rows 256 and 1,024: reduction / held k12 / taught k12 (gate on in brackets)\n")
    md.append("| rule | seed | rank | row 256 off (on) | row 1024 off (on) |")
    md.append("|---|---|---|---|---|")
    for rule in RULES:
        for seed in SEEDS:
            for rank in RANKS:
                rr = rows(load(rule, seed, rank)); cells = []
                for u in (256, 1024):
                    f, s = rr[u]["scores_gate_forced_off"]["summary"], rr[u]["scores"]["summary"]
                    cells.append(f"{f['reduction_score']:.1f} / {f['held_add_sub_mean']:.1f} / {f['taught_add_sub_mean']:.1f} "
                                 f"({s['reduction_score']:.1f} / {s['held_add_sub_mean']:.1f} / {s['taught_add_sub_mean']:.1f})")
                    out["forced_off"][f"{rule}_{seed}_r{rank}_{u}"] = dict(off=f, on=s)
                md.append(f"| {rule} | {seed} | {rank} | " + " | ".join(cells) + " |")
    # cosines
    md.append("\n## Column: under the sweep, cosine of each factor with sign(F)·mean|F|, rows 64 / 256 / 1,024 (down_i, up_i, down_h, up_h)\n")
    md.append("| seed | rank | row 64 | row 256 | row 1024 |")
    md.append("|---|---|---|---|---|")
    for seed in SEEDS:
        for rank in RANKS:
            rr = rows(load("S", seed, rank))
            out["cosines"][f"S_{seed}_r{rank}"] = {u: rr[u]["sweep_cosines"] for u in ROWS}
            f = lambda c: " ".join("–" if c[k] is None else f"{c[k]:.3f}" for k in ("down_i", "up_i", "down_h", "up_h"))
            md.append(f"| {seed} | {rank} | " + " | ".join(f(rr[u]["sweep_cosines"]) for u in (64, 256, 1024)) + " |")
    # cpu
    cpu = {f"{rule}_{seed}_r{rank}": load(rule, seed, rank)["cpu_total"] for rule in RULES for seed in SEEDS for rank in RANKS}
    out["cpu_seconds"] = cpu; out["cpu_total_hours"] = sum(cpu.values()) / 3600
    md.append(f"\nCPU of the 18 runs: {sum(cpu.values()) / 3600:.2f} h (one thread each).")
    (D / "E48_TABLE.json").write_text(json.dumps(out, indent=1))
    (D / "E48_REPORT_TABLES.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
