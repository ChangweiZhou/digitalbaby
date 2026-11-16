"""E43 addendum 1 reader: the reminder's curves and reading, the plain arm's curves and reading, P43.10-P43.14. Receipts only.

    python3 e43_addendum_read.py       # writes exp45/e43/addendum1/E43_ADDENDUM1_READ.json and E43_ADDENDUM1_TABLES.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "exp45" / "e43" / "addendum1"; E36 = HERE / "exp45" / "e36"
RULES, SEEDS, LR = ("S", "B"), (73129, 93683, 116089), 0.03


def summ(rows):
    return {r["updates"]: r["scores"]["summary"] for r in rows}


def main():
    rem, ref = {}, {}
    for r in RULES:
        for s in SEEDS:
            x = json.loads((D / f"e43_reminder_{r}_{s}.json").read_text()); assert x["complete"]
            rem[r, s] = summ(x["rows"])
            ref[r, s] = summ(json.loads((E36 / f"e36_reference_{r}_{s}.json").read_text())["rows"])
    rreading = {}
    for r in RULES:
        per = {}
        for s in SEEDS:
            a, f = rem[r, s], ref[r, s]
            per[s] = dict(held_128=a[128]["held_add_sub_mean"], wrap_1024=a[1024]["wrap_score"], ref_wrap_1024=f[1024]["wrap_score"],
                          wrap_minus_ref=a[1024]["wrap_score"] - f[1024]["wrap_score"])
            per[s]["back"] = per[s]["held_128"] >= 85
            per[s]["partly"] = not per[s]["back"] and per[s]["wrap_minus_ref"] >= 30
            per[s]["not_back"] = abs(per[s]["wrap_minus_ref"]) < 15
        n = lambda k: sum(per[s][k] for s in SEEDS)
        reading = "Back within about a hundred updates" if n("back") >= 2 else "Partly back" if n("partly") >= 2 else \
            "Not back" if n("not_back") >= 2 else "Mixed"
        rreading[r] = dict(per_seed=per, reading=reading)
    plain, pread = {}, {}
    pc = lambda d: 100 * d["exact"] / d["n"]
    for r in RULES:
        for s in SEEDS:
            x = json.loads((D / f"e43_unrelated-plain_lr{LR:g}_{r}_{s}.json").read_text()); assert x["complete"]
            plain[r, s] = {v["updates"]: v for v in x["rows"]}
            ext = D / f"e43_unrelated-plain_lr{LR:g}_{r}_{s}_to16384.json"
            if ext.exists():
                y = json.loads(ext.read_text()); assert y["complete"]
                plain[r, s].update({v["updates"]: v for v in y["rows"] if v["updates"] > 4096})
        tried, row = {}, None
        for u in (4096, 8192, 16384):
            if all(u in plain[r, s] for s in SEEDS):
                t = [pc(plain[r, s][u]["score3_new_task"]["taught"]) for s in SEEDS]
                tried[u] = dict(taught=t, held=[pc(plain[r, s][u]["score3_new_task"]["held"]) for s in SEEDS], holds=sum(v >= 80 for v in t) >= 2)
                if tried[u]["holds"] and row is None:
                    row = u
        per = {}
        for s in SEEDS:
            rows = plain[r, s]; base = rows[0]["score1_first_family_all_pairs_k0"]["reduction_score"]
            share = lambda u, key="reduction_score", rows=rows: 100 * rows[u]["score1_first_family_all_pairs_k0"][key] / rows[0]["score1_first_family_all_pairs_k0"][key]
            per[s] = dict(share={u: share(u) for u in (64, 256, 1024, 4096, 8192, 16384) if u in rows},
                          finite=all(v["finite_losses"] for v in rows.values()))
        if row is None:
            reading = "condition not met: nothing drawn"
        else:
            kept = sum(per[s]["share"][row] >= 80 for s in SEEDS); lost = sum(per[s]["share"][row] <= 20 for s in SEEDS)
            reading = "Kept" if kept >= 2 else "Lost" if lost >= 2 else "Partly kept"
        pread[r] = dict(per_seed=per, condition=tried, reading_row=row, reading=reading)
    rates = {}
    for lr in (0.01, 0.03, 0.1):
        x = json.loads((D / f"e43_unrelated-plain_lr{lr:g}_B_73129.json").read_text())
        last = x["rows"][-1]; pc = lambda d: 100 * d["exact"] / d["n"]
        rates[lr] = dict(taught_4096=pc(last["score3_new_task"]["taught"]), held_4096=pc(last["score3_new_task"]["held"]),
                         finite=all(v["finite_losses"] for v in x["rows"]), last_loss=last["loss_since_last_row"])
    chosen = min((lr for lr, v in rates.items() if v["taught_4096"] >= 80 and v["finite"]), default=None)
    two = lambda xs: sum(bool(v) for v in xs) >= 2
    P = {"P43.10": (two(rreading["S"]["per_seed"][s]["back"] for s in SEEDS), "reminder back at row 128, two of three, sweep", "50%"),
         "P43.11": (all(two(rreading[r]["per_seed"][s]["wrap_minus_ref"] >= 30 for s in SEEDS) for r in RULES),
                    "reminder wrap at 1,024 at least 30 above the fresh reference's, two of three, each rule", "85%"),
         "P43.12": (chosen is not None, "some rate of 0.01, 0.03, 0.1 meets the condition on seed 73129 under backpropagation", "70%"),
         "P43.13": (pread["B"]["reading"], "unrelated-plain, backpropagation: kept 15 / partly 30 / lost 55", ""),
         "P43.14": (two(pread["B"]["per_seed"][s]["share"][64] > 50 for s in SEEDS), "unrelated-plain share at row 64 above 50%, two of three, backpropagation", "55%")}
    out = dict(rates=rates, rate_chosen=chosen, reminder=rreading, plain=pread, predictions={k: dict(outcome=v[0], what=v[1], chance=v[2]) for k, v in P.items()},
               reminder_curves={f"{r}_{s}": rem[r, s] for r, s in rem}, reference_curves={f"{r}_{s}": ref[r, s] for r, s in ref})
    (D / "E43_ADDENDUM1_READ.json").write_text(json.dumps(out, indent=1, default=str))
    f = lambda x: f"{x:.1f}"
    L = ["# E43 addendum 1 tables, from the receipts", "", "## Reminder (held-out add/sub mean at k = 12 / wrap score / no-wrap score); E36's fresh reference beside it", "",
         "| rule, seed | run | " + " | ".join(str(u) for u in (0, 16, 32, 64, 128, 256, 1024)) + " |", "|---|---|" + "---|" * 7]
    for r in RULES:
        for s in SEEDS:
            for nm, c in (("reminder", rem[r, s]), ("E36 reference", ref[r, s])):
                L.append(f"| {r} {s} | {nm} | " + " | ".join(f"{f(c[u]['held_add_sub_mean'])}/{f(c[u]['wrap_score'])}/{f(c[u]['nowrap_score'])}" for u in (0, 16, 32, 64, 128, 256, 1024)) + " |")
    L += ["", f"## unrelated-plain, lr {LR} (reduction / other / max / min · taught / held-out new task · state change)", "",
          "| rule, seed | " + " | ".join(str(u) for u in (0, 64, 256, 1024, 2048, 4096, 8192, 16384)) + " |", "|---|" + "---|" * 8]
    for r in RULES:
        for s in SEEDS:
            cells = []
            for u in (0, 64, 256, 1024, 2048, 4096, 8192, 16384):
                if u not in plain[r, s]:
                    cells.append("–"); continue
                v = plain[r, s][u]; s1 = v["score1_first_family_all_pairs_k0"]
                cells.append(f"{f(s1['reduction_score'])}/{f((s1['add89']['other']['pct'] + s1['sub89']['other']['pct']) / 2)}/{f(s1['max89']['all']['pct'])}/"
                             f"{f(s1['min89']['all']['pct'])} · {f(pc(v['score3_new_task']['taught']))}/{f(pc(v['score3_new_task']['held']))} · {v['score6_state_change']['ratio']:.2f}")
            L.append(f"| {r} {s} | " + " | ".join(cells) + " |")
    (D / "E43_ADDENDUM1_TABLES.md").write_text("\n".join(L) + "\n")
    print(json.dumps(dict(rates=rates, chosen=chosen, reminder={r: v["reading"] for r, v in rreading.items()}, plain={r: v["reading"] for r, v in pread.items()},
                          predictions=out["predictions"]), indent=1, default=str))


if __name__ == "__main__":
    main()
