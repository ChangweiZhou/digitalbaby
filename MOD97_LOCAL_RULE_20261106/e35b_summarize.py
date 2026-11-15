"""E35-B stage 1a: tables from the birth receipts only (no readings). Writes exp45/e35b/E35B_STAGE1A_TABLE.{json,md}.

    python3 e35b_summarize.py [G_receipt_dir]      # optional: a directory holding the E35-A worker's arm-G receipts, if any

Every figure is % exact over the 300-item panel. held12/held40/taught12 are the mean of rows 16,384, 20,480, 24,576 ("last3") and the final
row; families: add/sub = mean of add89 and sub89; max/min = mean of max89 and min89.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "exp45" / "e35b"
ROWS = (16384, 20480, 24576)
OPS = ("add89", "sub89", "max89", "min89")
FAM = {"add/sub": ("add89", "sub89"), "max/min": ("max89", "min89")}


def pct(v):
    return 100 * v["exact"] / v["n"]


def metrics(rec):
    rows = {r["step"]: r for r in rec["curve"] if r["step"] in ROWS}
    if set(rows) != set(ROWS):
        return None
    out = {}
    def get(r, key, op):
        if key == "held12": return pct(r["sealed"][op]["12"] if "12" in r["sealed"][op] else r["sealed"][op][12])
        if key == "held40": return pct(r["sealed"][op]["40"] if "40" in r["sealed"][op] else r["sealed"][op][40])
        if key == "taught12": return pct(r["taught_k12"][op])
        if key == "off12" and "module_off" in r: return pct(r["module_off"][op]["12"])
        if key == "off40" and "module_off" in r: return pct(r["module_off"][op]["40"])
        return None
    for key in ("held12", "held40", "taught12", "off12", "off40"):
        per = {}
        for op in OPS:
            vals = [get(rows[s], key, op) for s in ROWS]
            if None in vals:
                per = None; break
            per[op] = dict(last3=sum(vals) / 3, final=vals[-1])
        if per is None:
            continue
        per["mean"] = {w: sum(per[o][w] for o in OPS) / 4 for w in ("last3", "final")}
        for f, (a, b) in FAM.items():
            per[f] = {w: (per[a][w] + per[b][w]) / 2 for w in ("last3", "final")}
        out[key] = per
    return out


def main():
    recs = {}
    dirs = [D] + ([Path(sys.argv[1])] if len(sys.argv) > 1 else [])
    for d in dirs:
        for p in sorted(d.glob("e35*_dev_*_f*_*.json")):
            r = json.loads(p.read_text())
            if not r.get("complete") or r.get("dev_updates") != 24576:
                continue
            recs[(r["arm"], r["fraction"], r["seed"])] = (p, r)
    table, L = {}, ["# E35-B stage 1a tables (from receipts; no readings)", ""]
    for (arm, f, seed), (p, r) in sorted(recs.items()):
        mt = metrics(r)
        diag = r["diagnostics"][-1] if r.get("diagnostics") else None
        table[f"{arm}/f{f}/{seed}"] = dict(file=str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p), metrics=mt, cpu_total=r.get("cpu_total"),
                                          final_loss=r["curve"][-1]["loss"], finite=all(c["finite_losses"] for c in r["curve"]), diagnostics_final=diag)
    for key, label in (("held12", "held12"), ("held40", "held40"), ("taught12", "taught12"), ("off12", "held12, module off"), ("off40", "held40, module off")):
        L += [f"## {label}", "", "| arm | f | seed | mean (last3 / final) | add | sub | max | min | add/sub | max/min |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in table.items():
            mt = v["metrics"]
            if not mt or key not in mt:
                continue
            arm, f, seed = k.split("/")
            cell = lambda x: f"{mt[key][x]['last3']:.1f} / {mt[key][x]['final']:.1f}"
            L.append(f"| {arm} | {f[1:]} | {seed} | " + " | ".join(cell(x) for x in ("mean",) + OPS + tuple(FAM)) + " |")
        L.append("")
    L += ["## Gate diagnostics at the end of each module birth (training batch at the last update; by the role of the byte just read)", ""]
    for k, v in table.items():
        dg = v["diagnostics_final"]
        if not dg:
            continue
        L += [f"**{k}** (step {dg['step']}): max op weight at '=' {dg['at_equals_max_op_weight']:.3f}; argmax op counts at '=' (mul, div, big, small) {dg['at_equals_argmax_op_count']}", "",
              "| byte just read | n | gate A | gate B | q tens | q units | op weights (mul, div, big, small) |", "|---|---|---|---|---|---|---|"]
        for role, d in dg.items():
            if isinstance(d, dict):
                L.append(f"| {role} | {d['n']} | {d['gA']:.3f} | {d['gB']:.3f} | {d['q_tens']:.3f} | {d['q_units']:.3f} | " + ", ".join(f"{w:.2f}" for w in d["op_weights"]) + " |")
        L.append("")
    L += ["## CPU", "", "| birth | CPU s | final loss | finite |", "|---|---|---|---|"]
    L += [f"| {k} | {v['cpu_total']:.0f} | {v['final_loss']:.4g} | {v['finite']} |" for k, v in table.items()]
    (D / "E35B_STAGE1A_TABLE.json").write_text(json.dumps(table, indent=1))
    (D / "E35B_STAGE1A_TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
