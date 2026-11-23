"""print one row per run from the results files (helper rcseeds)"""
import json, sys
files = sys.argv[1:] or ["rcseeds_runs/res.jsonl", "rc_runs/full_res.jsonl", "seeds23_runs/res_t23.jsonl", "idea12_runs/life_res.jsonl", "rules_runs/res_t23.jsonl"]
for f in files:
    try: L = [json.loads(x) for x in open(f)]
    except FileNotFoundError: continue
    for r in L:
        if r.get("net") != 12: continue
        P = {p["line"]: p for p in r["probes"]}; fm = r["first_met"]; g = lambda k: fm.get(k, {}).get("right", float("nan"))
        print(f"{f.split('/')[0]:14s} s{r['seed']} opts={r['opts']} rho={r.get('rho', 0)} tgt={r.get('tgt', 0)} | bits {r['bits']['total']:,} (text {r['bits']['text']:,}) | "
              f"b P0 {g('P0_copy_b'):.3f} P1 {g('P1_rev_b'):.3f} P3 {g('P3_succ_b'):.3f} P4 {g('P4_copy_b'):.3f} P5 {g('P5_rev_b'):.3f} c P0 {g('P0_copy_c'):.3f} | "
              f"text {P[17000]['text']:.3f}->{P[29000]['text']:.3f} | succ {P[23000]['succ']:.3f}->{P[26000]['succ']:.3f}->{P[29000]['succ']:.3f} | copy/rev {P[29000]['copy']:.2f}/{P[29000]['rev']:.2f}")
