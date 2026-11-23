import json
rs = [json.loads(l) for l in open("rcnewlife_runs/res.jsonl")]
def name(r): return ("rc" if "rc" in r["opts"] else "base") if r["net"] == 12 else "K8" if r["net"] == 10 else "store"
print("| net | cue | seed | bits total | input / answer / text | end probe copy / swap / rot / skip / **map** / text | map: after P1 (6000) -> after P6 (23000) -> 26000 | text after P2 (11000) | first-met b: P0 rot, P1 map, P3 swap, P4 skip, P6 map, P7 copy | first window P5 rot / P6 map / P8 swap | recruits | s |")
print("|"+"---|"*13)
for r in sorted(rs, key=lambda r: (-r["cue"], ["store","K8","base","rc"].index(name(r)), r["seed"])):
    b = r["bits"]; P = {p["line"]: p for p in r["probes"]}; e = P[29000]; f = r["first_met"]; w = r["windows"]
    fb = lambda k: f"{f[k]['right']:.2f}" if k in f else "-"
    print(f"| {name(r)} | {r['cue']} | {r['seed']} | {b['total']:,} | {b['input']:,} / {b['answer']:,} / {b['text']:,} | "
          f"{e['copy']:.2f} / {e['swap']:.2f} / {e['rot']:.2f} / {e['skip']:.2f} / **{e['map']:.2f}** / {e['text']:.2f} | "
          f"{P[6000]['map']:.2f} -> {P[23000]['map']:.2f} -> {P[26000]['map']:.2f} | {P[11000]['text']:.2f} | "
          f"{fb('P0_rot_b')}, {fb('P1_map_b')}, {fb('P3_swap_b')}, {fb('P4_skip_b')}, {fb('P6_map_b')}, {fb('P7_copy_b')} | "
          f"{w['P5'][0]:.2f} / {w['P6'][0]:.2f} / {w['P8'][0]:.2f} | {r.get('nrecruit') or '-'} | {r['s']} |")
