"""Summary table of rcuncued_runs/res_t23.jsonl and res_t27.jsonl."""
import json
for t, mp in (("t23", "succ"), ("t27", "map")):
    try: rows = [json.loads(l) for l in open(f"rcuncued_runs/res_{t}.jsonl")]
    except FileNotFoundError: continue
    rules = ("copy", "rev", "succ") if t == "t23" else ("copy", "swap", "rot", "skip", "map")
    print(f"\n### {t}\n\n| net | cue | seed | bits total (input / answer / text) | end probe {' / '.join(rules)} / text | {mp} probe series (every 1000 lines from its first teaching) | first-met b by phase | recruits |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r['cue'], r['seed'], r['opts'].get('rc_match', 0))):
        nm = "net13" + (" + rc_match=%d" % r['opts']['rc_match'] if r['opts'].get('rc_match') else "")
        b = r['bits']; e = r['probes'][-1]
        st = 17000 if t == "t23" else 3000
        ser = " ".join(f"{p[mp]:.2f}" for p in r['probes'] if p['line'] >= st)
        fm = ", ".join(f"{k.split('_')[0]} {k.split('_')[1]} {v['right']:.2f}" for k, v in r['first_met'].items() if k.endswith('_b'))
        print(f"| {nm} | {r['cue']} | {r['seed']} | {b['total']:,} ({b['input']:,} / {b['answer']:,} / {b['text']:,}) | {' / '.join(f'{e[x]:.2f}' for x in rules + ('text',))} | {ser} | {fm} | {r.get('nrecruit', '-')} |")
