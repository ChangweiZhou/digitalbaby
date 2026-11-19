"""Task 1: the nine entries over lives differing in the last bit; for each entry the smallest and largest S, and whether the smallest
is above level (a) and whether every life is above level (b) (levels.py's definitions)."""
import glob, json, subprocess, sys
rows = {}
for kind, seed, ref in [("N256", 73129, "N256_73129_16384_u0"), ("N256", 93683, "N256_93683_16384_u0"), ("N256", 116089, "N256_116089_16384_u0"), ("N64", 73129, "N64_73129_ref")]:
    tags = [ref] + sorted(p[4:-6] for p in glob.glob(f"out/{kind}_{seed}_p*.stats"))
    lv = {}
    for t in tags:
        if not glob.glob(f"out/{t}_levels.json"):
            subprocess.run([sys.executable, "levels.py", t, str(seed), "16384"], check=True, capture_output=True)
        lv[t] = json.load(open(f"out/{t}_levels.json"))
    ent = {}
    for e in lv[ref]:
        S = [lv[t][e]["S"] for t in tags]
        ent[e] = dict(min=min(S), max=max(S), smallest_above_a=all(lv[t][e]["above_level"] for t in tags),
                      all_above_b=all(lv[t][e]["stricter"]["above"] for t in tags), level_a=[round(lv[t][e]["level"], 4) for t in tags])
    rows[f"{kind} {seed}"] = dict(lives=len(tags), tags=tags, entries=ent)
    print(f"{kind} {seed} lives {len(tags)}: " + "; ".join(f"{e} {v['min']:.3f}-{v['max']:.3f}{'' if v['smallest_above_a'] and v['all_above_b'] else ' BELOW'}" for e, v in ent.items()), flush=True)
json.dump(rows, open("out/task1.json", "w"), indent=1)
