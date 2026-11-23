import json, sys
for f in sys.argv[1:]:
    for l in open(f):
        d = json.loads(l); L = d["lines"]; fr = d["free"][-1]
        o = {k: v for k, v in d["opts"].items() if k not in ("L",)}
        print(f.split("/")[-1], d["net"], d["fmt"], o, "s=%d" % d["s"], "P1 old/new %.3f / %.3f" % (L["P1_old"]["right"], L["P1_new"]["right"]),
              "free8 %.2f" % fr["8"], "s-bytes P1new c0/c1 %.3f/%.3f" % (d["bytes"]["P1_s_new_carry0"]["right"], d["bytes"]["P1_s_new_carry1"]["right"]))
        for k, v in d.get("carry_diag", {}).items():
            print("   ", k, {x: y for x, y in v.items() if x not in ("all_corr",)})
        if "nrecruit" in d: print("    recruit", d["nrecruit"])
