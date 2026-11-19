import json, sys
for pre in sys.argv[1:]:
    r = json.load(open(f"out/{pre}_LL.json"))
    for e, row in r.items():
        f = lambda v: f"{v['S']:.3f}" + ("" if v["a"] and v["b"] else ("(b)" if v["a"] else "(A)"))
        bad9 = [k for k, v in row["nine"].items() if not (v["a"] and v["b"])]
        print(pre, e, "nine:", "all pass" if not bad9 else "fail " + ",".join(f"{k} {f(row['nine'][k])}" for k in bad9),
              "| near enl", " ".join(f(row["near"]["enlarged"][f"L{L}"]) for L in (0, 1, 2)), "| near 1/8", " ".join(f(row["near"]["within_1_8"][f"L{L}"]) for L in (0, 1, 2)))
