import sys, json, time
B = sys.argv[1]; sys.path.insert(0, B)
import ptread, npnet as N
SEEDS = (73129, 93683, 116089); E6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73")
t0 = time.time(); bad = 0
for seed in SEEDS:
    P = N.earlier_panels(seed)
    red = {f"{t[0]}83": N.reduction_panel(t, .10, seed) for t in N.F4[:2]}
    for rule in "SB":
        e40 = json.load(open(f"{B}/rec/e40_lived_{rule}_{seed}.json")); e48 = json.load(open(f"{B}/rec/e48_{rule}_{seed}_r16.json"))
        want0 = e40["rows"][0]["scores"]
        net = N.Net(ptread.load(f"{B}/w/life_{rule}_{seed}.pt"))
        got = {n: net.score(P[n])[0] for n in E6}; w = {n: want0["earlier_k12"][n]["exact"] for n in E6}
        gr = {n: net.score(red[n], False) for n in red}; wr = {n: (want0["held_plain_k0"][n]["reduction"]["exact"], want0["held_plain_k0"][n]["reduction"]["n"]) for n in red}
        # the E48 term at row 1024 on the lived network (code rows from the term file)
        term = ptread.load(f"{B}/w/e48_{rule}_{seed}_r16_row1024.pt")
        sd = dict(net.w); bk = sd["book.weight"].copy(); bk[[101, 102, 103, 104]] = term["code_rows"]; sd["book.weight"] = bk
        tn = N.Net(sd, term)
        gt = {n: tn.score(red[n], True) for n in red}; row = e48["rows"][-1]["scores"]["held_plain_k0"]; wt = {n: (row[n]["reduction"]["exact"], row[n]["reduction"]["n"]) for n in red}
        ok = got == w and gr == wr and gt == wt; bad += not ok
        print(seed, rule, "earlier row 0:", got, "receipt:", w, "| reduction row 0:", gr, "receipt:", wr, "| E48 term row 1024:", gt, "receipt:", wt, "EQUAL" if ok else "DIFFER", f"{time.time()-t0:.0f}s", flush=True)
print("runs that differ from the receipts:", bad)
