import json, subprocess
W, D, M = "origin/claude/gallant-dijkstra-i9zgsr", "origin/codex/minifly-consolidated-20261023", "MOD97_LOCAL_RULE_20261106"
show = lambda ref, p: json.loads(subprocess.run(["git", "show", f"{ref}:{M}/exp45/{p}"], capture_output=True, check=True).stdout)
SEEDS = (73129, 93683, 116089)
def m9(c, at):
    rows = [r["R"] for r in c if at - 8 * 256 <= r["step"] <= at]; assert len(rows) == 9, len(rows)
    return sum(rows) / 9
def rise9(c):
    d = c[-1]["step"]; return m9(c, d) - m9(c, d - 8192)
n = 0
for cfg in ("G", "F10", "P8"):
    for rule in "SB":
        rs = [rise9(show(W, f"e42/e42_{cfg}_{rule}_{s}.json")["curve"]) for s in SEEDS]
        n += sum(r >= 3 for r in rs)
        print("E42 f=0.50", cfg, rule, "nine-row rise of R over the last 8,192:", [round(r, 1) for r in rs])
print("E42: curves not flattened by the nine-row rule:", n, "of 18")
for H, fp in ((512, 25), (512, 10), (256, 25), (256, 10)):
    for rule in "SB":
        rs = [rise9(show(D, f"e41/e41_{rule}_h{H}_f{fp}_{s}.json")["curve"]) for s in SEEDS]
        print(f"E41 H={H} f=0.{fp} {rule}: nine-row rise of R over the last 8,192:", [round(r, 1) for r in rs])
