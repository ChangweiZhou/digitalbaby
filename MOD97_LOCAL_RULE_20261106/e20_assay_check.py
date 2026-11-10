"""E20-A assay check: the memory scorer against two stand-in responders, for the four held operator pairs of three seeds.
  oracle        answers every prompt correctly: must score every item and every twin
  window-only   answers from the last ten bytes of the prompt alone (it memorises one answer per ten-byte suffix): must score no twin,
                and must still score the leading control fully, because there the whole prompt is inside the last ten bytes
A failure of either stand-in to behave so means the assay cannot separate memory use from its absence. Exits nonzero on failure."""
import json, os, sys
os.environ["E16_COMPGAP"] = "6"
import exp16_integrated as G, exp11_cell as E

real = E.respond; out = {}; ok = True
for seed in (73129, 93683, 116089):
    G.panels(seed); mem = G.MEMORY[0]
    truth = {bytes(p): t for key in ("twins", "lead") for tw in mem[key].values() for xy in tw for p, t in xy}
    truth.update({bytes(p): t for key in ("within_hidden", "taught_hidden") for p, t in mem[key]})
    E.respond = lambda m, prompts, store=None, cap=4, silence=False: [truth[bytes(p)] for p in prompts]
    orc = G.score_memory(None, mem)
    table = {}
    def window_only(m, prompts, store=None, cap=4, silence=False):
        return [table.setdefault(bytes(p[-10:]), truth[bytes(p)]) for p in prompts]
    E.respond = window_only
    win = G.score_memory(None, mem)
    E.respond = real
    tot = lambda r, key, f: sum(v[f] for v in r[key].values())
    n = sum(v["pairs"] for v in orc["twins"].values())
    row = dict(twins=n, oracle_twins_both=tot(orc, "twins", "both_correct"), oracle_items=tot(orc, "twins", "items_correct"), oracle_lead_both=tot(orc, "lead", "both_correct"),
               oracle_within_hidden=orc["within_hidden"]["exact"], oracle_taught_hidden=orc["taught_hidden"]["exact"],
               window_only_twins_both=tot(win, "twins", "both_correct"), window_only_twin_items=tot(win, "twins", "items_correct"), window_only_lead_both=tot(win, "lead", "both_correct"))
    good = (row["oracle_twins_both"] == n and row["oracle_items"] == 2 * n and row["oracle_lead_both"] == n and row["oracle_within_hidden"] == len(mem["within_hidden"])
            and row["oracle_taught_hidden"] == len(mem["taught_hidden"]) and row["window_only_twins_both"] == 0 and row["window_only_twin_items"] == n and row["window_only_lead_both"] == n)
    row["passed"] = good; ok &= good; out[seed] = row; print(seed, row, flush=True)
(G.OUT / "E20_ASSAY_CHECK.json").write_text(json.dumps(dict(passed=ok, per_seed=out), indent=1))
sys.exit(0 if ok else 1)
