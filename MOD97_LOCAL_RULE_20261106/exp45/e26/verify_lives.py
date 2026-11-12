"""E26 stage 3: verify the six lives and write their tables and the readings of P26.3-P26.5 (E26_LIVES.json, E26_LIVES.md).
Reads only. Run from MOD97_LOCAL_RULE_20261106: OMP_NUM_THREADS=1 python3 exp45/e26/verify_lives.py"""
import hashlib, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; MOD = HERE.parent.parent
sys.path.insert(0, str(MOD))
import torch as T
import exp20_reference as R
import exp16_integrated as X16
import exp26_candidate as C

SEEDS = (73129, 93683, 116089); ARMS = ("S", "B"); HELD = ("+-", "-+", "><", "<>")
F1 = ("add97", "sub97", "max97", "min97"); F3 = ("add73", "sub73", "max73", "min73"); DEV = ("add89", "sub89", "max89", "min89")
pct = lambda d: 100.0 * d["exact"] / d["n"]
fam = lambda sc, ns: sum(pct(sc[n]["12"]) for n in ns) / len(ns)
pinned = {f: hashlib.sha256(subprocess.run(["git", "show", f"360e1b8d:MOD97_LOCAL_RULE_20261106/{f}"], cwd=MOD, capture_output=True, check=True).stdout).hexdigest() for f in C.SOURCES}
out = dict(checks={}, lives={})
for s in SEEDS:
    for a in ARMS:
        p = HERE / f"e26_life_{a}_{s}.pt"; r = json.loads(p.with_suffix(".json").read_text()); c = r["complete"]
        st0 = r["stages"][0]["settings"]; birth = json.loads((HERE / f"e26_dev_{a}_{s}.json").read_text())
        ck = dict(complete_record=isinstance(c, dict) and c.get("stages") == 7, seven_stages=len(r["stages"]) == 7,
                  optimizer_steps_36864=c["optimizer_steps"] == [36864],
                  weights_sha256_matches_file=X16.sha_file(p) == c["weights_sha256"],
                  tensors_sha256_matches_file=R.tensors_sha(T.load(p)) == c["tensors_sha256"],
                  birth_is_admitted_birth=st0["birth_sha256"] == birth["weights_sha256"] == X16.sha_file(HERE / f"e26_dev_{a}_{s}.pt"),
                  sources_equal_360e1b8d=st0["provenance"]["sources_sha256"] == pinned,
                  arm_local=(r["arm"], r["local"]) == (a, C.ARMS[a]) and r["seed"] == s,
                  finite_losses_every_stage=all(x["finite_losses"] for x in r["stages"][1:]),
                  weight_decay_read_back=[g["weight_decay"] for g in c["optimizer_read_back"]["groups"]],
                  taught_panel_only_taught_presented=all(x["exposure_panels"]["within_or_held_presented"] == 0 for x in r["stages"][1:]))
        out["checks"][f"{a}_{s}"] = ck
        rows = []
        for x in r["stages"]:
            sc = x["scores"]; cp = sc["compose"]
            rows.append(dict(stage=int(x["stage"]), task=x["task"] or "birth", taught=pct(sc["taught"]), within=pct(cp["within"]),
                             held={h: pct(cp[h]) for h in HELD}, plus_minus=100.0 * (cp["+-"]["exact"] + cp["-+"]["exact"]) / (cp["+-"]["n"] + cp["-+"]["n"]),
                             within_final=100.0 * cp["within"]["final"] / cp["within"]["n"],
                             F1=fam(sc, F1), F3=fam(sc, F3), dev89=fam(sc, DEV), hard89=(pct(sc["add89"]["12"]) + pct(sc["sub89"]["12"])) / 2,
                             silenced_compose={k: pct(v) for k, v in sc.get("silenced_compose", {}).items()}, loss=x.get("loss")))
        out["lives"][f"{a}_{s}"] = dict(rows=rows, cpu_total=c["cpu_total"], cpu_updates=c["cpu_updates"], cpu_scoring=c["cpu_scoring"], data_digest=c["data_digest"]["sha256"])
    out["checks"][f"data_digest_S_equals_B_{s}"] = out["lives"][f"S_{s}"]["data_digest"] == out["lives"][f"B_{s}"]["data_digest"]
end = lambda a, s, k: out["lives"][f"{a}_{s}"]["rows"][-1][k]
V = {}
V["P26.3"] = dict(values=[end("S", s, "within") for s in SEEDS], seeds_meeting=[s for s in SEEDS if end("S", s, "within") >= 85])
V["P26.4"] = dict(values=[end("S", s, "plus_minus") for s in SEEDS], seeds_meeting=[s for s in SEEDS if end("S", s, "plus_minus") >= 40])
both = lambda a: [s for s in SEEDS if end(a, s, "within") >= 85 and end(a, s, "plus_minus") >= 40]
V["P26.5"] = dict(within=[end("B", s, "within") for s in SEEDS], plus_minus=[end("B", s, "plus_minus") for s in SEEDS], seeds_meeting_both=both("B"))
for k in ("P26.3", "P26.4"):
    V[k]["holds"] = len(V[k]["seeds_meeting"]) >= 2
V["P26.5"]["holds"] = len(V["P26.5"]["seeds_meeting_both"]) >= 2
V["candidate_pair_S"] = dict(seeds_meeting_both=both("S"), exists=len(both("S")) >= 2)
V["paired_S_minus_B_end"] = {s: dict(within=end("S", s, "within") - end("B", s, "within"), plus_minus=end("S", s, "plus_minus") - end("B", s, "plus_minus")) for s in SEEDS}
out["readings"] = V
(HERE / "E26_LIVES.json").write_text(json.dumps(out, indent=1))
L = ["# E26 stage 3: the six lives", "", "Written by verify_lives.py from the receipts. Percent whole answers (`exact`). within = new instances of taught pairs (600); "
     "each held pair 600; plus-minus = +- and -+ pooled (1,200); F1/F3/dev89 = family means at k=12; hard89 = add89/sub89 mean at k=12.", "", "## Checks", ""]
L += [f"- {k}: {v}" for k, v in out["checks"].items()]
L += ["", "## Readings", ""] + [f"- {k}: {json.dumps(v)}" for k, v in V.items()]
L += ["", "## Life checkpoints", "", "| arm | seed | stage | taught | within | +- | -+ | >< | <> | plus-minus | F1 | F3 | dev89 | hard89 |", "|" + "---|" * 14]
for s in SEEDS:
    for a in ARMS:
        for r in out["lives"][f"{a}_{s}"]["rows"]:
            L.append(f"| {a} | {s} | {r['stage']} {r['task']} | {r['taught']:.1f} | {r['within']:.1f} | " + " | ".join(f"{r['held'][h]:.1f}" for h in HELD)
                     + f" | {r['plus_minus']:.1f} | {r['F1']:.1f} | {r['F3']:.1f} | {r['dev89']:.1f} | {r['hard89']:.1f} |")
L += ["", "## Paired S minus B by checkpoint (within / plus-minus)", "", "| stage | " + " | ".join(str(s) for s in SEEDS) + " |", "|---|---|---|---|"]
for i in range(7):
    L.append(f"| {i} | " + " | ".join(f"{out['lives'][f'S_{s}']['rows'][i]['within'] - out['lives'][f'B_{s}']['rows'][i]['within']:+.1f} / "
                                      f"{out['lives'][f'S_{s}']['rows'][i]['plus_minus'] - out['lives'][f'B_{s}']['rows'][i]['plus_minus']:+.1f}" for s in SEEDS) + " |")
L += ["", "## Silenced readout, composition, end of life", ""] + [f"- {a} {s}: {json.dumps({k: round(v, 1) for k, v in out['lives'][f'{a}_{s}']['rows'][-1]['silenced_compose'].items()})}" for s in SEEDS for a in ARMS]
L += ["", "## CPU seconds", "", "| life | total | updates | scoring |", "|---|---|---|---|"]
L += [f"| {a} {s} | {out['lives'][f'{a}_{s}']['cpu_total']:.0f} | {out['lives'][f'{a}_{s}']['cpu_updates']:.0f} | {out['lives'][f'{a}_{s}']['cpu_scoring']:.0f} |" for s in SEEDS for a in ARMS]
(HERE / "E26_LIVES.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
