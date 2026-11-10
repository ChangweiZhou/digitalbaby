"""Tables for E16 from exp45/e16_life_{T,D}_{seed}.json -> exp45/SUMMARY_E16.md."""
import json
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "exp45"
SEEDS = tuple(int(x) for x in os.environ.get("E16_SEEDS", "73129,93683,116089").split(","))   # E16-F: E16_SEEDS=200003,300007,400009
SUMMARY = OUT / os.environ.get("E16_SUMMARY", "SUMMARY_E16.md")
F1 = ("add97", "sub97", "max97", "min97"); F3 = ("add73", "sub73", "max73", "min73"); DEV = ("add89", "sub89", "max89", "min89")
HELD = ("+-", "-+", "><", "<>")
pct = lambda d: 100 * d["exact"] / d["n"]
pctf = lambda d: 100 * d["final"] / d["n"]


ARMS = ("T", "D", "M", "T_oracle", "T_balanced", "T_all_balanced", "T_share", "T_res2048", "T_res4096", "T_res8192", "T_res16384", "T_tot24576", "T_tot65536", "T_raw65536", "T_raw65536_online", "T_raw65536_drift_online", "D_drift", "T_raw65536_lossall", "T_raw65536_nostore")
LABEL = {"T": "T learner (uniform replay)", "D": "D twin (joint ground truth)", "M": "M backprop on T's batches", "T_oracle": "T on the twin's stream",
         "T_balanced": "T balanced per family (1,024/block)", "T_all_balanced": "T balanced, every entry", "T_share": "T earlier families 50% of batch",
         "T_res2048": "T reservoir 2,048/family", "T_res4096": "T reservoir 4,096/family", "T_res8192": "T reservoir 8,192/family", "T_res16384": "T reservoir 16,384/family", "T_tot24576": "T one reservoir 24,576, no labels", "T_tot65536": "T one reservoir 65,536, no labels", "T_raw65536": "T raw bytes verbatim, 65,536, no parser", "T_raw65536_online": "T raw 65,536, admitted online (control)", "T_raw65536_drift_online": "T raw 65,536, drifting stream, no stages", "D_drift": "D twin on the drifting stream", "T_raw65536_lossall": "T raw 65,536, loss on every byte", "T_raw65536_nostore": "T raw 65,536, exact-key store removed"}


E17_ARMS = ("T_cell1_raw65536", "D_cell1", "T_bank_raw65536", "D_bank", "T_bank2_raw65536", "D_bank2", "T_bank2_frz1_raw65536", "D_bank2_frz1", "T_bank2_frz1_xfb_raw65536", "M_bank2_frz1_raw65536", "T_bank2_frz1_tup_raw65536", "T_bank2_frz1_xfb_tup_raw65536", "D_bank2_nbd", "D_bank2_nbd_frz1", "T_bank2_nbd_raw65536", "T_bank2_nbd_frz1_raw65536", "T_bank2_nbd_frz1_xfb_tup_raw65536", "T_bank2_nbd_frz1_xfb_raw65536", "T_bank2_nbd_frz1_tup_raw65536", "T_bank2_nbd_frzb_raw65536", "T_bank2_nbd_tall_raw65536", "T_bank2_nbd_tup_raw65536", "M_bank2_nbd_raw65536")
RS_ARMS = tuple(f"{a}_raw65536_drift_online_rs{r}" for a in ("T", "M") for r in (0, 25, 75)) + ("M_raw65536_drift_online",) + tuple(f"T_raw65536_drift_online_rs{r}" for r in (88, 94))   # 88, 94: E16-M4 (shares 0.875, 0.9375)
ARMS = ARMS + ("D_lossall",) + E17_ARMS + RS_ARMS
LABEL["D_lossall"] = "D twin, loss on every byte"
LABEL.update({a: f"E16-M3 {a[0]} raw 65,536, drift, replay share {a.rsplit('_rs', 1)[1] if '_rs' in a else '50'}%" for a in RS_ARMS})
LABEL.update({"T_bank2_nbd_tall_raw65536": "E17 bank2 nbd, nothing frozen, sign feedback, credit through time at every layer", "T_bank2_nbd_tup_raw65536": "E17 bank2 nbd, nothing frozen, trace extended to the write projections only", "M_bank2_nbd_raw65536": "E17 bank2 nbd, nothing frozen, backprop on the learner's pipeline"})
LABEL.update({"T_cell1_raw65536": "E17 cell1 (no window), raw 65,536", "D_cell1": "E17 twin, cell1", "T_bank_raw65536": "E17 bank (no window, fixed rates), raw 65,536", "D_bank": "E17 twin, bank",
              "T_bank2_raw65536": "E17 bank2 (two stacked banks), raw 65,536", "D_bank2": "E17 twin, bank2",
              "T_bank2_frz1_raw65536": "E17 bank2, first bank frozen, raw 65,536", "D_bank2_frz1": "E17 twin, bank2, first bank frozen", "T_bank2_frz1_xfb_raw65536": "E17 bank2 frozen, exact feedback", "M_bank2_frz1_raw65536": "E17 bank2 frozen, backprop on the learner pipeline", "T_bank2_frz1_tup_raw65536": "E17 bank2 frozen, trace to the projection", "T_bank2_frz1_xfb_tup_raw65536": "E17 bank2 frozen, exact feedback, trace to the projection", "D_bank2_nbd": "E17 Phase 2 twin, bank2, embeddings without decay", "D_bank2_nbd_frz1": "E17 Phase 2 twin, bank2 no-decay, first bank frozen", "T_bank2_nbd_raw65536": "E17 Phase 2 bank2, sign rule", "T_bank2_nbd_frz1_raw65536": "E17 Phase 2 bank2 frozen, sign rule", "T_bank2_nbd_frz1_xfb_tup_raw65536": "E17 Phase 2 bank2 frozen, exact feedback, trace to the projection", "T_bank2_nbd_frz1_xfb_raw65536": "E17 Phase 3 bank2 frozen, exact feedback alone", "T_bank2_nbd_frz1_tup_raw65536": "E17 Phase 3 bank2 frozen, trace to the projection alone", "T_bank2_nbd_frzb_raw65536": "E17 Phase 3 bank2, embeddings alone frozen, sign rule on both banks"})


def twin_of(arm):
    """The comparator shares the arm's stream and objective (audit, 8 Nov): drift -> D_drift, every-byte loss -> D_lossall, E17 arch -> its own twin."""
    if "drift_online" in arm and "lossall" in arm:      # 10 Nov audit: no such comparator exists; never fall through to the answer-loss one
        return "D_drift_lossall"
    if "drift_online" in arm:
        return "D_drift"
    if "lossall" in arm:
        return "D_lossall"
    for a in ("cell1", "bank2_nbd_frz1", "bank2_nbd", "bank2_frz1", "bank2", "bank"):     # longest tag first: "_bank2_" is not "_bank_"
        if f"_{a}_" in arm:
            return f"D_{a}"
    return "D"


def load(arm, seed):
    p = OUT / f"{'e17' if arm in E17_ARMS else 'e16'}_life_{arm}_{seed}.json"
    return json.load(open(p)) if p.exists() else None


def fam_mean(sc, fam, k=12):
    return sum(pct(sc[n][str(k)] if str(k) in sc[n] else sc[n][k]) for n in fam) / len(fam)


def K(sc, n, k):
    d = sc[n]; return d[str(k)] if str(k) in d else d[k]


def main():
    lines = ["# E16 summary: the integrated toy learner\n", "Scores are exact answers (%) on sealed panels; composition rows give final-answer accuracy. Blocks: F1.1–F1.2 mod-97 operations with distractors, F2.1–F2.2 taught operator pairs, F3.1–F3.2 mod-73 operations.\n"]
    runs = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    runs = {k: v for k, v in runs.items() if v is not None}
    # per-block family means
    lines += ["## Family means by block (k = 12 for operations; held-pair final-answer mean for composition)\n",
              "| Arm | Seed | Block | F1 mean | F1 add | F1 sub | F3 mean | Held comp | +- | -+ | >< | <> | within | dev89 mean |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (a, s), d in runs.items():
        for st in d["stages"]:
            sc = st["scores"]; c = sc["compose"]
            lines.append(f"| {a} | {s} | {st['task'] or 'birth'} | {fam_mean(sc, F1):.1f} | {pct(K(sc,'add97',12)):.1f} | {pct(K(sc,'sub97',12)):.1f} | {fam_mean(sc, F3):.1f} | "
                         f"{sum(pctf(c[h]) for h in HELD)/4:.1f} | " + " | ".join(f"{pctf(c[h]):.1f}" for h in HELD) + f" | {pctf(c['within']):.1f} | {fam_mean(sc, DEV):.1f} |")
    # final table: T vs D and the 80% gate
    def contract(a_, d_):
        """Second audit of 10 Nov: no ratio without a comparison contract. Returns (ratio allowed, status text)."""
        def f(r):
            st_ = [x["settings"] for x in r["stages"] if "settings" in x]; st_ = st_[-1] if st_ else {}
            return ("corrected" if "keyed by split" in st_.get("panels", "") else "original"), st_.get("loss", "answer"), st_.get("panel_sha256"), r.get("complete")
        (pa, la, ha, ca), (pd_, ld, hd, cd) = f(a_), f(d_)
        if pa != pd_:
            return False, f"WITHHELD: panels differ ({pa} / {pd_})"
        if la != ld:
            return False, f"WITHHELD: objective differs ({la} / {ld})"
        if ca and cd and ha and hd:
            return True, "bound: completion records and panel hashes present" + ("" if ha == hd else "; single-operation panels are architecture-specific, compare composition only")
        return True, "legacy: no completion record or panel hash; not admissible to a new gate"
    lines += ["\n## After the last block: learner against twin (gate: T ≥ 80% of D on every family)\n",
              "| Seed | Arm: family | arm | D | arm/D | F1 after block 2 (T) | T k=20 | T silenced k=12 | Evidence status |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for s in SEEDS:
        for a in ARMS:
            t = runs.get((a, s)); dd = runs.get((twin_of(a), s))
            if a.startswith("D") or not t or not dd or len(t["stages"]) < 7 or len(dd["stages"]) < 7:
                continue
            st, sd = t["stages"][-1]["scores"], dd["stages"][-1]["scores"]; s2 = t["stages"][2]["scores"]
            rows = [("F1 ops k=12", fam_mean(st, F1), fam_mean(sd, F1), fam_mean(s2, F1), sum(pct(K(st, n, 20)) for n in F1) / 4, sum(pct(st["silenced_k12"][n]) for n in F1) / 4),
                    ("F3 ops k=12", fam_mean(st, F3), fam_mean(sd, F3), None, sum(pct(K(st, n, 20)) for n in F3) / 4, sum(pct(st["silenced_k12"][n]) for n in F3) / 4),
                    ("held comp final", sum(pctf(st["compose"][h]) for h in HELD) / 4, sum(pctf(sd["compose"][h]) for h in HELD) / 4, None, None, sum(pctf(st["silenced_compose"][h]) for h in HELD) / 4)]
            allowed, status = contract(t, dd)
            for name, tv, dv, b2, k20, sil in rows:
                lines.append(f"| {s} | {LABEL.get(a, a)}: {name} | {tv:.1f} | {dv:.1f} | {f'{tv/dv:.2f}' if allowed else 'withheld'} | {'' if b2 is None else f'{b2:.1f}'} | {'' if k20 is None else f'{k20:.1f}'} | {sil:.1f} | {status} |")
    # E16-D: retention across arms
    lines += ["\n## Retention across arms: what each family keeps after the stream moves on (block 6 vs the family's own last block)\n",
              "| Arm | Seed | F1 k=12 after 2 → 6 | F1 add/sub after 6 | Held comp (final answer) after 4 → 6 | Held comp full scratchpad exact after 6 | +- / -+ after 6 | within after 4 → 6 | F3 after 6 |", "|---|---|---|---|---|---|---|---|---|"]
    for (a, s), d in runs.items():
        if len(d["stages"]) < 7:
            continue
        s2, s4, s6 = (d["stages"][i]["scores"] for i in (2, 4, 6))
        hc = lambda sc: sum(pctf(sc["compose"][h]) for h in HELD) / 4
        hx = sum(pct(s6["compose"][h]) for h in HELD) / 4
        lines.append(f"| {LABEL.get(a, a)} | {s} | {fam_mean(s2, F1):.1f} → {fam_mean(s6, F1):.1f} | {pct(K(s6,'add97',12)):.0f}/{pct(K(s6,'sub97',12)):.0f} | {hc(s4):.1f} → {hc(s6):.1f} | {hx:.1f} | "
                     f"{pctf(s6['compose']['+-']):.0f}/{pctf(s6['compose']['-+']):.0f} | {pctf(s4['compose']['within']):.0f} → {pctf(s6['compose']['within']):.0f} | {fam_mean(s6, F3):.1f} |")
    # per-operation detail at the end
    lines += ["\n## Per-operation exact % at k = 0 / 8 / 12 / 20 after the last block\n", "| Arm | Seed | " + " | ".join(F1 + F3) + " |", "|---|---|" + "---|" * 8]
    for (a, s), d in runs.items():
        if len(d["stages"]) >= 7:
            sc = d["stages"][-1]["scores"]
            lines.append(f"| {a} | {s} | " + " | ".join("/".join(f"{pct(K(sc, n, k)):.0f}" for k in (0, 8, 12, 20)) for n in F1 + F3) + " |")
    if runs:
        sc0 = next(iter(runs.values()))["stages"][0]
        lines += ["\n## Shortcut ceilings on the held pairs (final answer from a visible-only rule, %)\n", "| Pair | ignore o1, use a | ignore o1, use b | ignore o2 | answer = c |", "|---|---:|---:|---:|---:|"]
        for h, v in sc0["shortcut_ceilings"].items():
            lines.append(f"| {h} | " + " | ".join(f"{100*v[k]:.1f}" for k in ("ignore_o1_use_a", "ignore_o1_use_b", "ignore_o2", "answer_is_c")) + " |")
    else:
        lines.append(f"\nNo receipts found for seeds {SEEDS}.")
    # 10 Nov audit: a ratio is only a one-factor comparison if arm and comparator were scored on the same panels with the same objective.
    # Joined on the receipts' own fields, not on file names.
    def fields(r):
        st = [x["settings"] for x in r["stages"] if "settings" in x]
        st = st[-1] if st else {}
        return dict(panels=("corrected" if "keyed by split" in st.get("panels", "") else "original (pre-fix)"), panel_sha256=st.get("panel_sha256"),
                    stream=st.get("stream", "blocks"), loss=st.get("loss", "answer"))
    warn = []
    for arm in ARMS:
        for seed in SEEDS:
            a, t = load(arm, seed), load(twin_of(arm), seed)
            if a is None or t is None or arm == twin_of(arm):
                continue
            fa, ft = fields(a), fields(t)
            bad = {k: (fa[k], ft[k]) for k in ("panels", "loss") if fa[k] != ft[k]}
            if fa["panel_sha256"] and ft["panel_sha256"] and fa["panel_sha256"] != ft["panel_sha256"]:
                bad["panel_sha256"] = "differ"
            if bad:
                warn.append(f"- `{arm}` against `{twin_of(arm)}`, seed {seed}: {bad}")
    lines += ["", "## Comparator checks (arm and comparator must share panels and objective)", ""] + (warn or ["No mismatch among the receipts summarized here."])
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
