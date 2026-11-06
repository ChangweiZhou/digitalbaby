"""Verify the recorded A scores and write the campaign report."""
from hashlib import sha256
from pathlib import Path

from official_worked import ARMS, BLOCKS, DOSES, MODES, OUT, ROOT, read, receipt, write


def verify_and_report():
    report=read(OUT/"SUMMARY.json")
    lock=read(OUT/"LOCK.json")
    assert receipt()==lock["sources"]
    assert report["pairs"]==196608 and report["delivered_symbols"]==24182784
    assert len(report["results"])==48
    indexed={}
    hashes={}
    for row in report["results"]:
        b,m,a,law,dose=(row[k] for k in ("block","mode","arm","law","dose"))
        directory=OUT/f"{b}-{m}-{a}"
        raw=read(directory/f"law{law}-{dose}.json")["raw"]
        assert len(raw)==64
        assert sum(r["answers"][0]==r["target"] for r in raw)==row["taught"]
        assert sum(r["answers"][1:]==[r["target"]]*2 for r in raw)==row["held_both"]
        assert sum(o[-1]==256 for r in raw for o in r["answers"])==row["eos"]
        indexed[(b,m,a,law,dose)]=row
        for p in (directory/f"law{law}-{dose}.json",directory/f"law{law}-{dose}.pt"):
            hashes[str(p.relative_to(OUT))]=sha256(p.read_bytes()).hexdigest()
    assert len(indexed)==48
    write(OUT/"ARTIFACT_HASHES.json",hashes)
    scores=[]
    comparisons=[]
    for mode in MODES:
        for arm in ARMS:
            totals=[]
            detail=[]
            for dose in DOSES:
                rows=[indexed[(b,mode,arm,l,dose)] for b in BLOCKS for l in (0,1)]
                totals.extend(f"{sum(r[k] for r in rows)}/256" for k in ("taught","held_both"))
                if dose==8192:
                    detail.extend(" · ".join(str(r[k]) for r in rows) for k in ("taught","held_both"))
            scores.append(f"| {mode} | {arm} | "+" | ".join(totals+detail)+" |")

    def delta(mode,arm,base_mode,base_arm):
        return [indexed[(b,mode,arm,l,8192)]["held_both"]-
                indexed[(b,base_mode,base_arm,l,8192)]["held_both"] for b in BLOCKS for l in (0,1)]
    for arm in ARMS:
        d=delta("recurrent",arm,"window",arm)
        comparisons.append((f"Recurrent minus ordered window with {arm} teaching",d))
    for mode in MODES:
        for arm in ("answer","shuffled"):
            d=delta(mode,"worked",mode,arm)
            comparisons.append((f"Worked minus {arm} with {mode} model",d))
    contrasts="\n".join(f"| {name} | {' · '.join(f'{v:+d}' for v in d)} |" for name,d in comparisons)
    scored="\n".join(scores)
    final=[r for r in report["results"] if r["dose"]==8192]
    best=max(r["held_both"] for r in final)
    n_taught=sum(r["taught"]>=52 for r in final)
    n_e2=sum(r["held_both"]>=48 for r in final)
    selected=report.get("selected")
    if report["status"]=="COMPLETE_STOPPED_AT_A":
        headline="**The campaign stopped at the presentation-transfer gate.** No model and teaching condition met both thresholds in both blocks under both opcode laws."
        disposition="The locked stopping rule kept model evaluation on unseen graph families, the inconsistent-law control and B continuation out of this run. The exhaustive graph oracle and adaptive baseline were qualified independently of training. The earlier B rehearsal remains an engineering receipt."
    elif report["status"]=="COMPLETE_STOPPED_AFTER_E3":
        headline="**The campaign stopped at the held-graph gate.** Some conditions qualified on presentation transfer. None cleared the adaptive comparison on the held graph families."
        disposition="See the E3 raw files for the per-family and per-law scores. The inconsistent-law control and B continuation were conditional on an E3 pass."
    else:
        headline=f"**The campaign reached the conditional stage.** Selected condition {selected}."
        disposition="See the conditional continuation receipt for prior-teaching attribution and delayed transfer."
    eos=sum(r["eos"] for r in final)
    cap=sum(r["cap"] for r in final)
    (ROOT/"OFFICIAL_REPORT.md").write_text(f"""# MiniFly recurrent state and worked examples results

{headline}

All 24 histories completed the locked A dose, totaling **196,608 updates and 24,182,784 delivered symbols**. This is a two-block development comparison of gradient-trained references. It isolates the added effects of carried recurrent state and worked-example supervision on the same raw-byte task.

## Acquisition and presentation transfer

Taught accuracy scores literal training prompts. A held item counts only when both fresh edge-row orders give the exact location byte followed by native EOS. The thresholds are 52/64 taught and 48/64 held pairs per history. All histories continue to 8,192 pairs regardless of their early scores.

| Model | Teaching | Taught at 2,048 | Held at 2,048 | Taught at 8,192 | Held at 8,192 | Final taught by history | Final held by history |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{scored}

The per-history columns use block 93157 laws 0 and 1, then block 104729 laws 0 and 1. At the final checkpoint, **{n_taught}/24** histories cleared the taught floor and **{n_e2}/24** cleared the held floor. The best held score was **{best}/64**. The final probes emitted EOS on {eos:,}/4,608 outputs and hit the cap without EOS on {cap:,}. First-byte counts and full native sequences are in the receipts.

## Matched contrasts

These differences count both-correct held pairs at 8,192, with each condition and control compared under the same birth and law. A practical gain requires eight additional pairs in every history in a contrast.

| Comparison | Difference in held pairs by history |
| --- | --- |
{contrasts}

{disposition}

The highest evidence available from the presentation endpoint is E2 on the prescribed row-order change. A canonical lookup can recover the taught semantic key. Reliable rule reuse would require a separate positive E3 result with the prior-teaching control.

## Reproduction and cost

The main workers used {report['cpu_seconds']/60:.1f} aggregate CPU minutes. The highest recorded single-worker RSS was {report['max_worker_rss_mib']:.0f} MiB. These measurements cover the 12 main workers and their native A probes. The run lock records the feedback-stream and probe hashes, sources and thresholds. The 48 dose receipts and matching checkpoints are listed by SHA-256.

[Locked design](../minifly-whole-repo-audit/MINIFLY_RECURRENT_WORKED_EXAMPLES_EXPERIMENT_DESIGN_20261106.md) · [Three-cycle qualification](CYCLES.md) · [Run lock](official/LOCK.json) · [Main summary](official/SUMMARY.json) · [Receipt and checkpoint hashes](official/ARTIFACT_HASHES.json) · [Main runner](official_worked.py) · [Conditional gates](conditional_worked.py)
""")
    print("verified",len(indexed),"checkpoints",report["status"],"best held",best)


if __name__=="__main__":
    verify_and_report()
