"""Read finished receipts and produce the combined development report."""
from pathlib import Path

from campaign_common import ROOT,RUN,SEEDS,read,sha,write

BIRTHS=("aligned","shuffled","random","stock")
ARMS=("answer","shuffled","worked")


def progress():
    out={}
    for name in ("encoder","mod97","attractor"):
        root=RUN/name
        out[name]={"complete":sum(1 for _ in root.glob("*/COMPLETE.json"))}
    out["encoder"]["local_histories"]=sum(1 for _ in (RUN/"encoder").glob("*/*-consistent.json"))+sum(
                                            1 for _ in (RUN/"encoder").glob("*/*-mixed.json"))
    out["mod97"]["completed_branches"]=sum(1 for _ in (RUN/"mod97").glob("*/*/*-final.pt"))+sum(
                                          1 for _ in (RUN/"mod97").glob("*/*-final.pt"))
    total=0
    for p in (RUN/"attractor").glob("*.log"):
        steps=[int(line.split()[-1]) for line in p.read_text().splitlines()
               if len(line.split())==4 and line.split()[-1].isdigit()]
        total+=max(steps,default=0)
    out["attractor"]["groups"]=total
    print(out)


def encoder():
    root=RUN/"encoder"
    rows={(seed,arm,law,history):read(root/f"{seed}-{arm}"/f"{law}-{history}.json")
          for seed in SEEDS for arm in BIRTHS for law in range(2) for history in ("consistent","mixed")}
    assert len(rows)==48
    lives=[read(root/f"{seed}-{arm}"/"COMPLETE.json") for seed in SEEDS for arm in BIRTHS]
    dev=[read(root/f"birth-{seed}"/"COMPLETE.json") for seed in SEEDS]
    assert sum(x["episodes"] for x in lives)==196608 and sum(x["targets"] for x in dev)==6291456
    table=[]
    all_a=0
    gates={}
    for arm in BIRTHS:
        consistent=[r for (s,a,l,h),r in rows.items() if a==arm and h=="consistent"]
        mixed=[r for (s,a,l,h),r in rows.items() if a==arm and h=="mixed"]
        cells=[sum(r["taught_A"]["exact"] for r in consistent),
               sum(r["B"][0]["exact"] for r in consistent),
               sum(r["B"][-1]["exact"] for r in consistent),
               sum(r["B"][-1]["first"] for r in consistent),
               sum(r["B"][-1]["eos"] for r in consistent)]
        all_a+=sum(r["taught_A"]["exact"]>=52 for r in consistent)
        table.append("| "+arm+" | "+" | ".join(f"{n}/384" for n in cells)+" | "+
                     " | ".join(f"{sum(r['B_AUC'] for r in h)/6:.2%}" for h in (consistent,mixed))+" |")
    for seed in SEEDS:
        for law in range(2):
            a=rows[seed,"aligned",law,"consistent"]
            gates[f"{seed}-{law}"]={"taught":a["taught_A"]["exact"],
                  "over_controls":{arm:a["B_AUC"]-rows[seed,arm,law,"consistent"]["B_AUC"] for arm in BIRTHS[1:]},
                  "prior_law_effect":a["B_AUC"]-rows[seed,"aligned",law,"mixed"]["B_AUC"]}
    passed=all(r["taught"]>=52 and min(r["over_controls"].values())>=.05 and r["prior_law_effect"]>=.05
               for r in gates.values())
    meta=dict(histories=48,local_episodes=196608,local_symbols=10813440,development_targets=6291456,
              cpu_seconds=sum(x["cpu_seconds"] for x in lives+dev),max_worker_rss_mib=max(x["rss_mib"] for x in lives+dev),
              acquisition_passes=all_a,forward_transfer_passed=passed,aligned_gates=gates)
    write(root/"SUMMARY.json",meta)
    return meta,"\n".join(table)


def mod97():
    root=RUN/"mod97"
    decision=read(root/"SEALED_DECISION.json")
    audit=read(root/"PATH_AND_SHORTCUT_AUDIT.json")
    rows=[]
    table=[]
    counts=[]
    for seed in SEEDS:
        base=root/str(seed)
        complete=read(base/"COMPLETE.json")
        at=complete["fork_step"]
        warm=read(base/f"warm-{at}.json")
        state=complete["status"]
        paths=base/"early-branches" if state=="EARLY_TRANSFER" else base
        branches=state in ("EARLY_TRANSFER","BRANCHES_COMPLETE")
        if branches:
            scores={a:read(paths/f"sealed-{a}.json") for a in ("fork","none","periodic","smooth")}
            held=[f"{scores[a]['exact']}/{scores[a]['n']} ({scores[a]['exact']/scores[a]['n']:.1%})" for a in scores]
            delta=scores["periodic"]["exact"]-scores["none"]["exact"]
        else:
            scores={}
            held=["—"]*4
            delta=None
        extra=read(paths/"COMPLETE.json") if paths!=base else {}
        table.append("| "+" | ".join([str(seed),f"{at:,}",f"{warm['taught']['exact']}/{warm['taught']['n']}",
                      f"{warm['monitor']['exact']/warm['monitor']['n']:.1%}",*held])+" |")
        counts.append((complete,extra))
        rows.append(dict(seed=seed,status=state,warm=warm,periodic_minus_none=delta,
                         sealed={a:{k:v for k,v in s.items() if k!="rows"} for a,s in scores.items()},
                         shortcut_scores=audit[str(seed)]["shortcut_scores"]))
    steps=sum(r["warmup_updates"]+r["branch_updates"]+e.get("updates",0) for r,e in counts)
    cpu=sum(r["cpu_seconds"]+r.get("prior_cpu_seconds",0)+e.get("cpu_seconds",0) for r,e in counts)
    meta=dict(training_updates=steps,training_records=steps*64,symbols=steps*64*9,
         cpu_seconds=cpu,delayed_rise_decision=decision,results=rows)
    if decision["launch_sham"]:
        sham={}
        for seed in SEEDS:
            r=read(root/str(seed)/"sealed-sham.json")
            sham[str(seed)]={k:v for k,v in r["score"].items() if k!="rows"}
        meta["sham"]=sham
    write(root/"SUMMARY.json",meta)
    return meta,"\n".join(table)


def attractor():
    root=RUN/"attractor"
    decision=read(root/"DECISION.json")
    table=[]
    complete=[]
    result={}
    for mode in ("ring","rewired"):
        for arm in ARMS:
            key=f"{mode}-{arm}"
            dirs=[root/f"{seed}-{key}" for seed in SEEDS]
            done=[read(d/"COMPLETE.json") for d in dirs]
            assert all(r["groups"]==16384 for r in done)
            complete.extend(done)
            rows=[read(d/"A-16384.json") for d in dirs]
            result[key]=[dict(taught=r["taught"]["exact"],held=r["held"]["exact"],
                              both_orders=r["held"]["both_orders"],eos=r["held"]["eos"],
                              local_supports=r["dynamics"]["distinct_supports"]) for r in rows]
            cells=[" · ".join(str(r[field]) for r in result[key]) for field in ("taught","held","eos","local_supports")]
            table.append("| "+" | ".join([mode,arm,*cells])+" |")
    groups=sum(r["groups"] for r in complete)
    assert groups==294912
    meta=dict(groups=groups,symbols=groups*58,
              cpu_seconds=sum(r["cpu_seconds"]+r["prior_cpu_seconds"] for r in complete),
              max_worker_rss_mib=max(r["rss_mib"] for r in complete),
              selected=decision["selected"],results=result)
    if decision["selected"]:
        mode,arm=decision["selected"]
        meta["conditional"]={str(seed):read(root/f"{seed}-{mode}-{arm}"/"CONDITIONAL.json") for seed in SEEDS}
    write(root/"SUMMARY.json",meta)
    return meta,"\n".join(table)


def report():
    e,et=encoder()
    m,mt=mod97()
    a,at=attractor()
    closed=sum(r["status"]=="EARLY_TRANSFER" for r in m["results"])
    highest=max(r["held"] for rs in a["results"].values() for r in rs)
    path_audit=read(RUN/"mod97"/"PATH_AND_SHORTCUT_AUDIT.json")
    branches=[r["sealed"] for r in m["results"] if r["sealed"]]
    no_shrink=[r["none"]["exact"]/r["none"]["n"] for r in branches]
    lower=sum(r["periodic"]["exact"]<r["none"]["exact"] and
              r["smooth"]["exact"]<r["none"]["exact"] for r in branches)
    pathways=[r["none"] for r in path_audit.values() if "none" in r]
    def range_for(mode):
        vals=[r[mode]["exact"]/r[mode]["n"] for r in pathways]
        return f"{min(vals):.1%}–{max(vals):.1%}" if vals else "—"
    modulation=(f"The no-shrinkage arms reached **{min(no_shrink):.1%}–{max(no_shrink):.1%}** exact "
                f"on the sealed split. Both shrinkage schedules scored below no shrinkage in {lower}/{len(branches)} "
                f"seeds. In the no-shrinkage arms, the shared path alone reached {range_for('shared_only')}; "
                f"the associative residual alone reached {range_for('association_only')}." if branches else
                "The sealed comparison stayed closed.")
    max_b=max((r["B"][-1]["exact"] for path in (RUN/"encoder").glob("*/*-consistent.json")
               for r in [read(path)]),default=0)
    manifest={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob("*.json")
              if p.name!="RECEIPT_HASHES.json"}
    write(RUN/"RECEIPT_HASHES.json",manifest)
    (ROOT/"OFFICIAL_REPORT.md").write_text(f"""# MiniFly official results for the three remaining experiments

**November 6, 2026 · Development campaigns**

The three protocols tested frozen encoder initialization, downscaling during rest, and attractor topology with worked counting. Results belong to separate learners and tasks. [Locked design](../minifly-whole-repo-audit/MINIFLY_THREE_REMAINING_EXPERIMENTS_DESIGN_20261106.md) · [Engineering cycles](CYCLES.md)

## Frozen pretrained encoder

All 48 local histories finished. They consumed 196,608 graph episodes and 10,813,440 delivered symbols. The six trained encoder births consumed 6,291,456 next-byte targets. All frozen encoder tensor checks passed during life.

| Encoder | Literal taught A | Held B before teaching | Held B after 2,048 | Correct first byte at B end | EOS at B end | Consistent A B area | Mixed A B area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{et}

Counts pool the six consistent histories per encoder. The best final B score of a single consistent history was {max_b}/64. **{e['acquisition_passes']}/24** consistent histories reached the literal taught floor of 52/64. The forward-transfer gate {'passed' if e['forward_transfer_passed'] else 'failed'}. The local acquisition scores govern interpretation of the B curves. [Encoder summary](official/encoder/SUMMARY.json) · [Encoder lock](official/encoder/LOCK.json)

## Mod 97 downscaling

{closed}/3 seeds exceeded the 50% early-transfer ceiling at the common plateau. Early-transfer seeds remained in the matched shrinkage comparison and were excluded from the delayed-rise test. Their early warmup checkpoint was recovered by deterministic continuation and checked against the original taught and monitoring receipts. All sealed scores were generated after the branch checkpoints were frozen.

| Seed | Fork update | Taught at fork | Monitoring held at fork | Sealed at fork | No shrinkage | Periodic | Smooth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{mt}

{modulation}

Taught and held operands were split by unordered pair, so the swapped form of a held pair was also withheld. Exact accuracy requires two correct bytes and native EOS. The pathway diagnostic temporarily muted either the shared path or the associative residual. Its scores, raw native sealed outputs and shortcut ceilings are in the receipts. The conditional shuffled-teaching comparison was tied to a positive delayed-rise result. Interpret the early transfer as held behavior on this finite task with its own evidence boundary. [Mod 97 summary](official/mod97/SUMMARY.json) · [Path and shortcut audit](official/mod97/PATH_AND_SHORTCUT_AUDIT.json)

## Attractor and worked counting

All 18 histories completed 16,384 groups. Each group contained a count, addition and auxiliary record. The roster processed 294,912 groups and 17,104,896 delivered symbols. The behavioral gate required at least 60/75 taught and 20/25 held in each seed.

| Topology | Teaching | Taught exact out of 75 | Held exact out of 25 | Held native EOS out of 25 | Stable localized supports |
| --- | --- | --- | --- | --- | --- |
{at}

The highest held score was **{highest}/25**. Values are in seed order {', '.join(map(str,SEEDS))}. The count-only control and B continuation were conditional on the behavioral gate. Stable localized states and the tally-pulse and readout-muting diagnostics describe the mechanism separately from exact addition. [Attractor summary](official/attractor/SUMMARY.json) · [Mechanism and gate receipt](official/attractor/DECISION.json)

## Accounting

Aggregate main worker CPU was {e['cpu_seconds']/60:.1f} minutes for pretraining and local life, {m['cpu_seconds']/60:.1f} for modular training and branch monitoring, and {a['cpu_seconds']/60:.1f} for attractor training. Sealed postprocessing and separate diagnostics have their own receipts. [Receipt hashes](official/RECEIPT_HASHES.json) · [Reproduction commands](REPRODUCE_OFFICIAL.md)
""")
    print(e["acquisition_passes"],closed,highest)


if __name__=="__main__":
    import sys
    progress() if len(sys.argv)>1 and sys.argv[1]=="progress" else report()
