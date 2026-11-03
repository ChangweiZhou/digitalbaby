from pathlib import Path
import csv,json,shutil,hashlib
root=Path(__file__).resolve().parent.parent
res=root/'v42_results'
summary={(x['arm'],x['metric']):float(x['mean']) for x in csv.DictReader((res/'SUMMARY.csv').open())}
records=[json.loads(p.read_text())['data'] for p in sorted((res/'completed').glob('*.json'))]
assert len(records)==4 and all(r['training_updates']==r['new_stream_presentations']==0 for r in records)
policies=['READ-HYBRID','SLOT-ONLY','EXACT-ONLY','ANCHOR-ONLY','SUPER']
def mean(arm,metric):return summary[(arm,metric)]
def ratio(arm,metric):
 return f"{sum(r['endpoints'][arm][metric]['correct'] for r in records)}/{sum(r['endpoints'][arm][metric]['n'] for r in records)}"
rows=[]
for a in policies:
 rows.append(f"| {a} | {mean(a,'whole_bpb'):.4f} | {100*mean(a,'whole_accuracy'):.3f}% | {ratio(a,'A_after_D_seen')} | {ratio(a,'A_after_D_heldout')} | {ratio(a,'revision_after_D_heldout')} |")
text='''**V42 completed: read authority on four fixed V41 histories**

My assessment is that SLOT-ONLY is a useful candidate for the next experiment, with a clear tradeoff. Suppressing exact probationary-trace reads reduces log loss while preserving the measured terminal recall endpoints. It does not dominate the alternatives: next-byte accuracy falls slightly, and retaining similarity-based slot reads still costs ordinary-byte prediction loss.

The work is complete: the recovered recommendation was refined into a fixed five-policy protocol; the interruption-friendly single-file runner was implemented and verified; all four audits ran; and the results, numerical tables, source and checks are included. V42 reuses the four V41 learning histories. It adds **zero training presentations and zero learner updates**. The complete latest chat specification was unavailable, so the policy panel and evaluation details are explicitly documented refinements.

The five policies switch which saved prediction is read. All learning, admission and memory contents remain fixed. READ-HYBRID permits both exact trace and supported slot reads. SLOT-ONLY suppresses exact trace reads; EXACT-ONLY suppresses slot similarity reads; ANCHOR-ONLY suppresses both. SUPER uses the distributed predictor throughout.

**What the results show**

| Policy | Whole loss, bits/byte ↓ | Next-byte accuracy | Old seen recall | Old held-out recall | Revised held-out after D |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Counts pool the four histories for transparency; statistical comparisons use the four seeds as their units. Old recall concerns untouched cohort-A identities after the later cohorts; each identity has two seen renderings and one held-out rendering. Revised held-out retention is a new read-only query on the saved final state, not a new training run.

SLOT-ONLY improves whole-stream loss by **0.05759 bits/byte** against READ-HYBRID; the paired Student 95% interval is [−0.06847, −0.04672]. Its remaining loss gap to SUPER is **+0.04898** [+0.00174, +0.09623]. The exact four-seed sign-reversal reference gives p=0.125 for both contrasts. These are exploratory, pointwise comparisons on reused histories, with only four independent seed units.

Lower log loss does not mean more correct next-byte guesses. SLOT-ONLY makes **92 fewer correct predictions across 61,376 historical byte events**, a decline of 0.150 percentage points from READ-HYBRID. Its paired accuracy interval spans zero. The log-loss and top-1 metrics measure different aspects of prediction, and both should remain visible.

The loss decomposition is precise. Exact traces contribute +0.057337 ordinary-byte and +0.000256 target-byte loss, expressed per whole-stream byte, relative to SUPER. Removing that authority accounts for the improvement. Supported slot similarity contributes another +0.061881 ordinary-byte loss. Exact consolidated anchors contribute a net benefit of about 0.01290 bits/byte. ANCHOR-ONLY therefore has the lowest average loss in this panel, but loses the old held-out advantage: 45/64 correct versus SLOT-ONLY's 63/64. Similarity-based retrieval carries both the observed transfer benefit and a substantial ordinary-byte cost.

**How much retention evidence this adds**

All five policies already achieve 512/512 on E1 held-out accuracy, including SUPER. That endpoint is saturated here. For the same 64 untouched old identities, being correct at E1 and every delayed held-out checkpoint after B, C and D is 63/64 for READ-HYBRID and SLOT-ONLY, versus 43/64 for SUPER, EXACT-ONLY and ANCHOR-ONLY. This conjunction uses the same identities across checkpoints; it is not a conjunction of seen and held-out recall.

Several equalities are structural. Held-out contexts never occur in training, so they cannot use an exact trace; suppressing exact-trace reads cannot change these probes. The old-seen final probes comprise 124 exact-slot reads and four no-candidate reads, with no exact traces. The final revision-retention probes also contain no exact traces: 64 exact-slot seen queries and 32 supported-slot held-out queries. These endpoints support retained consolidated memory, but do not test whether trace suppression preserves an unconsolidated fact.

The final current-seen panel does exercise the intervention: 61 of its 256 queries use exact traces. READ-HYBRID and SLOT-ONLY both remain 256/256 correct; average loss improves from 0.119465 to 0.058660 bits/query. That is useful narrow evidence. It does not replace probing early exact recall before consolidation or after varied intervening experience. Final revised-seen retention is 64/64 for both policies versus 60/64 for SUPER; revised-held-out retention is 32/32 versus 29/32. These small terminal samples should not be treated as fresh confirmation.

**Recommended next decision**

Carry SLOT-ONLY forward as a candidate alongside READ-HYBRID, with ANCHOR-ONLY and SUPER retaining their control roles. The next useful experiment is a small prospective test on fresh short histories, with shared learning enabled and the same fixed read masks. Measure delayed exact recall after a first exposure while facts are still probationary, plus difficult related contexts where slot transfer can help or hurt. Record pre-observation prediction scores so more selective arbitration can later be audited without another expensive learner replay. Keep the write rule unchanged. A new learned gate or a larger policy grid is premature.

A policy must never use target labels, identity classes or the evaluator's target/ordinary distinction to choose authority. Avoid a broad threshold sweep on these four histories. The present results justify a focused question about when a memory should answer; they do not establish general persistent learning, better sample efficiency, lower memory use, or a reduced neuron count. Disabling reads leaves the traces and the inherited 4,096-unit predictor in place.

**Execution and reproducibility**

The completed run evaluated 609 units per history, 2,436 total. A second invocation verified and reused all four completed records with **zero new evaluation units**. Reference preparation made 672 read-only model calls and evaluated 1,344 scalar predictions; original terminal outputs matched exactly and learner state hashes remained unchanged. The audit process used approximately 144.7 MiB peak resident memory. This is evaluation resource use, not a memory saving in the underlying learner.

Verification covers synthetic policy selection, common-stratum reconstruction, held-out context absence, same-identity joins, pooled attribution denominators, real SIGINT and SIGKILL resume, completed-run reuse, identity and corruption rejection, concurrent writers, and archive integrity. The runner keeps atomic checksummed checkpoints and a previous generation, runs sequentially, and enables macOS caffeinate. The same command resumes an interrupted evaluation:

```bash
python3 elm_v42_read_authority.py --install-deps
```

The standalone file embeds the validated observations. It reproduces the audit and analysis, not the original V41 learning. The results ZIP contains per-seed completed records, reference inputs, protocol, numeric tables, plots and provenance. The complete handoff adds development modules, the build helper, preparation and independent audit reports, and execution logs. The original V41 model checkpoints are only required to repeat the earlier reference extraction.

The detailed machine-generated report follows. Its estimates are unchanged from the fixed-source run.

---

'''
text+=(res/'ANALYSIS.md').read_text()
(root/'v42_analysis_report.md').write_text(text)
shutil.copy2(res/'V42_results.png',root/'V42_results.png')
readme=root/'v42_README.md'
s=readme.read_text().replace('Start with `ANALYSIS.md`.','Start with `v42_analysis_report.md` in the complete handoff, or `ANALYSIS.md` in the results ZIP.').replace('`PAIRED_CONTRASTS.csv` and `ATTRIBUTION.csv`','`PAIRED_CONTRASTS.csv`, `ATTRIBUTION.csv` and `POOLED_ATTRIBUTION.csv`')
readme.write_text(s)
print(root/'v42_analysis_report.md')
