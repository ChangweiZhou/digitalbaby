# Step 4: panel-B write-lesion diagnosis

**Verdict: `WRITE_ASSOCIATED_RESCUE_DEMONSTRATED`.** The excluded technical world and all 16 new matched schedule worlds completed; the independent audit passed. This is an **E1 causal mechanism diagnostic** for directly taught `0+1=`, not a test of addition or changed presentation.

## Question

Step 3 had an unusually clean contrast: `0+1=` was correct in all eight panel-A worlds but wrong in all eight panel-B worlds. Panel B additionally taught `1+1=` and `2+1=`, which share the late `+1=` suffix and have nearby but nonidentical KC addresses. This run held the panel-B bytes, answer labels, record order, teacher-byte routing, clock, Full151 learners, FE0, and output rule fixed within each world. Every branch received all 48 question and feedback records. Only native writes on the two competing facts were independently disabled. `0+1=` retained its own 12 teaching records and native writes in W/L2/L3/L23; N disabled all native writes.

| Branch | Write `1+1=` | Write `2+1=` | End `0+1=` correct | Mean correct-class margin |
|---|---|---|---:|---:|
| W, all writes | yes | yes | **1/16** | −0.0638 |
| L2 | no | yes | 11/16 | +0.0284 |
| L3 | yes | no | 15/16 | +0.0612 |
| L23 | no | no | **16/16** | +1.3283 |
| N, all writes off | no | no | 0/16 | 0.0000 |

The prespecified primary paired contrast, L23 minus W on `0+1=`'s terminal correct-class margin, was **+1.3921** (two-sided world-level 95% Student-*t* interval **+1.3589 to +1.4252**). It was positive in all 16 worlds. W reproduced the original error in 15/16; L23 recovered 16/16. All three frozen rescue criteria passed. Disabling either competitor alone improved the margin over W in all 16 paired worlds, with descriptive mean gains +0.0922 for L2 and +0.1250 for L3. L2/L3 are secondary, unadjusted observations, and neither recovered both competitors' answers: the omitted fact itself was 0/16 correct. In W, `1+1=` and `2+1=` remained 16/16 correct. This is a tradeoff, not a free performance improvement.

The result establishes that **writes associated with the two competing facts causally suppress `0+1=` under this fixed stream**. It does not establish why at the synapse or coding level. Disabling a fact removes its positive update to its output class **and** its negative updates to the other three classes, reducing both memory load and response competition. The present experiment does not compare equally many writes from a distant cue, so it cannot attribute the loss uniquely to shared `+1=` KC overlap or to fading of the first operand. Claude's “minimum KC distance above 44” proposal would be an unjustified hard gate: Step 3's initial A/B minima were 44/28, but after a preceding record they were 42/26, and neither number is a learned-value threshold. The descriptive 100-fact minimum was 12; those facts were not taught here.

## Audit and next decision

The independent reducer checked the 16 unique new receipts, source and Step 3 provenance hashes, schedule/byte stream hashes, identical delivered bytes, route determined by the raw answer byte, exact native-write counts, pre-feedback output and value arithmetic, all read-only terminal probes, clock and sensory parity. Intentional alterations of a teacher byte, decoded route, probe output, target margin, write count and source digest were all rejected. No failure receipt exists; earlier scientific trajectories were neither changed nor rerun.

The next **small** mechanism comparison should keep four fully taught facts and equal write counts while swapping which class-2/class-3 questions compete with `0+1=` (the already observed A/B factors, plus their two mixed panels) on new matched schedules. That tests whether *which input addresses* compete matters, beyond merely removing writes. In parallel, a label-free candidate front end may be screened read-only for distinct full expressions, changed positions and longer-string counterexamples; no numeric KC-distance cutoff alone should qualify it. A candidate earns a behavioral claim only if native Full151 output improves on a prespecified directly taught crowded panel and a never-taught changed-presentation probe, with matching controls. Expansion to ten digits, learned stopping or 100 directly taught sums would add costs without resolving this mechanism or advancing beyond E1.

Artifacts: [frozen design](DESIGN.md), [runner](output_step4.py), [independent audit](audit.py), [source lock](results/SOURCE_LOCK.json), [technical receipt](results/TECHNICAL.json), [summary](results/SUMMARY.json), [world receipts](results/worlds).
