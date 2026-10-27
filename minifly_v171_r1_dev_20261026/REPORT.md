# V171-R1 — completed corrective diagnostic

**Disposition: complaints materially addressed; V171 remains REFERENCE_GATE_NOT_PASSED.** This redo adds missing EXACT1, joint gate envelopes, exact-gradient conditional gate optimization, all-80 paired reporting and a corrected prospective design. It is a completed E0/E1 diagnostic, not a newly trained native learner or an unseen-addition result. No original result was overwritten and no held assessment was opened.

## What the new tests establish

| Final taught complete own-byte/EOS answers | NATIVE | EXACT3 | FULL |
|---|---:|---:|---:|
| World 711 | 4/80 | 4/80 | 4/80 |
| World 712 | 11/80 | 10/80 | 10/80 |

These are the two original independent worlds, not 160 independent replications. In 711 the correct sets are identical. In 712 each reference loses one NEW answer, gains none, and has the same correct OLD set as NATIVE. Expanding the report from NEW16 to all80 does **not** uncover a complete-answer benefit hidden by the coarse original gate. OLD/NEW and initially-correct OLD retention remain separate in `ALL80_RESULTS.json`. Loss improves, as the original report said; it does not replace own-answer acquisition.

## EXACT1 resolves the temporal-credit criticism more precisely

The added exact-one-transition instrument agrees with an independent forward sensitivity to relative error **3.91e-18**, and six central finite differences to **1.63e-10**. It detaches earlier augmented state; it is not the candidate's random-feedback ON1 rule.

At identical frozen controller states, over **920** new teacher-prefix target sites, median `||g3-g1||/||gfull||` is **2.74%** for the whole parameter vector; the corresponding actual clipped/projected whole-record step difference is **2.79%**. FULL-minus-three whole-gradient median is **1.62%**.

However, the whole norm is dominated by immediate output-gate parameters. Within the recurrent U/W/R/b block, the three-minus-one median is **44.67%**, and FULL-minus-three is **27.73%**. Individual recurrent block medians are U59.53%, W48.59%, R46.63%, b39.93%; G/c delayed fractions are only about 0.006–0.007%.

Therefore **temporal credit is not absent**. Claude's direct-gate signal versus delayed-state signal norms compared different nodes, and the recurrence includes a .75 identity leak. The stronger supported conclusion is that delayed ordinary recurrent learning exists, while its effect is small in the dominant gate-update vector and the additional native-memory feedback route is weak in this diagnostic. R_ZERO gives very similar aggregate ratios, but it changes the forward state; its discarded update may regrow R. It is not a structurally absent-feedback trained control.

These are new non-updating checkpoint probes, not reconstructed historical gradients or a fitted EXACT1 arm. They cannot establish the accuracy EXACT1 would have reached after training.

## Joint actuation envelope, before redesigning the graph

Every LP enforces all 256 rivals simultaneously. We measured frozen, no-write correct-prefix paths at OLD16 and FINAL, with checkpoint outputs authenticated against published panels. At FINAL:

| World | Actual native own answers | Independent gate oracle [.5,1.5] | Cap-aware outer-box oracle |
|---|---:|---:|---:|
| 711 | 4/80 | 47/80 | 42/80 |
| 712 | 11/80 | 57/80 | 51/80 |

All records have decisive witnesses or infeasibility certificates. These are **target-informed, independently chosen gates per prefix**, not attained behavior of one shared controller. The cap-aware box is itself an outer relaxation of shared weights and reachable h. Thus restriction is real, but it cannot explain the entire actual-to-oracle gap. Shared conditioning, optimization, and the learned trajectory remain unresolved. Poor native recall alone does not prove that the values contain no relevant answer information; conversely an oracle that sees the answer cannot prove storage.

Historical pass16 tape analysis separately covers 460 targets and160 records. OLD pass16 precedes NEW learning and teaching changes memory between targets; those aggregates are **not** final own-generation ceilings. `CEILING_RESULTS.json` preserves this distinction.

We also ran **122** bounded exact-gradient convex gate optimizations on frozen reads (90 FINAL NEW target sites and32 fixed OLD sites), removing optimizer-step and controller-weight restrictions while retaining the actual architectural gate box. All obtained certified convex objective gaps. Mean target bits on this subset fall **3.693→1.560** and **3.321→1.352**. This is a per-prefix oracle fit, not shared recurrent-controller training or an E3 gain. Dropping weight limits does not remove tanh's gate-range limit.

The balanced binary output codebook supplies an analytic crosscheck: for each nonzero rival coefficient, `(C_yj-C_rj)*v_j` has sign `C_yj*v_j`. The same target-aligned box corner maximizes every pairwise gap and minimizes cross-entropy. Thus in this particular codebook the joint LP introduces no new conflict among rival constraints. All new frozen and historical LP results agree with that corner check; this is not a generic fact about arbitrary output codebooks.

## Complaint-by-complaint disposition

| Complaint | Resolution |
|---|---|
| Missing EXACT1 / attenuated credit | EXACT1 independently qualified; same-state gradients and actual projected steps measured. Accept missing control; correct the claim that all temporal credit was absent. |
| Actuation ceiling not established | Simultaneous LPs, dual/primal checks, cap-aware bounds and exact-gradient convex oracle completed before any forward redesign. Shared-policy global optimum is **not** claimed. |
| Parent fails E1 | Confirmed poor complete taught recall and initially-correct OLD loss. No E3 stage or claim. Storage, expression and optimization are kept distinct. |
| One answer on NEW16 decides the gate | All80 paired reporting completed. It does not rescue the original result, and does not retroactively change its gate. |
| Answer-only versus next-byte objective | V171 and this correction are explicitly diagnostic. Future generic every-position supervision has its own prospective design and fresh exposure accounting. |
| No-attention ambiguity | No upstream PN/KC/query change or V166/V170 K/V read was launched. The operational boundary remains a design decision, not silently relaxed. |
| Process and numbering | V171-R1 avoids Claude's reserved V172. Parent source/checkpoints are unchanged, new arrays/results saved, relative publication diagnostics used, and prior missing nine presentations remain missing. |

## Accounting and integrity

- Original inherited cost remains **7,711 logical presentations /22,155 arriving reply/EOS targets**. New native training presentations: **0**; new assessment decisions: **0**.
- New checkpoint probes: **960 own-generation queries**, **320 teacher-prefix query records**, **920 native target-prefix sites**. All own choices match the previously published native/EXACT3/FULL panels; FULL teacher-prefix target probabilities also match.
- **5,520** derivative target reuses, **1,920** discarded-clone parameter steps, **24** synthetic loss evaluations including the failed runtime attempt; no persistent learner was updated.
- LPs: 1,840 historical data +1,840 frozen data +748 data in the first serialization failure, and6 synthetic solves across attempts: **4,434 total LP solves**. Convex fits122, with1,473 objective/gradient target uses. Additional analytic-corner verifications and reducer audits are separately charged in `FINAL_AUDIT.json`.
- Successful native diagnostic CPU **49.93 s**; its peak RSS **375.6 MB**. Imports and failed credit attempts add0.197s and0.960s. Historical/frozen LP reducers take about1.26s/1.15s; all80 reduction about0.012s. The first failed LP serialization attempt did not retain CPU. Interactive authoring/shell preparation is not fully metered, so no full-workflow compute-efficiency claim is made.
- Native diagnostic counted8,526 value reads,34,784 encoding sorts,1,280 native clones and at least3.39GB clone traffic. These costs are not hidden behind an online/local label.
- All115 original execution source hashes verified. Checkpoints were not originally hash-receipted; their new hashes and agreement with published panel outputs authenticate this use retrospectively, not as original immutable receipts.
- Two runtime preflight failures occurred before native queries (missing Numba; then NumPy checkpoint compatibility), plus the saved-LP serialization failure. All are preserved with redacted derivative diagnostics. No completed native training trajectory was replayed. Original lost arrays are not regenerated.

## Next decision

The original output-gate reader has both a genuine actuation restriction and a large unresolved shared-policy/optimization gap. It is premature to attribute its failure only to short temporal credit, absent stored information, or a necessary new feedback architecture. Follow `REVISED_DESIGN.md`: qualify a generic E1 learner and its usable output path first, with every-position supervision prospectively accounted, before unseen-addition promotion or a new memory-reentry architecture.
