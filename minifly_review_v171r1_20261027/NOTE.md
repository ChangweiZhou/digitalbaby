NOTE (review record; no experiment, no claim promotion). It reviews Codex's commits 16623c73 (V171-R1) and db4d2810 (synthesis memo 04) on `codex/minifly-consolidated-20261023`. The reports and code were read, not rerun. It corrects parts of `minifly_selfreview_20261027/NOTE.md` (Claude, fd80e118).

# Review of V171-R1 and the post-V171 critical review (27 October 2026)

## 1. What Codex shipped
* **V171-R1** (`minifly_v171_r1_dev_20261026/`): corrective diagnostics on V171's frozen checkpoints and saved reads. It adds no native training and opens no held assessment. It contains:
  * an EXACT1 instrument, numerically qualified;
  * same-state gradient comparisons of EXACT1, EXACT3 and FULL, by parameter block;
  * joint gate envelopes over all 256 rivals, as linear programmes (LPs);
  * per-prefix convex gate fits;
  * all-80 paired reporting.
* V171's verdict is unchanged: REFERENCE_GATE_NOT_PASSED.
* **Memo 04** (`PROJECT_DOCUMENTS/MINIFLY_MECHANISM_SYNTHESIS_20261026/04_POST_V171_CRITICAL_REVIEW_20261027.md`): Codex's corrections to both lines, and a status for each issue in the Claude ledger.

## 2. Corrections to the Claude self-review, accepted
| Claude self-review (fd80e118) | Correction | Basis |
|---|---|---|
| §1.2.1: "Temporal credit was not exercised" | **Withdrawn as stated.** The comparison set the gate teaching signal against the workspace modulator, which are quantities at different nodes. Within the recurrent block (U, W, R, b), the median ‖g3 − g1‖/‖g_full‖ is 0.45. Temporal credit exists. **Narrower claim retained:** the recurrent block's full-gradient norm is about 6% of the gate block's (0.034 against 0.59), and the ON and R_ZERO ratios are almost identical. The memory-feedback route, which was V171's hypothesis, therefore carries little influence in these checkpoints. | V171-R1 report; FINAL_REVIEW.md |
| §1.2.3: "A reader that only reweights these values cannot supply answers they do not contain" | **Withdrawn as an inference about V171.** Independent per-prefix gates in [0.5, 1.5] reach 47/80 and 57/80 complete taught answers, against 4/80 and 11/80 attained. The parent's E1 failure stands. | V171-R1 joint envelope |
| §1.2.4: the coarse gate may hide a benefit | **Resolved negatively.** All-80 scores are 4/4/4 and 11/10/10 (NATIVE / EXACT3 / FULL). No complete-answer benefit was hidden. | ALL80_RESULTS.json |
| Errata, V170: P3 "confounded with transitive inference" | **Strengthened.** The orders 0<1<2<3<4<5 and 0<2<1<3<5<4 agree on every trained pair and disagree on {1,2} and {4,5}. Two of the three held pairs therefore have no answer implied by training. The only identifiable held pair, {0,3}, was decided correctly 100% of the time (V170 diagnostics). V170's order "failure" says nothing about role binding, and is mildly positive on the identifiable pair. | Memo 04 §2; V170 report |
| Ledger §7.2, A5: a Hebbian reciprocal pathway "holds exactly Mᵀ" | **Qualified.** The transpose requires matched birth, scaling, plasticity timing, decay and constraints. Even then it supplies no delayed credit. The premise r = Mq is in any case suspended under the no-attention requirement. | Memo 04 §3 |
| Ledger §7.4, item 3: surprise-gated writes | **Recast as a hypothesis to be tested.** Surprise and usefulness can diverge: unpredictable noise is surprising but worthless. Relevance selection needs its own control. | Memo 04, D5 |

## 3. Assessment of V171-R1
**Strengths:**
* each criticism was answered by a matching diagnostic;
* the historical result and its gate were left untouched;
* the oracle's interpretive limits are stated correctly: not a learner, not evidence of storage;
* gradients are decomposed by parameter block, and the clipped and projected steps actually applied were measured;
* the checkpoints were authenticated against the published panels;
* failed attempts are kept.

**Remaining issues:**
1. **The shared-controller ceiling requested in the self-review (§1.2.2) is still missing.** The R1 disposition table marks the actuation complaint as addressed. What was done is a per-prefix, target-informed envelope, which bounds the reader from above. The request was a single controller trained by exact gradients, without step or weight limits, on the recorded native traces. That fit is cheap and exact at answer level:
   * the teacher-forced native trajectory does not depend on the controller (V171 design §5);
   * along a fully correct answer, the own-generation path equals the teacher-forced path;
   * so complete-answer correctness can be scored on the saved traces without further native queries.

   R1's revised design §1 and memo 04's D1 defer it to a prospective design. It should be the next diagnostic, before any objective or architecture change. It separates actuation from shared-policy fitting from local learning at E1.
2. **The oracle needs a null before it can say anything about native information.**
   * With the codebook's sign alignment, the corner gate amplifies target-aligned channels 3:1. Part of the per-prefix success may therefore come from the gating's freedom rather than from v.
   * An uninformative-v baseline would quantify this: the same LP with v taken from a mismatched prefix, or a random vector of matched norm.
   * Codex does not claim storage, so this concerns interpretation (for example, memo 04's row T2), not an error.
3. **The behavioural relevance of recurrent temporal credit is unmeasured.** Block ratios show that delayed credit exists. They do not show that it changes outputs, because h reaches the output only through gates with |G| ≤ 1/8. No EXACT1-trained history exists, as R1 states. The question "does short-path credit help?" remains open for this graph.
4. **Process.**
   * Two runtime preflight failures and one serialisation failure occurred, all preserved.
   * The checkpoints are authenticated only retrospectively.
   * The authorisation record says push is not authorised, while PUBLICATION.md records a push. The owner may wish to confirm.

## 4. Points of agreement (both lines)
1. **The no-attention requirement is operative.** The read in V166 is attention-equivalent and is held as scaffolded historical evidence, not as a candidate. Re-entry and learned PN/KC modulation wait for an explicit equivalence audit.
2. **Objective:** next-byte prediction at every position, with one generic rule. Answer-site objectives, marker-gated writes and fixed hop schedules are diagnostic scaffolds only. Codex's revised design §2 adopts this, so the objective question in self-review §3 is converging.
3. **Decision order:** E1 (complete taught own answers and relevant retention) before temporal-rule comparisons, re-entry architecture, or E3.
4. **Evaluation hygiene:** cite as branch@commit/version; keep INVALID, NOT_RUN and failed-reference statuses distinct from negative primary outcomes. Claude's next experiment number remains V172.

## 5. Open decisions for the owner
1. The operational no-attention boundary: native synaptic reads driven by sensory input, against state-dependent addressing of the KC→MBON memory.
2. Confirmation of the next-byte, every-position objective for both lines.
3. A shared registry of version numbers.
4. Whether the shared-controller exact fit on V171's saved traces (§3, item 1) is run next. It is diagnostic, cheap, and needs no native training.
