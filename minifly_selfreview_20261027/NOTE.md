NOTE (review record; no experiment, no claim promotion). It reviews Codex's commit 09bad998 and records the self-review of V163–V170, the memo and the issue ledger. It amends:
* `minifly_memo_credit_horizon_20261027/MEMO.md`;
* `minifly_note_issue_ledger_20261027/NOTE.md`;
* the V164 report.

# Review of Codex's V169/V171, and self-review errata (27 October 2026)

## 1. Codex's commit 09bad998
**What it contains:**
* a V169 design, numbered on Codex's branch (`minifly_v169_design_20261026/`);
* that design's execution, numbered V171 (`minifly_v171_dev_20261026/`).

**The architecture:**
* a 64-unit recurrent workspace reads the native parent's 16 expressed values;
* it reweights them through gates bounded in [0.5, 1.5];
* it is trained by a local three-step feedback circuit.

**Verdict: REFERENCE_GATE_NOT_PASSED.** The exact-three and exact-full-record instruments did not improve complete taught NEW answers in either calibration world. The main comparison was therefore not run.

### 1.1 Strengths
* **Pre-registration:**
  * reference qualification before any comparison of local rules;
  * matched controls (ON1, MIS3, OFF3, CAP-OFF3);
  * a constant-carrier intervention;
  * complete own-answer endpoints.
* **Honest reporting:** the stop rule was obeyed, the endpoint was not moved to likelihood, and exposure accounting is thorough.
* **Feedback design:** the carried feedback is conditioned on the actual native values, consistent with synthesis memo 02 §8.
* **Diagnosis:** the read-only actuation certificate in the saved-response audit.

### 1.2 Substantive issues
1. **Temporal credit was not exercised.**
   * The largest carried signal is 0.012–0.020 (L2), against 1.13–1.46 for the direct signal (SAVED_RESPONSE_AUDIT.json).
   * EXACT3 and EXACT_RECORD agree to 1e-4 bits/target.
   * This follows from the forward design: leaky ¼ integration, R bounded at 1/32, and a further factor of about ¼ in each delayed round.
   * An EXACT1 reference would have shown it before calibration. The experiment is uninformative about short-path credit, its stated question.
2. **The actuation ceiling was not established before calibration.**
   * The gates lie in [0.5, 1.5] by design, effectively in [0.58, 1.42] under the weight bounds. The largest departure realised was 0.124.
   * Only 8 of 90 sites are certified unreachable. For the other 82, joint infeasibility, bounds and optimisation are not separated.
   * The native reads are exogenous to the controller (design §5). A joint ceiling, the exact gradient without step or weight limits on the saved reads, could therefore be computed cheaply. It should precede the forward-graph redesign.
3. **The parent fails E1.**
   * Native taught recall after 16 passes: OLD 4/64 and 3/64; NEW 1/16 and 5/16. In world 712, none of the initially correct OLD answers survives.
   * A reader that only reweights these values cannot supply answers they do not contain.
   * Native storage and readout are binding (compare V153's attribution to the output port). E3 is premature.
4. **Gate resolution.**
   * Strict improvement is required on 16 NEW items per world, so a single answer decides the gate.
   * The likelihood gains of 0.13–0.20 bits/target are invisible to it.
   * A qualifying gate should resolve the expected effect, for example by using all 80 taught items.
5. **Objective.**
   * Supervision is applied at answer sites only.
   * The owner's principle of 27 October makes next-byte prediction the meta-task. Under it, answer-only objectives are admissible only as diagnostics. The two lines now differ, and the owner must reconcile them.
6. **No-attention boundary.**
   * The revised design routes recurrent state into native PN/KC processing, which then reads KC→MBON. That is a state-dependent, content-addressed read.
   * The requirement of 13 October excludes "content-addressed key/value reads … and equivalent fast-weight retrieval". An operational boundary is needed.
   * The revision converges on V170 strategy A (re-entry through the input expansion; two-hop composition learned with one-step credit from 500 presentations). V170 is its most relevant evidence.
7. **Process.**
   * Most of the three cycles were consumed by engineering failures, all disclosed.
   * The arrays for 9 technical presentations are lost.
   * Interactive CPU was not metered.
   * The published tracebacks contain local absolute paths.
   * The number V169 now names two different experiments. **The next Claude experiment is V172.**

## 2. Consequences for the Claude line
### 2.1 No-attention requirement (unchecked until now)
V166's learned query reads an outer-product memory:

r = Σᵢ e_{bᵢ} (k(aᵢ)·q) / 16.

This is unnormalised linear attention over stored episodes, which the requirement of 13 October excludes. The following build on it:
* V168 and V170;
* the memo, §3.3 and §6.1;
* the ledger, A1 and A5, and items 1–3 of §7.4.

**Owner ruling required:** is a state-dependent read of the native KC→MBON memory admissible? Until the ruling, those items are suspended.

### 2.2 Errata
| Source | Statement | Correction |
|---|---|---|
| Memo §3.2; ledger §5 | Spatial credit is "resolved" or "settled" | **Supported under a back-propagated evaluator; E0 open.** The local learner's own B bits after continued learning (3.407, V169) are worse than a back-propagated probe on the raw KC code (3.374). Measured from TOP rather than RAND-H, the gain recovered is 78% (V165), 75% (V167), 55% (V169) and −11% (V163). The alignment cosines are not computed comparably across rows. |
| Ledger §5 | Fully local learner replicated on two pairs: V165, V167, V169 | Fully local on two pairs: V167 and V169 only. V165 uses a back-propagated top. |
| Memo §3.4 | KF-RTRL is "too noisy" | Fails at this budget. Variance and its interaction with AdamW and clipping are plausible causes, not identified ones. |
| Memo §3.4; V164 report, "What it shows" 1 | α ≈ 0 because targets are independent; the closure contributes nothing | Partly numerical: the fixed ε dominated the statistics (`posthoc/`). The structural argument stands. |
| Memo §5.1, §6.1 | FE-PERM with fade 0.807 since the CLS round; the fade was "retained" | The CLS round used FE-PERM with fade 0.95. Fade 0.807 (P81) was **restored** from V153 onwards. The 31% overlap and 100% order evidence were measured at 0.95. |
| Memo §5.3 | Order and temporal credit are "the same problem" | Coupled on the recurrent route only. V170's order failure persisted under exact gradients, although that test was confounded with transitive inference. |
| Memo §6.1; ledger A2 | Long-range information "must" go through the associative memory | Alternatives include a larger learned recurrent state. The memory route is itself restricted (§2.1). |
| Memo §6.2 | Interference is the expected failure of the learned layer | V169 found no forgetting across two related authors. |
| Ledger A6 | BPTT required more than 9,600 presentations | 9,600 sufficed at lr 3e-3 (V168 E1); 24,000 did not at lr 1e-3 (V168 D). The A-versus-LOOP comparison confounds 1,282 against about 47,000 parameters. |
| V170 design | Strategy B: eligibility traces × fixed random feedback | Defective in principle: fixed feedback cannot carry Mᵀd (synthesis memo 02 §8). |

**Process deviations, already disclosed and recorded here:**
* the AUT-02 departure in V163;
* reference grids in V168 and V170 extended after reference outcomes, but before any learner code;
* V170's guard G1, which measured confidence rather than retrieval.

### 2.3 Convergence
Both lines now place one question ahead of refining the credit rule: how memory content re-enters computation (V170 strategy A; V171's revision).

## 3. Decisions for the owner
1. The no-attention boundary for state-dependent reads of native memory.
2. One objective for both lines: answer-site supervision or the next-byte meta-task.
3. A shared numbering registry across the branches.

## Addendum (27 October 2026): corrections after V171-R1
`minifly_review_v171r1_20261027/NOTE.md` §2 corrects this note:
* **§1.2.1 is withdrawn as stated.** Temporal credit exists in the recurrent block. The narrower claim retained is that the memory-feedback route is weak.
* **§1.2.3's inference is withdrawn.** Per-prefix oracle gates reach 47/80 and 57/80. The parent's E1 failure stands.
* **§1.2.4 is resolved negatively.** All-80 scoring shows no hidden benefit.
* **The V170 order erratum is strengthened.** Two of the three held pairs are unidentifiable from training.
