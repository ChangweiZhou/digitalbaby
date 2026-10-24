# Audit — A11-A reduced-scope modular redesign v2

**Date:** 2026-09-25
**Reviewed:** `A11_MODULAR_A_REDESIGN_V2.md` (24 Sep, 22:31), with its predecessor `A11_ONE_HOUR_MODULAR_REDESIGN_DRAFT.md`
**Read against:** `minifly_A11_v3_8_…md` §§25–33, 40–47; errata v3.8.1 (event contract), v3.8.2 (endpoint timing), v3.8.3 (L>1 boundary); `CORE_RECOVERY_FB1_FOLLOWUP/v1/arms/*.json.gz` receipts; `CORE_RECOVERY_FB1/v3_2/B_PARENT_LOCK.json` and worlds
**Nature:** design audit only. No learner was run and no A11 file was changed.
**Line:** P (PN-cue line: B / FB1 / HASH8). Nothing here touches BYTE-CORE.

---

## Verdict

v2 is careful where it is careful: pairing, nesting, the whole-life rule, sealed scores and status labels are right. But the scope reduction rests on a cost premise that the project's own receipts contradict. Running the whole nine-cell A11-A grid on the same engine costs about **19–23 CPU-hours (≈5–6 h wall on 4 workers)**. That is inside A11's own §40 caps, and no single life comes near one hour. The three cells kept also drop the one cell that anchors the load axis (K1/L1), and one of the two primary endpoints is probably at floor in M3 because of the slow time constant.

Recommendation: do not freeze v2 as written. Either run the full grid in whole-life one-hour packets, or keep a reduced scope but (a) add K1/L1, (b) fix the Ω_s omission, and (c) settle the D-return floor question before science. Details follow.

---

## Evidence: what a life actually costs

FB1 follow-up receipts use the same `v88.evaluate_life` path (`WorkstreamFB1_Followup_v1/fu_engine.py:30`) and the same FPF world grammar. An FB1 `k8` world has 14,206 events and 13,824 teaches, exactly v3.8.1's `8 × 1,770 + 5 + 41`.

| FB1 receipts (observer on) | Lives | Median s | Max s | Peak RSS |
|---|---:|---:|---:|---:|
| HASH8, canonical world (1,728 teaches) | 96 | 3.9 | 5.0 | 0.37 GB |
| HASH8, K8 world (13,824 teaches) | 128 | 31.7 | 37.7 | 0.74 GB |
| module-B G8, K8 world | 64 | 34.6 | 38.7 | 1.20 GB |

Cost is linear in teaches: 2.26 ms per teach on canonical worlds, 2.29 ms on K8 worlds. One canonical world of teaching (1,728 teaches) is the natural unit. Per seed and anchor, the nine-cell grid needs 256 such units. This matches the "256 cohort applications per seed and anchor" in `A11_A_NARROW_SOURCE_QUESTION`.

Planning estimate at 2.29 ms/teach, with matched rests at 0.1–0.6 ms each (not measured; a rest is one kernel step with no KC input):

| Cell | Teaching units | Rests | Est. per life |
|---|---:|---:|---:|
| K1/L1 | 2 | 25,920 | 10–24 s |
| K8/L1 (M1) | 9 | 13,824 | 37–44 s |
| K16/L1 (M2) | 17 | 0 | ≈67 s |
| K8/L4 (M3) | 30 | 60,480 | 2.1–2.6 min |
| K8/L16 (M4) | 114 | 247,104 | 8–10 min |

| Scope, 32 seeds × 2 anchors | CPU-h | Wall, 4 workers |
|---|---:|---:|
| v2: M1–M3 | 4.1–4.7 | ≈1.0–1.2 h |
| v2 + K1/L1 | 4.3–5.1 | ≈1.1–1.3 h |
| All L1 and L4 cells (drop K8/L16 only) | 10.4–12.5 | ≈2.6–3.1 h |
| Full nine-cell grid | 18.9–23.2 | ≈4.7–5.8 h |

Caveats: the FB1 numbers include FB1's observer; the A11 `B` anchor was not timed directly. At BC1's FULL_151 rate (about 3.8 ms per pair-teach, including live byte encoding) the full grid would be about 31–35 CPU-h. That is still under §40's 72 CPU-h. v2's own anchor × cell timing qualification should confirm the estimate. It will not overturn its order of magnitude.

---

## Findings (most consequential first)

### F1. The scope reduction is not needed for cost or for one-hour packets

- v2 §1 cuts to 18.4% of planned rows. In teaching work that is 56 of 256 units (21.9%). It saves about 15 CPU-hours, roughly 4 wall-hours on 4 workers.
- What it gives up: A11-A's load slope over K = 1…16 (§32.1), its lifetime slope (§32.2), its load × lifetime interaction (§32.3), and six of the nine absolute cells that §§45–47 would use as preservation constraints.
- The longest life in the grid (K8/L16) is about 8–10 min, so every packet can hold whole lives. The mid-life checkpoint question in §5 and the "a packet may exceed one hour" SLO caveat never arise.
- If the owner's aim for "one-hour experiments" is visibility and failure containment, the full grid runs as 5–6 one-hour packets of whole lives, with no redesign of the science.

### F2. Without K1/L1 the load axis has no anchor

- Δ_K compares K16 with K8, which is only the top octave of load. If competence falls steeply from K1 to K8 and then flattens, v2 would report a small Δ_K while the learner has already lost most of its unloaded competence.
- A later minimal core (§§46–47) would also carry no preservation constraint at low load. A reduced child could keep K8–K16 behaviour, lose single-cohort competence, and pass.
- K1/L1 costs 2 teaching units per life (about 4% more than v2). If any reduced scope is kept, add it. It uses the same D and A, with slots 1–15 as matched rests.

### F3. D-return at L4 is probably at floor, set by a known constant

B's kernel (`B_PARENT_LOCK.json`, the same constants as FULL_151): fast τ = 3,375.6 s, slow τ = 12,319,263 s (142.6 d). D is dormant from its post-canonical reference at 2,298,240 s:

| Return at | Dormancy | Passive slow factor e^(−t/τ) |
|---|---:|---:|
| L1 | 125.6 d | 0.41 |
| L4 | 502.4 d | 0.030 |
| L16 | 2,009.6 d | 7.6 × 10⁻⁷ |

The fast state is gone at every return. So Δ_L on first-use D return mostly re-measures the slow time constant. At L4 both anchors may sit at the same floor, which would make the paired anchor difference uninformative by construction.

- Before science, run a deterministic qualification-only check: a technical seed's D prelude, then pure matched rest to 502.4 d, then the native D-return score for each anchor.
- If the predicted D margin is at floor, move D-return Δ_L and the M3 D-return anchor difference out of the primary families now, before any science.
- The same arithmetic shows D-return at L16 is structurally zero. Any future M4 should drop or redefine that endpoint.
- A_PERSIST_SET ages the same way. At L4 pre-return, A was last taught about 385 d earlier (slow factor ≈ 0.07), and its 12 facts are 12.5% of the 96-fact K8 score. Δ_L on current usefulness is therefore mostly the difference in fresh-cohort learning between episode 4 and episode 1, plus A's decay.

### F4. Ω_s omits the slot-0 clock donors for episodes 2–4

- v2 §2 lists `A` plus `E[e,r]` for r = 1..15 only.
- v3.8.1 §7 requires 16 fresh clock-donor worlds per later episode, and in episodes e > 1 slot 0 is an inactive donor. M3 therefore needs E[2..4, 0], which Ω_s does not contain.
- A runner built from Ω_s as written must either draw a world, violating §2's "no mutable RNG draw", or reuse A's world. Reusing A would replay A-identity zero-time rows (including `change_truth`) as inactive rows and risk corrupting A_PERSIST_SET truth.
- Fix: enumerate r = 0..15 for e ≥ 2. Add a negative fixture: a missing E[e,0] must be rejected, and so must an A-world reused as the slot-0 donor.

### F5. Make the shared-target score primary for Δ_K now

- v2 correctly notes that K16 changes the composition and denominator of current usefulness (12K facts). It then keeps the composition-confounded native score as primary and demotes the clean contrast, A + F[1,1..7], to a diagnostic.
- v2 is a new program and nothing has been observed, so it can pre-register the shared-target score as the primary current-usefulness estimand for Δ_K and report the native score descriptively.
- First-use D return is already a common-target contrast.

### F6. A11's interpretation patterns need contrasts v2 does not include

- §33's "constant shift", "load-amplified advantage" and "lifetime-decaying advantage" are statements about anchor × contrast differences: Δ_anchor(K16) − Δ_anchor(K8), and Δ_anchor(L4) − Δ_anchor(L1).
- v2's four families include neither. Either add the 4 difference-in-differences contrasts ({K, L} × 2 endpoints) to a family, or state that v2 makes no §33-type interpretation.

### F7. n has no target; the only sourced practical reference is 5 pp

- §5 says to choose n from "a sourced precision requirement", but no requirement exists in v2 or in A11 §29.
- The project's existing practical reference for ordinary behavioural endpoints is 5 pp (`CLAUDE_ACTION_BRIEF.md`).
- Proposal: target a simultaneous half-width of 2.5 pp for the anchor-difference family. Compute n from qualification-only SDs (technical seeds, or FB1 historical inputs). Lives cost 40–150 s, so n = 64–128 is affordable, and precision rather than CPU should set n.
- State which downstream decision the three-cell result gates: parent choice (anchor differences matter) or minimal-core preservation (absolute envelopes matter).

### F8. There is no feasibility step before a confirmation-style design

- v2 goes straight from protocol to sealed confirmation, with no small screen.
- Add a qualification-only technical pilot. 8 technical seeds × 4 cells (including K1/L1) × 2 anchors is 64 lives, about 1 CPU-h.
- The pilot gives measured runtime and RSS per anchor × cell × logging mode, the F3 floor check, endpoint SDs for n, and a ceiling check. None of it enters science.

### F9. Smaller specification gaps

- **Birth key inside Ω_s.** A11 §2 says the HASH8 key is learner-birth state, not world data, and never enters B. Sealing it with Ω_s is fine, but label it as birth state. Add the negative control: changing the key leaves every B digest unchanged and changes HASH8 route maps.
- **Negative controls for §6 and receipts.** The streaming-versus-ledger equality needs tamper tests: a single altered, dropped or reordered event must break the digest and be caught by a named check. A receipt with a wrong input hash must be refused. §2's rejection list is good; §6 has no equivalent.
- **C0 relationship.** A11 v3.8 §7 orders C0 before A. v2 is a separate program and should state that it does not wait on C0, and that C0 and C remain unexecuted under v3.8.
- **Checkpoint gate.** Delete the conditional mid-life checkpoint gate from §7, or mark it N/A with the measured life lengths. Keep the whole-life rule.

---

## What v2 gets right (keep)

- §1: it says plainly that scope is reduced by giving up claims, that there is no CPU speedup, and that M1–M3 form one correlated study rather than three trials.
- §2: one master object per seed, deterministic projections, anchor-independent events, the M1/M2 slot identity and the M1/M3 `CORE_END(1)` nesting all follow v3.8.1 and v3.8.3 correctly. The rejection list is concrete.
- §3: Δ_K and Δ_L are named as total contrasts, with no pure-interference or pure-decay reading, and no interaction claim.
- §4: whole-seed resampling. Four 95% families, correctly not called one joint statement. No interim inference, and scores stay sealed.
- §5: whole lives only. Technical failure resumes the identical input and never replaces a world.
- §6: streaming receipts instead of 60 M CSV rows. Narrow status labels, and no nine-cell claims.

---

## Suggested path

1. **Preferred:** keep A11-A's nine-cell science under v3.8.3. Replace only its execution with a whole-life one-hour packet contract (v2 §§5–6 already contain most of it): about 19–23 CPU-h, 5–6 packets on 4 workers.
2. **If the owner still wants less:** run all L1 and L4 cells and defer only K8/L16. This keeps the load slope and the interaction; F3 shows L16's D-return is uninformative anyway. About 10–12.5 CPU-h, 3 packets.
3. **Minimal repair of v2:** add K1/L1, fix Ω_s (F4), pre-register the shared-target Δ_K (F5), settle F3 with the deterministic check, and add the F8 pilot. About 5 CPU-h at n = 32.

In every option, first run the F8 technical pilot. It also discharges v2's runtime qualification.

## Limits of this audit

- The runtime numbers are extrapolated from FB1 receipts (HASH8 and module-B arms, observer on) and from BC1. The A11 `B` anchor and the cost of a matched rest were not measured here.
- F3 uses passive slow decay only. Whether the L4 D margin is actually below the reader threshold depends on the size of the D reference trace, which the proposed deterministic check measures.
- Findings F1–F3 were checked against the files cited above. The v3.8 protocol was read in the A11-A sections only (§§0–3, 25–33, 40–47), not in full.
