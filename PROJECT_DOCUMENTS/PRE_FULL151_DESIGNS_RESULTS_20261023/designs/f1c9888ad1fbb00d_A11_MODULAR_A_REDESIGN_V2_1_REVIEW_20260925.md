# Review — A11-A reduced-scope modular redesign v2.1 and the audit response

**Date:** 2026-09-25
**Reviewed:** `A11_MODULAR_A_REDESIGN_V2_1.md`, `A11_MODULAR_V2_AUDIT_RESPONSE_20260925.md`
**Follows:** `A11_MODULAR_A_REDESIGN_V2_AUDIT_20260925.md`
**Nature:** design review only. No learner was run and no A11 file was changed.

## Verdict

v2.1 fixes the one definite defect (the Ω_s donors) and takes the specification points on board. What still stands between v2.1 and a freeze is mostly owner decisions, not design defects. Four small text gaps remain (R1–R4). The response is right on four of its points against my audit and misreads one (C5).

## Corrections to my audit (the response is right)

- **C1. B was timed, and it is faster than HASH8.** I checked `CORE_RECOVERY_FB1/v3_2/receipts`:

  | Arm | Canonical world, median | K8 world, median |
  |---|---:|---:|
  | B | 3.21 s | 26.15 s |
  | HASH8 (follow-up) | 3.92 s | 31.66 s |

  That is about 1.9 ms per teach for B against 2.3 ms for HASH8. The "B at 1.7× slower" allowance in my audit was in the wrong direction. Every cost estimate there only gets smaller.

- **C2. Runtime does not decide scope.** I called the full grid "preferred" on cost grounds. The user asked for smaller scientific questions, and that is a legitimate reason to reduce scope. My point should have been narrower: cost is not a reason to cut, so the cells should be chosen for the downstream decision they serve.

- **C3. The floor at L4 is a risk, not a result.** The decay arithmetic stands (0.414 / 0.0295 / 7.57e-7), but whether scored accuracy is at floor depends on the size of the trace and on the readout threshold.

- **C4. Inspecting pilot endpoints is exploratory.** Timing and RSS are technical. Looking at endpoint SDs or at the floor to revise the protocol is exploratory use, so it needs its own seed bank and must be labelled as exploratory. Agreed.

## One point where the response misreads the audit

- **C5. The n = 128 cost.** The response scales the *full-grid* figure (19–23 CPU-h) to 75.6–92.8 CPU-h at n = 128. My n = 64–128 remark was about v2's cells, although I did not say so explicitly.
  - At v2's scope, n = 128 is 16–19 CPU-h at the HASH8 rate, and less for B. That is well under 72 CPU-h.
  - For the full grid, n ≤ 64 fits under the cap and n = 128 does not. The response is right about that case.
  - Agreed on the substance: n must come from a stated decision and precision rationale. The 2.5 pp half-width was a proposal (BC1's δ/2 convention), not a derivation.

## What v2.1 fixed

- Ω_s now enumerates E[e,r] for r = 0..15 in episodes 2–4. It adds rejection fixtures for a missing E[e,0] and for A's world reused as a slot-0 donor.
- The HASH8 birth key is a separate learner-birth record. A key-only change must leave every B digest unchanged and change the HASH8 route map, which is a proper negative control.
- K1/L1 and native-versus-shared-target Δ_K are now explicit pre-freeze scientific choices, not silent defaults.
- A floor or variance pilot must be a labelled exploratory bank that is excluded from confirmation.
- The program states that it does not wait for C0 and inherits no `A11_COMPLETE_*` status.

## Remaining text gaps (small, but needed before a freeze)

- **R1. Scope status is internally inconsistent.**
  - The header says "design direction accepted" and §1 says "precommit exactly three cells".
  - But line 5 says the full-grid versus reduced-scope choice "is still open", and §7 leaves K1/L1 open.
  - The date is still 2026-09-24, and §7 still reads "This v2 document".
  - State one position. For example: "M1–M3 fixed; K1/L1 pending owner decision X before freeze; full grid not adopted unless the owner reverses the one-hour-question preference."
- **R2. Anchor × contrast claims (§33 patterns) are still unaddressed in the text.** The response accepts the point, but v2.1 §§3–4 neither add the 4 difference-in-differences contrasts nor rule the pattern language out. One sentence would settle it.
- **R3. There is no rule for degenerate endpoints.** v2.1 says the D-return floor is not proven. It still needs a pre-science rule for what an interval reports if an endpoint is near-constant across seeds, since bootstrap simultaneous intervals degenerate there. BC1 used: zero estimated variance → unresolved, unless proved structural. This belongs in the reducer and statistics lock, and it needs no pilot.
- **R4. The streaming-reducer equality in §6 has no negative controls.** §2 now has good rejection fixtures, but §6 has none. A single altered, dropped or reordered event must break the digest and be caught by a named check, and a receipt with a wrong input hash must be refused.

## Decisions that are the owner's, not the designer's

1. Which downstream decision the three-cell envelope gates: parent choice (anchor differences matter) or minimal-core preservation (absolute levels matter, and K1/L1 becomes much more important).
2. The primary current-usefulness estimand for Δ_K: native (composition changes with K) or shared-target A + F[1,1..7] (a clean interference question).
3. The precision target that sets n.
4. Whether to run an exploratory floor/variance pilot before the statistics lock.

Once those four are answered and R1–R4 are patched, the remaining gates are the implementation qualifications §7 already lists.

---

## Addendum — v2.2 check (2026-09-25)

`A11_MODULAR_A_REDESIGN_V2_2.md` was diffed against v2.1.

| Gap | v2.2 | Status |
|---|---|---|
| R1 scope inconsistency | Header, §1 "Intended use" and §7 now say one thing: three cells chosen; K1/L1 and the full grid need separate, pre-locked decisions. Date and version wording updated. | Closed |
| R2 §33 patterns | Out of scope as inferential claims. The difference-in-differences contrasts appear only as labelled descriptive arithmetic, and no §33 label is issued. | Closed |
| R3 degenerate endpoints | `INFERENCE_UNRESOLVED_ZERO_VARIANCE` unless a structural proof exists before science. The near-constant criterion, fallback interval and family handling are deferred to the statistical lock. | Closed at design level |
| R4 streaming tamper tests | `ALTER_EVENT`, `DROP_EVENT`, `REORDER_EVENT` and `WRONG_INPUT_HASH`, each required to fail a named check. | Closed |

The cost correction (three cells at n = 128 ≈ 16–19 CPU-h; 76–93 CPU-h applies to nine cells) is stated accurately. Choosing native current usefulness as primary with shared-target as a predeclared diagnostic is consistent with the stated competence-measurement use.

Optional sharpening for R4 (not blocking):
- Place at least one tamper on rows where an endpoint-only check could miss it: an inactive-slot zero-time row, and the L>1 `A_PERSIST_SET` boundary row.
- Include one `REORDER_EVENT` that swaps the boundary row with the next episode's first block-0 probe. That is the v3.8.3 failure mode.
- Report the result as a caught ratio (4/4 or more).

I see no further design-text defects. What remains is what v2.2 §7 lists: the precision target and n, the interval algorithm, whether to run an exploratory pilot, the resource caps, and the implementation qualifications.
