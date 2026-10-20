# Wrapped E3 assay qualification — final report

**Verdict: `WRAPPED_E3_ASSAY_QUALIFIED`.** The fixed 12-byte `## + seven spaces + XY + space` cue permits both the existing FE0 route and the four-visible-byte CONTENT route to receive the same old/new teaching stream and be read by the same route-preserving, label-blind `L/R` output organ. In 64 fresh confirmation worlds, FE0 met the prespecified limited-E3 emitted-choice criterion. CONTENT retained taught cues but did not show a reliable final held-out benefit. These are two baseline assays, **not** an integrated two-route learner or a general-purpose reasoning result.

## Locked question and method

The V9.8 E3 generator supplied six taught P–N edges in both orientations, three withheld edges in both orientations, 144 old teaching records, a simulated day, 72 disjoint-symbol new records, and another simulated day. Only the fixed cue wrapper changed. Every branch began from the same V9.2 checkpoint. W wrote old associations; N saw the identical old bytes, outcomes, labels and time without those writes. Both wrote the new cohort. Each branch used one native Full151 value store. At every scored probe the output organ received both options in balanced order, compared its own route's native values, and emitted `L` or `R` before feedback. No withheld orientation was taught.

The exact/canonical pair-key recitation countermodel is at 50% on the withheld, order-balanced choices; a symbol-class rule could solve this particular relation. The experiment therefore tests limited E3 reuse of old teaching on never-reinforced pairs, not open-ended relational inference.

Technical world 30300 and 16 pilot worlds 30301–30316 were excluded from confirmation. The pilot computed only taught-cue output and passed all five prespecified competence checks for **both** routes. Confirmation used exactly worlds 30401–30464. The scientific unit for intervals was the world, not an individual option presentation.

## Confirmation results

| Route / endpoint | W | N | Paired W−N, percentage points (95% world-level CI) |
|---|---:|---:|---:|
| FE0, withheld pairs at final probe — **primary** | 67.7% | 50.5% | **+17.2 [+9.8, +24.6]** |
| FE0, withheld pairs at old-teaching end | 69.5% | 47.9% | +21.6 [+12.2, +31.0] |
| CONTENT, withheld pairs at final probe — descriptive | 45.3% | 46.4% | −1.0 [−8.2, +6.1] |
| CONTENT, withheld pairs at old-teaching end — descriptive | 51.0% | 42.7% | +8.3 [+0.2, +16.5] |

The primary FE0 interval has a positive lower bound and FE0 W exceeds the frozen 60% absolute criterion. Both routes also passed the confirmation trained-cue guards. At the final probe, FE0 W scored 84.9% on old taught pairs versus N's 49.7%; CONTENT W scored 97.7% versus N's 51.3%. New-cohort taught-choice scores at the new-end checkpoint were FE0 W 83.9%, FE0 N 89.3%, CONTENT W 99.5%, and CONTENT N 100%. The CONTENT no-write held-out baseline was not assumed to equal exactly 50%; its final observed value was 46.4%.

CONTENT's final interval includes modest positive and negative effects. Its descriptive result does **not** prove zero transfer, nor does the small old-end contrast establish lasting transfer. The present exact four-visible-byte address is an effective taught-key bridge, while the FE0 route remains the qualified, similarity-preserving probe for this limited E3 task. Neither result shows that one unchanged core simultaneously has CONTENT-level retention and FE0-level E3 reuse.

## Validation and limits

The pre-lock test checked source-fixture parity with the independent auditor, route-preserving clones, CONTENT's `##XY` query window, FE0 parity with the older choice organ, and the pilot's held-out embargo. The full technical life checked 216 training rows and 336 output rows, matched clocks and predictor states, 216 W versus 72 N native writes per route, read-only probes, and agreement of native values with a separate full-model query path. The independent final audit reconstructed all pilot and confirmation rosters, output scoring and world-level reductions without replaying a trajectory. It rejected eight deliberate receipt corruptions, including changed teacher, route, write flag and dropped probe. The audit verdict is `pass: true`.

All 64 confirmation worlds completed with no scientific-failure receipt. Their summed learner runtime was 432.1 seconds; stored results occupy about 16 MB, below the frozen caps. The source lock digest is `03bf6039179796e74d6d37413d3d5e9a89540e97ef772b794660cb8e2649612c`.

The fixed wrapper keeps each question's two symbols in their original byte positions while adding two constant visible bytes. It is an engineered task format, not naturally occurring text. The choice organ receives two supplied alternatives; the learner does not generate candidates or choose when to answer. Teaching still uses a supplied binary valence bit. The old CONTENT E1/persistence evidence was not rerun or enlarged here. Future core comparisons must give baseline and candidate the same input route(s), output organ, teacher budget, timing and world roster; separately successful CONTENT-E1 and FE0-E3 assays cannot be counted as one integrated mechanism.

Primary machine-readable evidence: [SOURCE_LOCK.json](SOURCE_LOCK.json), [TECHNICAL_AUDIT.json](TECHNICAL_AUDIT.json), [PILOT_AUDIT.json](PILOT_AUDIT.json), [SUMMARY.json](SUMMARY.json), [FINAL_AUDIT.json](FINAL_AUDIT.json). Per-world pilot and confirmation receipts are in `pilot/` and `worlds/`.
