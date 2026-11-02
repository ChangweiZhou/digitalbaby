# V179 continuous-life local learner: three-cycle result

**Verdict: LIFETIME_DEV_GAPS. The exercise completed, but the full advancement criteria did not pass.** The local learner exhibits bounded prior-experience reuse and feedback-specific revision on both opened development worlds. Acquisition and retention are inconsistent, and complete autonomous continuations remain unsuccessful. No new core is adopted.

## Registered outcomes

| Capability | W1 Thursday→Edgeworth | W2 Emma→Ball |
|---|---|---|
| acquisition | FAIL | PASS |
| reuse | PASS | PASS |
| retention | PASS | FAIL |
| revision | PASS | PASS |

These are development screens, not tests of statistical significance. Both text pairs were previously opened. Positions, repeated passages and forked histories are correlated, not thousands of independent worlds.

## What was learned, retained and revised

| Quantity (bits per actual target; lower loss is better) | W1 | W2 |
|---|---:|---:|
| A held at birth | 7.978293 | 7.979835 |
| A held after A | 4.573233 | 4.278426 |
| A train-fitted unigram | 4.444852 | 4.489725 |
| B-COLD minus RESET-OPT after one B pass | 0.160251 | 0.209061 |
| B-COLD minus RESET-OPT after three B passes | 0.240172 | 0.454275 |
| Stable A loss change after B | -0.121155 | 0.177351 |
| Stable A loss change after edited C (relative to after A) | -0.219530 | 0.065794 |
| B loss change after edited C | 0.050154 | 0.127670 |
| Correction margin advantage vs NOEDIT | 7.577892 | 8.140227 |
| Corrected-target loss advantage vs NOEDIT | 5.466718 | 6.524182 |

Acquisition requires ≥0.20-bit improvement from birth **and** ≥0.05 bits beyond the A unigram. W1 learns substantially from birth but finishes 0.128381 bits worse than the unigram. W2 beats its unigram by 0.211299 bits. Reuse requires ≥0.05 bits at B passes 1 and 3 for RESET-OPT versus B-COLD; both worlds pass. This supports operational transfer of A-trained parameters, not abstract rule discovery. KEEP's continuing optimizer improves practical B loss over cold as well; the relative KEEP/reset effect varies by checkpoint/world.

Retention gates are ≤0.10 stable-A degradation after B, ≤0.15 after C relative to after A, and ≤0.10 B degradation after C. W2 fails after B (+0.177351) and for B after C (+0.127670). W1's stable-A means improve; this is not a case where every new phase causes forgetting. Mean improvement does not mean every site improved: see per-site degradation counts and paired-record summaries.

Revision requires ≥0.20-bit extra new/old log-odds gain and ≥0.10-bit corrected-target loss benefit over NOEDIT. Both pass by substantial margins on 416 and 233 corrected held sites. EDITED and NOEDIT start from identical KEEP parameters, Adam state and dropout RNG; they observe the same 64 original record IDs, equal target counts and two presentations, with only legitimate raw-byte corrections differing. These train-derived edits define a synthetic environment, not factual/semantic understanding. They do not give the learner privileged keys, rules, task labels or old/new identities.

## Learning deterioration is a separate finding

All six B branches have worse held loss after three passes than after one:

| Branch: B3−B1 held loss | W1 | W2 |
| KEEP | 0.251371 | 0.177557 |
| RESET_OPT | 0.141598 | 0.080498 |
| B_COLD | 0.221518 | 0.325711 |

This is descriptive evidence of an unfavorable dose/optimization trajectory under this fixed chronological schedule. The saved data do not distinguish training underfit, overfit, recency bias, writer magnitude or local-credit geometry causally. Do not diagnose it solely as forgetting, prove a representation impossibility, or choose a validation-best pass after seeing these scores. The primary checkpoints remain the fixed final passes.

The chosen calibrated count controls predict B at 2.916540/3.018339 bits, versus KEEP at 4.533722/4.441904. The gap is 1.617182/1.423566 bits. The countermodels are evaluator-side, fitted using their own disclosed training/calibration budget, and never supply the learner's answers.

## Own bytes/EOS and evidence boundary

Every neural score comes from the same unmasked own 257-class head: payload 0–255 and genuine EOS256. Full next-byte panels use observed prefixes. Separately, all A/B held records receive three observed prefix bytes and then the model's own argmax continuation for up to32 bytes/EOS; truncation is explicitly reported. Complete continuations score **0 exact over all registered generation queries**. Most do not emit EOS before the cap; in W2 only a few pre-correction continuations do. No forced EOS or decoder repair was used. Predictive loss gains therefore cannot be presented as complete autonomous answers.

This is E0-qualified access plus bounded predictive-lifetime development evidence. It does not establish a complete-response E1 battery, E2 relevance/retention, E3 never-taught addition, E4 broad cognition or biological validity. No attention, Transformer, arithmetic representation, BPTT or evaluator-fitted learner head was introduced. The eventual unseen-addition objective remains unmet.

The descriptive exposure audit finds no identical train/held records, but some prefixes and labels overlap. Exact-prefix, canonical-prefix and suffix-16 tables with unigram fallback are reported, not silently excluded. Learned conditional frequencies remain an explanation for reuse/revision. A two-byte environmental edit can be learned statistically; passing it is not evidence of reasoning. The STABLE subset requires unchanged expected bytes and preceding16 observed bytes; the longer decaying sensory trace is not asserted identical. Retention is scored on the original still-valid held contexts.

## Fidelity, three cycles and limits

Cycle1 recovered pinned V167/V169/V163 source, authenticated all corpora, froze train-only edits and audited sensory timing. It preserved V167's input empty-state column5177 and pad5178; every original A train/calibration code matches the V176 cache after that explicit representation adapter. B and other panels are newly exported with the authenticated literal sensory function; no historical B binary-cache comparison is claimed. Full-byte fixtures compare literal native and exported codes, and changing a future byte leaves earlier codes unchanged.

Cycle2 used the literal V167 local_step, fixed feedback seeds21/22 and original V163 births. Independent synapse gradients, local-surrogate/readout finite differences, zero hidden-feedback, independent forks and save→restore→next-update equality passed. The local hidden teaching signal is random output-error broadcast; gradients do not pass through learned neighboring layers. AdamW is retained as an explicit optimizer, not claimed to be an animal synapse.

Cycle3 executed every registered A/B/C history on both worlds without efficacy-based gating, resets of the primary trajectory, best-checkpoint rescue, additional passes or retries. A is physically learned once per world; KEEP, RESET-OPT and B-COLD fork as declared. C forks the actual final KEEP checkpoint. Source/data, starting RNG, checkpoint hashes and complete full-class saved-output panels support a separate-process saved-only reduction. This is an internal artifact audit, **not an external independent reconstruction**. Raw panels/checkpoints remain local; publication manifests make that limit explicit.

Source/setup issues corrected during engineering are preserved in history and the ledger: NumPy metadata serialization, a dependency-order check, a generic module-name collision and loading existing native dependencies into the Torch environment. None involved a main scientific optimizer update or a scientific retry. A malformed historical commit string in SPEC is resolved using the full pinned commit in DESIGN.md, disclosed in SOURCE_RECOVERY.json. No historical source is modified.

V179 explicitly freezes **no added warmup**, unlike the broader project's warmup preference. This is the requested literal baseline adaptation; it does not adopt a new curriculum or settle whether gradual warmup helps. V179 changes the old A-specific OTHER interface to257, chronological batching and optimizer continuity; it is not a numerical replication of historical V169. Those adaptations are confounded when comparing across versions, so historical 3.443-bit performance cannot be attributed to a single changed mechanism.

## Exposure, cost and release

Main learning consumed **7,532 full-record presentations / 463,277 target applications**. Technical optimizer tests used24 records /1,401 targets. Total: **7,556 records /464,678 supervised target applications**, within7,600/510,000. These include repeated exposure; each descendant inherits the A learning cost. The1832 recorded science optimizer updates exclude the separately charged technical steps. Gradient-only qualification sites, read-only target panels, own generation, sensory export and count-model fits are separately disclosed and charged CPU. The count controls fit 28,696 record presentations /1,744,149 targets; the posthoc table audit has its own ledger.

The full sequential scientific worker used **203.91 CPU seconds** and 207.15 wall seconds. Across export, technical checks, count controls, saved-only reduction and conservative engineering allowances, accounting before final documentation is **296.59 CPU seconds**; peak process RSS is 897.1 MB. Final receipt adds packaging/documentation cost. Pure immutable sensory caches avoid repeated encoding; no learned activations are cached. The dense H/T learner still performs work over its matrices; this is not a sparse-computation scaling claim. No global environment install, GPU migration or model-equation optimization was needed.

The revised document is a prospective diagnostic plan, not permission to rerun V179 or invent a successor automatically. Existing source locks, final losses and failure gates remain unchanged. Package contains compact code, summaries and manifests; raw data, checkpoints, dependencies and full raw output panels are excluded.
