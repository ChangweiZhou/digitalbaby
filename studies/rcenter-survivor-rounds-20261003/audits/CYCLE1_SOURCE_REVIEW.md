# Independent Cycle 1 source/design review

## Decision and exact scope

**ACCEPTED FOR THE LOCKED CYCLE 1 SMOKE ONLY.** This is a static design/source acceptance, not a passed smoke test, acceptance of Cycle 2 or Cycle 3, scientific qualification, or permission to launch the official worlds or publish files.

The accepted design is `protocol/ROUND1_DESIGN_CYCLE1.md`, SHA-256 `9d8c6b4c8aed261e0ecd21d38fa4339ec62c2fa234f3165342f5ac4adad0dccb`. The accompanying JSON binds all 122 files in the runtime admission closure, including the unchanged learner/vendor projection, new evaluator/governance code, exact smoke script, supervisor, and preserved design history. A changed file invalidates this acceptance.

Authorized technical scope is exactly the first two records of development seed 310000, one R_center base newborn with eight separately born stores and one matched clone, W and N_old_relation only. The supervisor's 512 MiB smoke RSS ceiling and 900-second cap apply. A fresh test must still pass runtime, source, report, resource and exclusive-supervisor admission. No additional development native outcomes are authorized by this report.

No native learner, store birth, vendor import, training event, or native prediction was executed in this review. Work was static reading, AST parsing, archive/file hashing, and pure finite fixture evaluation using the standard library. No external upload occurred. The coordinator must check host memory availability and conflicting heavy work before admission, as required by the design.

## 1. Provenance and imported mechanism

The supplied archive is 89,884,699 bytes, has 330 members, and hashes to `c9894b2bf9e868cf9d0981abe6009d8c6804eaf2a21aeef4bf3a9a43e93e9363`. All 113 extracted manifest entries match their sizes/hashes and their corresponding bytes inside that archive. The manifest itself hashes to `15175fc684b1626c7b66c5a654c1cb3e14cc70a1f7c065607ba4dfb5f15650d5`. Old behavioral claims were not treated as evidence for this programme.

The operative source chain was read through:

- baseline `src/learner.py`, `bootstrap.py`, and original `integrity.py`/`PROTOCOL.md`;
- vendored `paths.py`, `stores.py`, `portable_birth.py`, `common_platform.py`;
- `content_model.py`, `brain_byte.py`, `bytecore.py`;
- native `model_evo.py`, `common_evo.py`, `model73.py`, `model76.py`, and `smallfly.py` methods relevant to birth, encoding, adaptation, teaching, clone, readout and clocks.

The immutable learner SHA is `d69b5c1561048aaa37b432a22a107202bf2fbba666a161b392a7e7ab7c245973`. `stores.birth()` calls a separate canonical birth for every store. The private conversion clones that store's newborn state and installs the frozen CONTENT front end. The native FE0 and CONTENT paths, canonical B anchoring, two fast columns, slow state, and signed interface are unchanged. The fixed read scales match the calibration file. No new learner support/identity ledger, gain, topology, eligibility or write rule is introduced.

`engine.py` binds the new local governance adapter, verifies its path, then imports the byte-identical archived Learner. New fixture code is named `survivor_fixture.py`, avoiding collision with the dormant inherited `fixture` import in `common_platform`. Static reading finds no call from the Learner into either evaluator world generator. Dynamic poison/path-resolution tests remain a Cycle 2 requirement.

## 2. Actual feedback boundary

`Learner.predict()` computes shared/private raw values and their fixed combined output, then caches only shared values. Its returned emitted byte is not read by `observe_outcome()`. That method receives the actual environmental outcome byte, constructs y there, calls shared native teaching with `int(ALPHABET[j] != byte)`, and for R_center sets c=0 and s=1/4−y. The shared-softmax p is still computed/logged but is not used in R_center's coefficient selection. Neither private scores nor combined scores feed the shared teacher or the centered coefficient.

This supports the narrow static no-combined-feedback assertion. It does **not** mean the entire native write depends only on the outcome: `_Teach.teach_signed()` computes native W(0)/W(1) from that store's own pre-event state and cue representation, and native dynamics contain their own within-store feedback. Those mechanisms are frozen.

The declared offline policy scores are exactly S/sS + alpha P/sP for alpha 1, .5 and 0. Under the fixed, action-independent byte stream, changing only an emitted readout cannot change future learner input or teacher coefficients. Full-state counterfactual evidence is still pending; this review does not upgrade alpha=.5 to an executed closed-loop learner.

## 3. Fixture identifiability, balance and limits

The independent pure checker enumerated all 576 order-four Latin squares and all 13,824 triples of L/R/O permutations, representing 1,152 distinct table/holdout pairs. Every partial table has exactly one consistent Latin completion. Each held-out row, column and outcome occurs once; each taught row, column and outcome occurs three times. The GF4 map [0,2,3,1] gives x XOR 2x = [0,3,1,2], hence a balanced outcome transversal.

The actual new evaluator module was separately executed as pure fixture code for development seeds 310000/310101/310102 and the reserved pilot seed 310200, without importing any learner or vendor. Each has 384 total events, 192 old and 192 new; each old taught cue appears 16 times and each new cue 12 times; each outcome appears 48 times per block; held-out cues appear zero times. Its endpoints are old_end=31,680, new_start=118,080, new_end=149,760, final=236,160 seconds. Full 12-byte cue length is preserved.

A row-only missing-symbol solver attains 100% on this fixture. The final endpoint label should be **taught-dependent held-out table completion**; E3 is shorthand for this fixture-specific endpoint. The revised design correctly limits its interpretation accordingly. A stronger future claim needs a prespecified fresh split/control that separates row completion from cross-row reuse; this fixture must not be changed post hoc. It would not establish a reusable symbolic operator, out-of-alphabet transfer, arithmetic, or an abstract relational representation. The old exact-cue denominator is now 12 rather than the historical 16, so old-study accuracy is not a contemporaneous replication benchmark.

Evidence: `CYCLE1_SOURCE_FIXTURE_CHECK.py/.json` and `CYCLE1_SOURCE_RUNTIME_FIXTURE_CHECK.json`. The first certificate records the initial design hash because the finite fixture was checked before the prose revision; the fixture definition is unchanged, and the second certificate binds the exact runtime fixture hash.

## 4. Causal branches, teaching and clocks

The external actuator wrapper leaves byte/outcome histories intact and forces both teach methods to write=False only in the designated branch. This suppresses all native value-write components, including both fast columns and slow updates, while preserving the event's nonplastic advancement. The native adaptation equation depends on x, elapsed interval and preceding adaptation, not punishment or the plastic flag. Thus matched nonplastic sensory/adaptation history is source-supported.

W−N_old_relation estimates the total effect of permitting old value learning, including changes it causes during subsequent new teaching. It does not hold exact old facts fixed, selectively remove a distinct relation module, or estimate an effect of the shared bank alone. The revised text correctly says so.

F151 byte handling first commits the previous interval, captures pre-arrival features, feeds the arriving byte, and reserves that interval for teach. Shared/private teach use the pending pre-outcome code, then clear it; newline and flush advance unreinforced time. Backward-time checks exist in both brain and front end. Every probe starts on a fresh disposable clone, consumes cue bytes only, records its first response, and scores against a target held outside the learner. The continuing branch is hashed before/after. This is a sound static design; adversarial time, clone-alias, and byte/outcome leakage tests are still required.

Existing state digests cover operative mutable arrays and principal clocks, with CONTENT additionally including its visible window. They do not constitute a blanket serialization guarantee: if later work uses inherited snapshot/restore for private stores, it must explicitly cover `fe.visible`, which the generic F151 snapshot does not store. Fixed B/reader/parameter identity and excluded instrumentation must also be considered in a claimed full-state counterfactual certificate.

## 5. Inference and routing

The fixed 11-contrast family, world-only statistical unit, complete-64-world rule, paired means, df=63, two-sided Bonferroni intervals, and registered range widths are internally consistent. Branch/policy differences lie in [-1,1] (width 2); accuracy minus .25 has width 1. The design acknowledges that ordinary Student-t coverage is approximate, and does not use within-world cues or repetitions as independent n.

The zero-variance Hoeffding fallback is intentionally severe: with n=64 and 11 intervals its half-width is about .218066 for width 1 and .436133 for width 2. Consequently, even an observed identical half/current accuracy across every world cannot establish the 5 pp noninferiority bound via this fallback. This is a power/precision limitation to report, not a reason to change the procedure after outcomes.

Current-core usefulness requires both causal held-out gain and an inferential above-chance raw held-out score, plus retention/acquisition gates. The revised half-policy text removes an ambiguity: alpha=.5 is secondary and cannot be promoted as a useful mechanism in Round 1. Its E3 plus two harm bounds may motivate a separately frozen read-authority plan; that later study must establish its own utility/joint gates. There is no raw half held-out-above-chance contrast in the present family, so no such confirmatory claim is licensed here.

A positive shared contrast and an inconclusive combined contrast do not prove that the two read policies differ. Route B is acceptable as a decision about which fresh plan to prepare, not a confirmed suppression mechanism. Route C likewise means failure to establish shared E3, not proof of failed formation. Joint-gate failure stops mechanism-success claims. No branch launches a later round automatically. No analysis implementation or final-result audit is accepted by this source review.

## 6. Corrected governance and bounded smoke

The initial revision was not accepted; its report and hash snapshot are preserved separately. The coordinator corrected the exact alpha formula, half-policy status, row-completion/scope limitation, route-localization wording, smoke quota, runtime requirement, and explicit launch gates before any native observation.

The current runtime hash closure includes the supervisor and excludes the generated official lock from self-hashing. Technical admission requires an accepted review, exact design hash, a nonempty source map equal to the full current closure, exact report hash, supervised PID/source identity, and exact pinned runtime. The supervisor records runtime, holds an exclusive file lock, rejects an existing receipt, polls RSS/time/disk, records attempts, and refuses a failed-key retry without separately accepted infrastructure classification bound to failed-attempt count and current source. The smoke result uses fsync followed by no-overwrite hard-link publication, avoiding a partially written successful destination.

Pinned package metadata was inspected: Python configuration 3.11.15, numpy 2.2.6, scipy 1.14.1, numba .61.2, pandas 2.2.3. These are static installation observations; the actual supervisor must still validate imported runtime versions before a newborn. The smoke checks signed coefficients, no-write logging, clock flush, policy score construction and branch divergence. Two records cannot qualify the entire learner, full clock lifecycle, leakage boundary or scientific inference.

## 7. Explicit later-cycle / prelaunch requirements

The following remain open and are **not approved by this report**:

1. Cycle 2: separate independent design/code review, exact target/leakage adversaries, constructor/import provenance, native state-equivalent output interventions, clone isolation, all branch clocks including both full delays, and source/report/clock tamper rejection.
2. Cycle 3: complete fixed pilot; exact fresh replay; interruption identity/hash reconciliation and conservative attempt charging; no duplicate writers; missing-only resume; proof that incomplete/rejected attempts never become accepted receipts. Preserve every failed artifact and receipt. Ledger/journal crash durability must be tested, not inferred from rename alone.
3. Resource qualification: host MemAvailable and concurrent-work admission; bounded result/log/cache storage; final high-water checks and live total-time/disk enforcement; attempt-level unique artifacts; explicit feasibility forecast and coordinator approval. Sixty-four receipts at the 900-second cap would take 16 worker-hours before qualification, exceeding the 12-hour total. Therefore the cap alone is not a feasibility certificate: the full pilot and reserved retry/qualification costs must support launch without expanding budget or reducing sample after outcomes.
4. Exact public/private persistence paths: actual immutable off-host backup, downloaded-byte hash verification and restore; a representative new technical receipt through the exact incremental GitHub path and readback; no content change means no new archive/commit; cohort barrier and restart reconciliation. None of these paths was exercised here.
5. Public release provenance/licensing: the source contains GPL-3.0-or-later notices, but no LICENSE/COPYING member was found in the supplied archive. Retain notices and resolve the corresponding license/attribution materials and any other distribution constraints before public source release. This is a release gate, not a claim that private static testing is prohibited. Publish only the authorized new scientific projection; do not infer permission to re-upload the old archive or administrative metadata.
6. Final official source/runtime lock, all three distinct design→audit→test acceptances, independently reviewed analysis code and final result audit. Historical acceptance files and this Cycle 1 report cannot substitute for those gates.

## Final disposition

The revised design and source are acceptable to attempt the single specified Cycle 1 smoke under the frozen admission controls. Smoke success remains unobserved. Every later cycle, public action and official launch remains gated.
