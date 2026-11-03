# V182 — one fewer locally taught hidden layer

Date: 2026-11-02 (America/New_York). Source: codex/minifly-consolidated-20261023@5871b412c9582fbdec72165193294ae9b3db5226.
Owner request: “Let us activate the newly built skill, and do an new round iteration now as V181 is out.” This authorizes this bounded iteration; no adopted-core change or budget expansion is inferred.

## Question and opposing decisions

V181 NORMAL remains worse than unigram after A2; CGLP slows acquisition. Hypothesis: the extra independently broadcast-taught T transformation contributes materially to the compact learner's acquisition gap. Candidate SHALLOW removes T and connects learned H directly to its own output-local softmax. If the fixed-dose acquisition gain and lifetime guards pass, retain the shallow architecture for future confirmation. If they fail, reject this exact shallow package as a sufficient repair; do not add another gate or automatically allocate V183. This does not uniquely diagnose capacity, feedback alignment or sensor information.

## Frozen intervention

NORMAL is literal V181 learner.py, gain 1, H64/T64/O257, 37,313 parameters. SHALLOW retains exactly the same H, O, FH, incidence-map and dropout births, deletes T/Tb and FT, and uses H64/O257, 33,153 parameters. Preserve 0.1 dropout immediately before O: its position therefore moves from T to H, and H's local derivative includes that actual mask. This is a shallow architecture package comparison, not a unique causal estimate of depth. Both learn from scratch, with manual fixed-feedback H signals, output-local CE, identical AdamW constants, 257 classes, genuine EOS and no warmup. No learned-layer BP, replay, task labels, parser, forced EOS or attention.

## Data and complete lives

Use the exact V181 corpus hashes, source record parser, sensor fingerprint, split manifest, train-only correction selector and original immutable IDs. M1/M2 are already opened development worlds. Recovered native arrays must reproduce V181 sensor digest; recovered data must reproduce the full V181 data audit and manifest. Path recovery is declared, not a scientific change.

Two worlds × two arms × 560 record presentations = 2,240 main records, 113,574 supervised byte/EOS targets. Per arm: A160×2, B48×3, then equal EDIT24×2 and NOEDIT24×2 continuations from identical B3 weights/moments/clock/dropout RNG. No score-dependent stopping or best-checkpoint restoration. Checkpoints: birth, A1/A2, B1/B2/B3, EDIT1/2, NOEDIT1/2 (40 total). All real tail batches included. Future labels never enter sensory state before prediction.

## Outcomes locked before scientific fitting

Primary: NORMAL A-held bits at A2 minus SHALLOW, at least 0.05 bits in BOTH worlds. Functional acquisition additionally requires SHALLOW to beat train-only unigram by at least 0.05 bits in both worlds. Original V181 counting scores are authenticated reference results, not candidate output or uncharged new fits.

Guard: each of B-held B3, B-held EDIT2, corrected A SAME_KC EDIT2 and CHANGED EDIT2 is no worse than NORMAL by over 0.10 bits in either world. Feedback-specific NOEDIT2 minus EDIT2 changed-target loss must improve by at least 0.10 bits in both worlds. Authentic EOS NLL on A-held and B-held at A2/B3/EDIT2 must be no worse than NORMAL by over 0.25 bits. Report all components and all stages. Own argmax generation: three genuine observed prefix bytes, at most 64 outputs, 16 A/16 B held records at A2/B3/EDIT2. Report real EOS, truncation, length-feasible exactness descriptively; arbitrary literary exactness is not a capability gate.

Verdicts: SHALLOW_DEV_GAIN only if primary, competence, lifetime and EOS guards pass; SHALLOW_RELATIVE_GAIN_BASELINE_GAP if primary/guards pass but unigram competence fails; SHALLOW_HYPOTHESIS_NOT_SUPPORTED otherwise. A missing technical/source/resource prerequisite gives an explicit INVALID/NOT_RUN reason, not an efficacy verdict. No E1 deterministic-answer, E3 arithmetic, semantic revision, sealed confirmation or adoption claim.

## Physical contract and checks

At most 40k parameters per arm, 8 MiB deployed model-specific inference state, 1 MiB full mutable checkpoint, 256 MiB output, 5,000 complete record presentations, 600k target applications, 900 aggregate experiment CPU seconds, 1,800 wall seconds and 1 GiB peak process RSS. Main run is 2,240 records; technical cap 32 complete records. Account every technical update, all read-only scoring/generation, checkpoint and sensory arrays. Runtime dependency installation is setup and separately disclosed; it supplies no learned features.

Before fitting: source/hash/data/sensor recovery; H/O/FH birth equality; independent output CE/local-surrogate gradient check; zero-feedback liveness; serialization and continuation; resource census. An isolated autograd reference is a technical instrument only. Baseline same-stage published V181 scalar parity is checked after the roster, tolerance 1e-4 bits; mismatch means no exact reproduction claim and is investigated rather than silently replacing old evidence.

Freeze source hashes before main training. Save full mutable states and sufficient per-site/record outputs. A separate saved-only reducer recomputes decisions, validates receipts and source integrity without training. A same-family reviewer is not cross-family certification. Historical MRK-1 checker remains unavailable. Preserve original V181 and adopted core.

Pre-training recovery amendment: source arrays are exact file-hash matches. Reconstruct receptor uniform draws using explicit libm fused multiply-add to reproduce the original ARM/FMA sensor fingerprint on x86; 205 float64 entries differ by rounding under ordinary x86 uniform. The complete pinned sensor digest matches exactly. Pin NumPy 2.4.6 and Torch 2.14.0+cpu; failed setup checks applied zero scientific updates.

Technical accounting amendment before science: two source-qualification executions applied eight complete-record technical presentations in total; both are charged. Main run checks carry qualification CPU plus 120 conservative setup/recovery seconds. Technical/recovery mismatches never triggered a scientific update. The source lock includes the exact external V181 spec/data/manifest/count/compact-panel dependencies and the saved-only reducer.
