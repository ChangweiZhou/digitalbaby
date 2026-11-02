# V179-PLASTICITY-1 — paired hidden-plasticity diagnostic

**Completed bounded causal-package diagnostic. Official run not launched; no new model adopted.**

Prespecified interpretation: **CLAMP_HELD_BETTER_BOTH_OPENED_WORLDS**. This compares ordinary continuing local H/T/O learning with a temporary H/T learning clamp during B, from the exact same A-trained state. Both branches then resume ordinary hidden learning for C.

## Primary comparison: B held prediction

Bits per actual byte/EOS target; lower is better. Effect is CLAMP−NORMAL, so a negative value favors the clamp. Final pass3 is primary; pass1/2 are reported without choosing a preferred stopping point.

| World | B pass | NORMAL train | CLAMP train | NORMAL held | CLAMP held | CLAMP−NORMAL held |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 1 | 3.793819 | 4.011057 | 4.282350 | 4.338836 | +0.056485 |
| W1 | 2 | 3.446056 | 3.940812 | 4.297691 | 4.407441 | +0.109750 |
| W1 | 3 | 3.213919 | 3.867256 | 4.533722 | 4.371220 | -0.162501 |
| W2 | 1 | 4.040985 | 4.304624 | 4.264347 | 4.410200 | +0.145853 |
| W2 | 2 | 3.760601 | 4.116280 | 4.309520 | 4.289677 | -0.019843 |
| W2 | 3 | 3.482919 | 4.109729 | 4.441904 | 4.316607 | -0.125297 |

| World | Branch | Δtrain B1→B3 | Δheld B1→B3 | Pattern |
|---|---|---:|---:|---|
| W1 | NORMAL | -0.579900 | +0.251371 | OVERFITTING_LIKE |
| W1 | CLAMP | -0.143801 | +0.032385 | OVERFITTING_LIKE |
| W2 | NORMAL | -0.558066 | +0.177557 | OVERFITTING_LIKE |
| W2 | CLAMP | -0.194896 | -0.093593 | MIXED_OR_FLAT |

Numerical tie tolerance1e−4bits is not a statistical significance threshold. The saved paired-record effects retain the distribution of differences. Positions, repetitions, forks and records within a text are not independent worlds; both literary-text pairs were previously opened.

## Continuation after the intervention

H/T learning is switched back on for both C branches. The clamp branch resumes its actual A-preserved hidden moments/steps; the normal branch carries its actual B-adapted hidden moments. Neither receives a reset, synthetic moment aging or best-checkpoint reload. Each B endpoint is forked into equal-dose EDITED and NOEDIT feedback, with identical initial parameters/moments/RNG for that pair.

| World | B history | New/old margin advantage EDITED−NOEDIT | Changed-target loss benefit NOEDIT−EDITED | Revision screen | Stable A change after C vs after A | B loss change after C vs after B | Retention screen |
|---|---|---:|---:|---|---:|---:|---|
| W1 | NORMAL | 7.577892 | 5.466718 | PASS | -0.219530 | +0.050154 | PASS |
| W1 | CLAMP | 7.251078 | 5.145219 | PASS | -0.109482 | +0.401093 | FAIL |
| W2 | NORMAL | 8.140227 | 6.524182 | PASS | +0.065794 | +0.127670 | FAIL |
| W2 | CLAMP | 9.230858 | 7.983343 | PASS | +0.173182 | +0.380477 | FAIL |

These are the original V179 revision and retention screens: margin advantage≥0.20bits and corrected-target loss benefit≥0.10; stable A degradation≤0.10 after B and≤0.15 after C, B degradation≤0.10 after C. They are development screens, not confirmation tests. We do not re-label an inherited A acquisition/reuse result as a new test of those capabilities. Source A learning and historical B-COLD/RESET results remain separately scoped.

| World | History | B held at B3 | B held after edited C2 | A held after edited C2 |
|---|---|---:|---:|---:|
| W1 | NORMAL | 4.533722 | 4.583875 | 4.512244 |
| W1 | CLAMP | 4.371220 | 4.772314 | 4.620824 |
| W2 | NORMAL | 4.441904 | 4.569574 | 4.369878 |
| W2 | CLAMP | 4.316607 | 4.697084 | 4.469206 |

## What this establishes and leaves unresolved

Preserving the A-trained hidden package during B improves final held prediction in both opened worlds. Under this fixed dose/history, continued hidden plasticity is a contributor to the weaker final B prediction. The result localizes a useful causal handle; it does not establish that the learner cannot acquire taught information or that all feature learning is harmful. The full B curves show whether an early benefit reverses later. Any held improvement must be considered alongside the C continuation results above.

The clamp jointly preserves hidden weights, biases, Adam moments and per-parameter optimizer age and removes hidden writing. NORMAL and CLAMP can consequently produce different own-output gradients and write amounts. The test does **not** isolate representation drift from effective plastic capacity, writer magnitude, moment history, H versus T, or generalization effects of the output layer. It also does not prove that freezing features is a suitable final architecture. Hidden representations are A-trained features, not a new random-feature learner.

The C comparison is a downstream consequence of that B history, including its different optimizer state. A C failure cannot be attributed solely to weight unfreezing. An improved B endpoint cannot be promoted while suppressing a continuation failure. A future single intervention must target the measured history-dependent tradeoff with a matched comparator; it is not authorized by this report.

## Actual writing and integrity

| World | B arm | H sum squared updates | T sum squared updates | O sum squared updates |
|---|---|---:|---:|---:|
| W1 | NORMAL | 15.832322 | 2.703075 | 0.432601 |
| W1 | CLAMP | 0.000000 | 0.000000 | 0.444045 |
| W2 | NORMAL | 16.635029 | 2.843959 | 0.485688 |
| W2 | CLAMP | 0.000000 | 0.000000 | 0.519178 |

After hidden learning resumes, the edited-C T-layer sums of squared parameter updates are larger for CLAMP histories: W1: 1.0372 vs 0.8364 (1.240×), W2: 1.0082 vs 0.7788 (1.295×). This saved-trace association supports examining the resumption dynamics, but does not isolate moment age, parameter position or feature/readout co-adaptation. The dropout RNG hashes also match across B-history arms at every corresponding fork.

These are sums of squared parameter movements, not biological energy or net displacement. H/T movements are exactly zero in every CLAMP B update; all six B-boundary digests verify that weights, moments and steps equal the common A state. Discarded hidden teaching gradients are still computed and logged. Actual hidden writing resumes during C. Output O learns throughout, using the same local CE rule and full257 alphabet. No evaluator-fitted decoder is used.

Technical source/lesion qualification passed before main scores. Source NORMAL and explicit NORMAL updates match exactly; the clamp preserves decay/moments/steps while O changes; a checkpoint roundtrip preserves the clamp state; H/T updates resume; read-only scoring preserves state/RNG. NORMAL reproduces all16 corresponding A/B/C parameter/optimizer/RNG hashes and held scores. The two B forks start identical, and every C feedback pair starts identical. The protocol, sources, parent checkpoints, reducer and data manifests were locked before main training. No source, dose, gate, seed, edit mapping or optimizer policy was tuned after observing outcomes.

## Own output, data and evidence level

Complete own continuations: **0/10095** queries over all registered checkpoints. Three observed bytes are followed by at most32 model-generated bytes/EOS. Own EOS, truncation and exactness are saved separately; predictive loss is not a substitute for complete responses.

Both arms observe the identical raw B stream and teacher information, in identical chronological batches and matched dropout streams at the fork. Sensory inputs encode observed prefixes only. H/T supervision is computed in both arms; the intervention discards its gradients before the optimizer. Correction is generated from the original train-only mappings; edited raw records are re-encoded, no keys/old-new identities or phase IDs enter neural input. The freeze is an experimenter-controlled intervention, not a deployed task-phase controller.

Retain the original exposure/recitation audit and count references. Exact held records are not duplicates, but overlapping prefixes and conditional-frequency learning remain countermodels. No additional counting/probe model is fitted here. Class masks are evaluator statistics only. Same raw257 head for payload/EOS, no forced EOS, arithmetic structure, attention, Transformer, ELM, BPTT or pretrained answerer. Original no-added-warmup is kept for causal comparability; it is not adopted as a preferred curriculum.

Evidence is E0-qualified access and a bounded predictive-learning causal-package diagnostic. No E2/E3/E4, never-taught arithmetic, biological mechanism, broad architecture success or adoption claim. This does not launch the official experiment or a new version. The eventual goal remains retained reusable learning and own-byte/EOS unseen-addition generalization per information and compute.

## Exposure and compute

New main learning: **2,980 full-record presentations /187,606 target applications**. Technical tests: **8 /381**. New total: **2,988 records /187,987 targets**, within3,100/200,000. Main optimizer updates: **748**. Both source A states inherit **4,086 A training presentations /246,108 targets**, learned before this diagnostic and charged as prior experience; no physical A retraining or free-pretraining efficiency claim. Shared descendants inherit that history; its logical cost is not erased by cloning.

Across the three V179 runs, physical recorded learning totals are **18,084 record presentations /1,116,342 targets**. These include repetitions and multiple control lives, not unique samples. Current accounted CPU before documentation: **195.60s**; peak process RSS **1018.4MB**. Read-only panels, own generation, discarded hidden-gradient computation, source hashing and saved-only reduction all count toward CPU. Existing authenticated sensory assets and identical checkpoints are referenced rather than copied; new control states are retained. Serialization reuse occurs only after the corresponding normal learning was physically executed and hash-matched.

The separate saved-only reducer checks scores/labels/denominators, fork equality, all-update coverage, exact frozen/restored writing, sources and state/panel hashes. Per-site metrics and checkpoints stay local and allow reconstruction; no external raw reconstruction is claimed. No model was retrained merely for reporting. Compact package excludes raw texts, encoded data, raw output panels, weights, local dependencies and per-update raw traces.
