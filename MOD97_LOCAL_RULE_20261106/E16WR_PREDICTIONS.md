# E16-WR locked predictions

User confirmed all12 scientific lives, including observed-target C16384, before any E16WR training.
P1: final held composition strictly rises across self-target C1024<C4096<C16384 in every seed; largest-capacity loss is at most half the exact reused C_uni loss in every seed.
P2: self-target C16384 final mod97 operation mean is at least88% in every seed.
P3: C1024 is within5 percentage points of reused C_uni on final held composition and final old-mod97 mean, every seed. Similarity would not uniquely prove capacity rather than sampling caused prior loss.
P4: post-composition held final-answer mean is within3 percentage points of C_uni for every arm and seed, including observed-target. This is a literal two-sided prediction; improvements outside the band also count as misses.
P5: observed-target C16384 loses at least10 percentage points less held composition than self-target C16384 in every seed. Loss is each arm's own post-composition score minus final score. Within5pp is a null benefit at this capacity, not proof wrong targets never matter;5–10pp is intermediate.

No tuning, seed replacement, new tasks or additional scientific arms. Carry pairs, comparison shortcut limits, exact full scratchpads and operations all reported separately.
AlgorithmR samples duplicate presentations;4096slots do not guarantee all operation identities. Nine per-stage strata differ from the cell study's3families, and observed/self-target replay differ. No cell-only inference.
Observed treatment changes targets for all replayed operations and compositions throughout life, not only late composition targets. It tests that target-source package, including downstream Store writes and differences in acquisition.
Both largest arms save identical observed-byte payloads and use independent admission RNG plus common sampling RNG, so input order, coverage and reservoir hashes must match exactly.

## Provisional cost forecast
Prior C_uni receipts imply6.315 CPU-hours if all12cost the same; those CPU counters exclude final evaluation. Observed targets skip generation and may cost less; larger reservoirs add overhead. Budget forecast6–9 CPU-hours plus approximately0.4h full parity and short qualification, with1.5–3h wall AFTER the machine is free, provisional until measured. No promise of an under-one-hour finish. Qualification/technical costs, source hashes, reused births and comparator receipts recorded separately.

Raw model:470801 trainable scalars(1883204 float32 bytes); Adam moments approximately twice the weights plus scalar clocks, activations and gradients additional. Inference includes an unchanged65536-key exact-window Store. Reservoir cap is per stage: largest147456slots, each carrying prompt plus observed answer; Python object/RAM costs exceed byte payload. Per-worker peakRSS will be measured, not inferred from payload alone. Twelve scientific lives each57344updates×64items; one additional full default parity life. No repeated development births.
