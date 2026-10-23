# Cycle 1 independent pre-test audit

Verdict: PASS for the stated source/history-only small test. This is not approval for a fresh-world life or full run. No outcomes from this cycle were read before this audit.

The fixed signed, cue-only, pre-update elementwise EMA equation is explicit and does not receive evaluator labels, pair indices, digit positions, future data or outcomes. Keeping beta, gain, decay and the residual update fixed across all three cycles prevents pilot-result rescue. The old completed response histories are authorized development data; unopened Package A outcomes remain excluded. Negative preflight results must remain in the record.

## Required interpretation and implementation checks

1. The displayed normalized 16-query Gram compression is a geometry diagnostic. Its 9-dimensional interaction minimum does not establish task-relevant finite-horizon gain, especially with history-varying EMA, normalization, decay, cross-subspace coupling and unequal query/train features. The design explicitly recognizes this. Preserve all eigenvalues including zeros, normalization units, absolute feature norms, and coupling blocks.
2. Compare original J reconstruction at its literal checkpoint, time and frozen parameter values. Source default gain is not necessarily the locked historical gain. Raw response tensor RMS, decision-centered RMS, added-bank RMS and W-minus-N_old RMS are different quantities; do not match a quote by silently switching them.
3. Probe each cue from the same cloned history state. Its within-cue EMA evolution must match normal cue processing, but none of that query's updates may affect the next query or continuing history. Test label permutation and query-order invariance of feature panels explicitly.
4. The next cycle's formation operator needs an explicitly declared teacher target domain and readout codomain. For a fixed target repeated across bouts, compose teacher-to-query K with the known repeat incidence. Use the symmetric part only on a common target/query basis, and include cross-space terms. A task-specific projection is not a worst-direction guarantee.
5. A linear score forecast requires a verified no-clipping certificate or a piecewise-exact update. Total-score and total-margin predictions cannot use an observed final native score as if it were predicted.

## Required correction before cycle 2 shared-relation test

The proposed off-diagonal cosine of centered kernels does not itself rule out exact-address codes. Centering an identity kernel produces H=I−11ᵀ/n, whose off-diagonal entries are nonzero. Remove that isotropic centered-space component before the alignment test:

C(K)=H K H; R(K)=C(K)−trace(C(K))/(n−1) H.

Use the cosine of the off-diagonal entries of R(native) and R(candidate), require both residual norms above a predeclared numerical tolerance, and treat a zero residual as undefined/failure, never perfect agreement. Report residual magnitude as a fraction of original centered-kernel norm; a tiny residual on a nearly identity kernel must not receive an unsupported strong-preservation interpretation. A minimum fraction should be declared if categorical exclusion of near-address models is desired.

Fix the 30 ordered distinct-symbol cues over six symbols from source before evaluation. Define X by concatenating sender and receiver one-hot incidence, H=I−11ᵀ/30, and P_shared as the orthogonal projector onto col(HX), with rank and tolerance recorded. No label is needed by the feature path. For centered trace-normalized kernels C/trace(C), shared mass is trace(P_shared C)/trace(C). Compare candidate mass against native FE0 using the same cues, literal times, state snapshots and trace conventions. Publish all off-diagonal residual similarities rather than only their aggregate.

This is an operational source-geometry preservation certificate, not evidence of transfer or reasoning. Native features must be the actual FE0 encoding used by the native route, with independently checked extraction; replacing them by byte identity or invented embeddings would not answer the claim.

## Resource and outcome handling

The 120 CPU-second, single-thread, 1 GiB and 20 MB cycle limits are explicit. A technical timeout or reconstruction mismatch is a failed qualification requiring diagnosis; it is not a scientific null and does not justify relaxing the frozen numerical gates. The source and old histories may be used to finish the remaining validation cycles after an A gate fails, but no fresh full run is authorized by this audit.
