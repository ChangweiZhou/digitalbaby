# V34 — Sparse Expanded Compartment Memory

This protocol fixes the user's eight-arm experiment before scientific outcomes. There are exactly 64 scientific trajectories, with no parameter sweep, selection, or automatic Stage II. This is a test of a wiring hypothesis motivated by the supplied fly audit. It is not a simulation of the fly connectome or evidence for the functional interpretations of individual lobes.

## Fixed workload and independent seeds

Scientific seeds are **34101–34108**. Engineering seeds are **974301–974302**, with **974399** for full-length reference checks. The scientific seeds are new relative to the available preceding manifests and source seed lists; no parameters are selected from their outcomes. All eight arms share the exact stream and lag code for a seed. Seeds, not individual byte events, are the statistical units.

Reuse V33's stream generator without editing it: four waves A→B→C→D, 16 arbitrary six-byte keys per wave, 32 acquisition presentations per key alternating two renderings, four A keys corrupted once after B, and four disjoint A keys revised eight times in rendering 0 after C. The eight untouched A keys never appear in training after A. A third rendering is probed only. The learner receives only past bytes and the subsequently observed byte. There are no live resets or evaluator-triggered changes to learning.

Each trajectory has 64 logical keys, 128 learned target contexts, 64 never-trained alias target contexts, 2,084 target events, and 29,176 next-byte predictions. The complete experiment uses eight underlying streams, each repeated across eight arms: 1,867,264 next-byte predictions. Repetition does not count as a new unique example. Per-key and per-rendering exposures are saved separately.

## Locked arm roster

| Arm | Representation and participation | Writer |
|---|---|---|
| H-O | Exact V33 lag96 and fixed eight-bank argmax router | Ordinary V33 |
| H-P | Same hard-bank model | V33 full covariance |
| G-D-O | Fixed 768-KC expansion, all units participate | Global ordinary |
| G-S-O | Same expansion, global Top48 absolute activation | Global ordinary |
| C1-O | Highest-energy compartment, Top48 within it | Ordinary |
| C4-O | Four highest-energy compartments, Top12 in each | Ordinary |
| C4-P | Same C4 code | Local block covariance |
| Dict | Exact last-12-byte count dictionary | V33 dictionary |

All neural arms have exactly 196,608 learned byte-readout coefficients in float64: either 8×96×256 or 768×256. H-P and C4-P each have 73,728 inverse-covariance entries, initialized to eight identity matrices. Covariance is additional adaptive state, reported separately from learned output coefficients. The expansion adds fixed contacts/weights and computation; equal readout budgets do not imply equal total cost.

## Numerical completion of unspecified choices

The lag96 code is copied from V33 unchanged: signed byte bits, last 12 bytes in chronological order, leading zeros during startup, and one L2 normalization. It preserves byte identity bound to lag. H-O/H-P retain the original fixed routing centres and RNG order.

For KC arms use a separate RNG with seed `seed + 3410003`. Draw eight distinct contact indices without replacement from 0–95 for each of 768 KCs, in increasing KC order. Then draw a 768×8 standard-normal weight array and normalize each row to unit L2 norm. Contacts and signed weights never change. Compute `u_i = tanh(sum_j w_ij z_lag[contact_ij])` without bias. All KC arms for a seed use identical contacts and weights.

Compartments are consecutive, disjoint groups of 96 KCs. Their context memberships overlap because a context can participate in several groups; this does not reproduce overlapping anatomical axon territories literally. Rank compartments by the sum of squared **unpruned** activations. Rank KCs by absolute activation. Ties use the lower fixed index. G-D-O retains all 768 values; G-S-O retains global Top48; C1-O retains Top48 in one selected group; C4 retains Top12 in four groups. Preserve activation signs. Normalize the whole surviving 768-vector once to unit L2 norm. Do not normalize each group separately.

The empty-history code is zero and produces no weight or covariance change. Subsequent zero codes, if encountered, are counted explicitly. Selected support slots and genuinely nonzero active coordinates are both recorded, so exact cancellations cannot masquerade as 48 nonzero features. No replacement, threshold tuning, or allocation is introduced to correct such a degeneracy.

## Shared writer and exact local protection

Predictions use the sum of participating readout contributions, followed by V33 softmax temperature 0.1. After the next byte appears, compute the single residual `r = onehot(byte) − scores` from that combined prediction. All ordinary arms use `q0 = z/(z'z)` and `W += 0.6 outer(q0,r)`.

C4-P computes `pz_c = P_c z_c` within selected compartments, then divides every block by the **same** scalar `s = sum_c z_c' P_c z_c`. Thus `q_raw = pz/s`, not four independent corrections. Apply the exact V33 orthogonal-component cap globally: remove numerical projection along z from `q_raw−q0`, cap its norm at `sqrt((1/0.6²−1)/(z'z))`, and add q0. This preserves `z'q=1` and bounds `||q||/||q0||` by 1/0.6. The combined current score changes by exactly 60% of its residual.

After the weight update, update each participating block independently with the V33 Sherman–Morrison statistic:

`P_c ← P_c − outer(P_c z_c, P_c z_c)/(1 + z_c' P_c z_c)`.

Consequently `P_c = inverse(I + sum z_c z_c')`, within floating-point accuracy. Cross-compartment covariance is not stored. This is the specified block-diagonal historical geometry, not the inverse of a full 768×768 covariance. There is no forgetting factor or RLS weight gain.

An unselected compartment's W and P remain bit-for-bit unchanged. Covariance **can write inactive KC rows inside a selected compartment**: P_c z_c need not have the support of z_c. Therefore the 48-row write-budget match applies to C1-O versus C4-O; it does not apply to C4-P versus C4-O. Record actual nonzero write rows and compartments, including zero-residual cases, rather than claiming every selected block necessarily changes on every byte. No extra mask is imposed on the covariance direction.

## Prespecified statistical decisions

Untouched-A accuracy after D, averaged over both trained renderings, is primary. Always report its absolute acquisition level after A. D-current means the final after-D accuracy on D's own keys and two trained renderings. Acquisition validity for each arm requires mean A-after-A ≥90%, minimum seed A-after-A ≥75% (the operational meaning of no gross seed failures), and mean D-current ≥90%. Display the minimum seed and all individual results. Invalid acquisition cannot establish a retention mechanism.

Use complete paired two-sided 95% Student-t intervals across eight seeds. Missing seeds never pass. The planned contrasts are:

| Gate | Candidate minus comparator | Required mean A-after-D gain |
|---|---|---:|
| P1: sparse expansion | G-S-O minus G-D-O | ≥0.10 |
| P2: multiple compartments | C4-O minus C1-O | ≥0.10 |
| P3: local covariance | C4-P minus C4-O | ≥0.10 |
| P4: advance over V33 | C4-P minus H-P | ≥0.15 |

Each gate requires acquisition validity for **both** arms, paired 95% lower retention bound >0, and paired 95% lower D-current difference ≥−0.05. Applying the same D noninferiority requirement to P4 makes “beats V33” exclude refusing new learning. A promising V34 candidate additionally requires C4-P mean absolute A-after-D ≥50%. Do not require all P1–P3 explanations to pass for P4; each is a separate scientific claim.

The user's 95% intervals are reported as planned. Also report Holm-adjusted two-sided paired retention p-values across P1–P4 as a familywise sensitivity, without silently changing the registered gates. A failed gate is split into adverse evidence versus unresolved precision. A nonsignificant difference is not equivalence. Whole-stream, non-target and target-only prequential bits/byte remain visible diagnostics, without adding a new loss gate.

Normalize retention only for seeds with A acquisition ≥90%; save `(A_later−chance)/(A_acquired−chance)` for both 1/256 and the labeled 1/24 support-chance sensitivity. For loss use `(8−H_later)/(8−H_acquired)` only when acquired gain >0.5 bits as well. Do not clip fractions or report them without absolute levels. Ratios on an invalid arm do not qualify it.

## Read-only diagnostics

Before any training, compute representations of all keys and aliases in the evaluator only. For forms 0/1 and 0/2, report same-key and different-key cosine, active-KC Jaccard, compartment Jaccard, and the differences. Different-key references use all ordered i≠j pairs. Preserve per-key values and code uniqueness. Fixed features imply these representation measures cannot change during training.

Track utilization over all observed bytes and separately over target bytes: each KC's activation frequency, mean, maximum, Gini, and fraction never active. Record per-compartment participation, nonzero weight-write frequency, nonzero covariance-update frequency, distinct logical keys at target positions, and load imbalance. Include cumulative wave summaries and final per-KC frequency arrays. Logical key IDs are used only by this evaluator accounting.

For H-P and C4-P save by-wave per-block effective rank of P, C=P⁻¹, and C−I; trace(P), eigenvalue/condition summaries and occupancy. For writes save aggregate z'Pz, capped/uncapped norm ratios, cap frequency, normalized cap removal magnitude, fractional perpendicular reduction, Frobenius update norm, active rows, actual nonzero write rows, and selected/altered compartment counts. Distinguish all-byte and target-byte summaries. No per-byte dumps or fitted model matrices are exported.

After A, a read-only ledger sums old-probe RMS score disturbance per write, its squared sum, and net score movement from the after-A snapshot. Trained and held-out A probes remain separate. This is a description of score movement, not a unique causal attribution of forgetting.

Keep revision curves after 1/2/4 occurrences, plus the existing 8-occurrence endpoint from the unchanged schedule. Report exact, known-alternate and held-out rendering accuracy, loss and obsolete-target probability. Keep prequential first/second presentations, second exact-rendering encounters, noise damage, and new-cohort learning curves.

## Execution and scope

One self-contained Python file and one command, NumPy/SciPy, one BLAS thread, automatic macOS caffeinate, process lock, fixed manifests, atomic per-job resume and result archive, and embedded executing source hash. Re-running a completed experiment reuses its jobs; changed source/protocol, corrupt records and foreign jobs are rejected. Compact outputs include REPORT.json, SUMMARY.csv/MD, REPRESENTATION.json, UTILIZATION.json, COVARIANCE.json, REVISION_CURVES.csv, ACQUISITION_CURVES.csv, COSTS.json, exposure counts, protocol, manifest and executing source. Exactly 64 scientific jobs, regardless of results.

No MESU, gate, horizon, RLS writer, homeostasis, recurrence, learned encoder, replay, episodic-slot candidate, or automatic follow-up is added. Good alias overlap is not proof of semantic identity. Failure of these seven neural arms is not a theorem against all shared parametric memories. The growing dictionary diagnoses exact storage; its cumulative counts can delay revision.
