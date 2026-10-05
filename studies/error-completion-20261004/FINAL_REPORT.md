# ERROR versus R center final completion results

Final complete-cohort report, 5 October 2026.

## Conclusion

ERROR did not repair held-out completion in this fixed experiment. The registered decision is **JOINT_REPAIR_EXCLUDED_BY_REGISTERED_BOUND**. Its held-out accuracy was below the 25% chance level, and its taught-old causal held-out contrast was negative. This conclusion is terminal for the registered question and is confined to taught-dependent held-out table completion in the locked relabeled Latin-table fixture.

The direct held-out differences between ERROR and R_center remain unresolved. The result does not establish that ERROR is better or worse than R_center on held-out completion, and it makes no claim about broader memory, reasoning, general intelligence or other tasks.

## Fixed design and valid controls

The analysis includes exactly the 64 registered worlds 320001 through 320064, with one accepted complete receipt per world. The world is the unit of inference. Each world pairs R_center and ERROR across the W, N_old_relation and N_new conditions. Each branch has 192 old and 192 new teaching events, with the registered delays and disposable evaluation probes. The three qualification cycles and pilot/replay are technical evidence, not additional official sample members.

Independent verification checked every original scientific receipt part, source/runtime identities, coefficient and clamp rules, clocks, disposable-probe conditions, and paired shared-trajectory identities. The final analysis's 16 contrast definitions, intervals and decision were independently reproduced. Both rules pass the positive taught-old and taught-new causal gates, their old/new above-chance gates, and the registered mean-accuracy threshold of 90%. Thus the failed completion result is accompanied by successful taught-item learning controls.

The scientific registration was frozen privately before official execution. Its first public projection was published after launch; this is not a claim of contemporaneous public preregistration. See [the protocol](protocol/PROTOCOL.md), [the scientific lock](protocol/SCIENTIFIC_LOCK.json), and [the initial release](https://github.com/ChangweiZhou/digitalbaby/tree/79c13ab8cede0f9fd7bda6bd45cc176735247702/studies/error-completion-20261004).

## Accuracies in the W condition

Values are percentages, with the registered adjusted interval in brackets. These intervals are the corresponding registered above-chance intervals translated by 25 percentage points.

| Outcome | ERROR | R_center |
| --- | ---: | ---: |
| Held-out | 14.06 [8.96, 19.16] | 15.23 [9.66, 20.81] |
| Taught old | 94.92 [92.33, 97.51] | 98.05 [96.36, 99.73] |
| Taught new | 91.89 [88.03, 95.76] | 94.92 [92.47, 97.37] |

ERROR's held-out result is 14.06% [8.96%, 19.16%], below 25% chance. Its taught-old causal held-out contrast, W minus N_old_relation, is -10.94 percentage points [-17.97, -3.91]. The registered bounds therefore establish negative taught-old contribution to held-out completion in this fixture. R_center's corresponding causal held-out contrast is -8.98 points [-16.32, -1.65].

The ERROR-minus-R_center direct held-out accuracy difference is -1.17 points [-3.84, +1.50]; the direct causal held-out difference is -1.95 points [-5.89, +1.99]. Both intervals include zero. Neither direct superiority nor a clear direct held-out difference is established.

## Preservation and the joint decision

ERROR's taught-old and taught-new means are 94.92% and 91.89%, respectively. The registered 90% gate is a gate on the mean, not on the lower confidence bound.

Relative to R_center, the old-accuracy difference is -3.13 points [-5.06, -1.19], and the new-accuracy difference is -3.03 points [-5.12, -0.93]. The preservation rule requires each lower bound to be strictly greater than -5 points. Both fail that rule. Failure to establish this noninferiority margin does **not** prove that either loss exceeds 5 points.

The ERROR completion gates fail: both its held-out above-chance contrast and its taught-old causal held-out contrast have upper bounds below zero. Its old/new learning controls pass, but old/new noninferiority does not. The registered decision is therefore joint repair excluded by the registered bound, rather than an inconclusive completion claim or a reduction-in-harm classification.

## All sixteen registered contrasts

All values below are percentage points. The raw_ fields mean accuracy minus the 25% chance level. E3 is W minus N_old_relation on held-out probes; E1 is the analogous taught-old contrast; new_gain is W minus N_new on taught-new probes. Direct contrasts and NI contrasts are ERROR minus R_center.

| Contrast | Meaning | Mean | Lower | Upper |
| --- | --- | ---: | ---: | ---: |
| R_center_E3 | R_center held-out taught-old causal contrast | -8.98 | -16.32 | -1.65 |
| R_center_raw_heldout | R_center held-out accuracy minus chance | -9.77 | -15.34 | -4.19 |
| R_center_E1 | R_center taught-old causal contrast | 72.53 | 69.22 | 75.84 |
| R_center_new_gain | R_center taught-new causal contrast | 70.41 | 67.35 | 73.47 |
| R_center_raw_old | R_center taught-old accuracy minus chance | 73.05 | 71.36 | 74.73 |
| R_center_raw_new | R_center taught-new accuracy minus chance | 69.92 | 67.47 | 72.37 |
| ERROR_E3 | ERROR held-out taught-old causal contrast | -10.94 | -17.97 | -3.91 |
| ERROR_raw_heldout | ERROR held-out accuracy minus chance | -10.94 | -16.04 | -5.84 |
| ERROR_E1 | ERROR taught-old causal contrast | 69.92 | 66.42 | 73.42 |
| ERROR_new_gain | ERROR taught-new causal contrast | 67.38 | 62.91 | 71.86 |
| ERROR_raw_old | ERROR taught-old accuracy minus chance | 69.92 | 67.33 | 72.51 |
| ERROR_raw_new | ERROR taught-new accuracy minus chance | 66.89 | 63.03 | 70.76 |
| direct_E3 | ERROR minus R_center causal held-out contrast | -1.95 | -5.89 | 1.99 |
| direct_heldout | ERROR minus R_center held-out accuracy | -1.17 | -3.84 | 1.50 |
| NI_old | ERROR minus R_center taught-old accuracy | -3.13 | -5.06 | -1.19 |
| NI_new | ERROR minus R_center taught-new accuracy | -3.03 | -5.12 | -0.93 |

All contrasts use 64 paired world-level values. The registered family has 16 contrasts and family alpha 0.05. The displayed bounds use the registered paired Student t construction with Bonferroni adjustment and theoretical-support clipping. Simultaneous coverage is approximate. All actual contrasts had nonzero sample variance, so the prespecified support-width Hoeffding fallback was not used.

## Operational amendments and scope

The [original infrastructure blocker report](results/report/INFRASTRUCTURE_BLOCKER_20261005.md) remains unchanged as a historical record of the original 18-hour host gate stopping the run. A later approved 24-hour operational allowance permitted continuation, retaining all prior charges. The original 18-hour cap is not represented as satisfied. An independently reviewed analysis-only import-path repair was required before final calculation. None of these operational steps changed the numerical source, learning rules, fixture, registered worlds, contrasts or decision rule. No already accepted completed world receipt was rerun; the complete fixed cohort, not a selected subset, determines this result.

The result is restricted to this locked fixture, schedule, output policy and registered controls. It does not authorize sample extension, parameter tuning, a later ERROR variant, or a broader conclusion about learning systems.

## Published evidence and reproducibility boundary

- [Lossless scientific receipt bundle and restoration instructions](results/bundle/README.md), preserving all 8,453 reviewed final scientific files
- [Complete public terminal analysis](results/final/analysis.json), including all per-world values and exact interval data
- [Public scientific verification summary](audits/FINAL_ANALYSIS_SUMMARY.json)
- [Final publication integrity manifest](results/final/PUBLICATION_MANIFEST.json)

The receipts are delivered in a 34,716,904-byte lossless tar.xz archive split into 89 small downloadable chunks. The bundle manifest and readable per-file SHA256 lists identify every decoded path and byte. The supplied Python-standard-library utility verifies the complete archive and all 8,453 files before restoring them to a new directory. Receipt-relative links in the analysis and archived report resolve after restoration. These files are not individually browsable as a raw receipt directory on GitHub.

The 8,320 non-header numeric receipt parts are byte-identical to the sealed originals. The 128 receipt headers/manifests are explicitly labeled public projections, and their complete public index is inside the bundle. The terminal analysis preserves every numerical value and decision; only its input-manifest references and projection metadata differ. Original execution administration, private checkpoint data and approval/session identities are excluded. These public projections must not be passed to validators that require the original private execution identities; the [initial README](README.md) explains the execution-adapter boundary.

The archive also preserves the exact reviewed report, analysis, verification-summary and publication-manifest snapshots. This live report and publication manifest add availability directions; all scientific text, numerical findings and limitations above are unchanged. The initial 397-file source/qualification release and its root integrity manifest remain historical and unchanged. Use the restored supplement alongside that source checkout. Inherited source and attribution remain intact; see [NOTICE.md](NOTICE.md) and [COPYING](COPYING).

SPDX-License-Identifier: GPL-3.0-or-later.
