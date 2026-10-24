# V32 run summary

Status: **completed**. Verdict: **not_qualified**.

Completed trajectories: 66. All intervals below use seeds, not bytes.

## Scientific encoder preflight

| Code | Rendering 0 acquired | Rendering 1 acquired | Held-out alias | Whole-stream bpb |
|---|---:|---:|---:|---:|
| last_fd | 86.5% | 70.8% | 9.4% | 2.5500 |
| v31 | 100.0% | 96.9% | 1.0% | 1.9807 |
| lag | 100.0% | 100.0% | 76.0% | 2.3073 |

Gate: True.

## Stage B

| Writer | Whole bpb | Target bpb | A initial | A after D | Rev 4 exact | Rev 4 alternate | Rev 4 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary | 2.4250 | 4.6005 | 75.8% | 8.6% | 56.2% | 0.0% | 0.0% |
| full | 1.4763 | 1.7393 | 100.0% | 7.8% | 100.0% | 15.6% | 3.1% |
| dictionary | 2.8572 | 0.8176 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |

Gate components: whole_bpb=pass, target_noninferiority=pass, retention_advantage=inconclusive, second_presentation_noninferiority=pass, retention_effect_size=adverse, A_initial_accuracy=pass, A_after_D=adverse, revision_4_alias_0=pass, revision_4_alias_1=adverse, revision_4_alias_2=adverse, noise_damage=adverse.

## Stage D

| Writer | Whole bpb | Target bpb | A initial | A after D | Rev 4 exact | Rev 4 alternate | Rev 4 held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary | 2.3878 | 4.6376 | 81.2% | 6.2% | 46.9% | 0.0% | 3.1% |
| full | 1.4525 | 1.6718 | 100.0% | 2.3% | 100.0% | 9.4% | 0.0% |
| full_h2048 | 1.4422 | 1.6690 | 100.0% | 3.1% | 100.0% | 9.4% | 0.0% |

Gate components: whole_bpb=pass, target_noninferiority=pass, retention_advantage=inconclusive, second_presentation_noninferiority=pass, retention_effect_size=adverse, A_initial_accuracy=pass, A_after_D=adverse, revision_4_alias_0=pass, revision_4_alias_1=adverse, revision_4_alias_2=adverse, noise_damage=adverse.

## Interpretation limits

The unseen alias changes both header and separator. Similar-key code geometry and route agreement are recorded; alias failure may reflect this sensory interface. The exact dictionary has growing capacity and cumulative counts, so its revision lag is not a limit on every possible dictionary.

Whole-stream bpb includes predictable syntax and unpredictable key bytes; target-only bpb and absolute acquisition/retention levels must accompany it. Retention fractions are undefined below their acquisition gate. Large data effective rank alone does not prove covariance saturation.

See REPORT.json for paired intervals and gate decisions, METRICS.csv for per-seed levels, jobs/ for probes and target events, and PROTOCOL.md for the locked specification.
