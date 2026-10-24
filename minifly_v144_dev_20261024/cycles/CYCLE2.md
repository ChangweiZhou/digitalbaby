# Cycle 2 — native diagnostic and short complete-addition trial

## Test run

The two source-locked native D2 histories each completed 150 passes/11,250 records. Their final per-prompt outputs exactly match the previously published D2 LEARN/FROZEN outputs. Added read-only checkpoints therefore did not alter the measured learning outcome. Actual saved KC arrays match their hashes; FROZEN's learned-bank codes are invariant.

The LEARN informative-bit absolute residual averages 0.438 over passes 10–20 and 0.0427 over passes 125–150. The latter window has mean feature-update norm 0.00955 against decay-update norm 0.000560, approximately 17 times larger. This contradicts the strong absence-of-teaching-signal explanation for this recipe. It does not establish that remaining credit is well directed, nor causally exclude every form of signal limitation.

The complete-addition fixture then ran BP/DFA/feature-frozen for 64 passes on one explicitly developmental world, H128, learning rate0.03, decay0.001, gradual 1,024-record ramp. Each received4,864 record presentations and11,904 observed output/EOS tokens. No CAP-D/H world was opened and no setting was chosen from these results.

| Mode | Taught complete orbit accuracy | Held complete orbit accuracy | Own EOS within cap | EOS at correct held payload length |
|---|---:|---:|---:|---:|
| BP | 77.50% | 6.67% | 100% | 86.67% |
| DFA | 81.25% | 3.33% | 100% | 73.33% |
| Feature-frozen | 15.00% | 6.67% | 100% | 56.67% |

All scored answers use the generated prefix and own EOS. Teacher-forced scores are separately labeled. EOS-within-cap means termination; it does not mean correct stopping position or correct answer. Complete accuracy requires all three.

## Audit and revision

Per-prompt hex outputs were reduced with separate stdlib code; the complete orbit metrics agree. Reference state hashes are unchanged by probes. Countermodels and the actual development split are retained in local manifests/results. An exact raw/canonical lookup has no reinforced held relation; all declared fitted countermodels scored0 here. This does not make one or two correct held orbits a reliable E3 result: BP does not exceed the frozen control, this is one world, and causal persistent-reuse controls are not included.

The prospective shortlist revision in REVISED_DESIGN.md retains a finalist at every hidden size and uses development teacher-forced loss only to resolve equal complete-accuracy scores. No new mechanism or arithmetic representation was introduced after the fixture results. The full finite capacity screen remains a separate execution;64 passes cannot prove its larger/longer configurations infeasible.

Native telemetry covers residual and update magnitudes; raw credit vectors/gate masks and per-compartment raw-alpha histories are not retained. Their absence is disclosed rather than filled by replaying the native histories.
