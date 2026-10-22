# External-memory V1-R1: three native diagnostic cycles

The audit helped. Three bounded **native-memory** design/audit/test/revise cycles are complete; the Transformer/learned byte bridge and H0/H1 training are **not implemented or qualified** by them. These are E1 component diagnostics, not a full hybrid experiment or addition result. Final recommendation: keep gradual warmup and test learned-address interference and generic EOS calibration before scaling host work.

| Cycle | Actual test | Result | Revision |
|---|---|---|---|
|1|Random88-channel PN signals through canonical B/PN/KC; matched single W/N write, readonly/no-write/all257 codec|~9.8% cross-key same-slot overlap; localized write effect; collapsed-slot control100%|Retain signed per-slot margins, not overlap alone|
|2|16 random4-byte facts;1/4/16/64 cumulative passes; original gradual ramp|Payload bits87.3%→95.5%; exact own-EOS0% throughout|Separate content from EOS, count extra dose/clock honestly|
|3|Fresh16OLD/16NEW;16OLDpasses;two1hrests;paired W/N/SH|Final W bits91.4% vsN49.4%,SH48.8%; exact0% for all|Investigate stop/readout bias, actual learned geometry; no rate-only rescue|

The strongest new observation is **content-specific storage without correct stopping**, even under gradually ramped feedback and random surrogate addresses. At Cycle3 final, W has positive target-aligned payload margin0.03333, N0 and SH0.000233. Thus the original “only common format is learned” diagnosis does not apply universally to this new surrogate. Yet forcing the answer length would produce only31.25% exact four-byte payloads, so EOS is not the only remaining issue. That number is a diagnostic upper bound under forced length, never a scored success. W's EOS slot4 margin is negative, identical to SH, and its own generated sequence does not end at4. Positive EOS occurs once versus four negative payload-slot targets per record; positional overlap can bias this channel. This is a supported hypothesis, not an isolated causal proof of the EOS learning mechanism.

The passive fast-component estimate after7200s is exp(-7200/3375.62)≈11.8%; actual payload bits remained fairly stable while margins declined. Slow storage, thresholding, and nonlinear readout matter. A low delayed exact score cannot be assigned to decay or low dose alone.

## What was actually run

Canonical5,177 cells and9compartments, native `gated=False` equations from the published v5.1 engine. Random target-independent tanh-normal88-channel signals, threshold centering on256 disjoint random frames, unchanged actual PN/B wiring and native side competition. Each code has184 active cells. Independent random position signals are an optimistic storage reference; they are **not** learned position representations. No arithmetic structure, channel partition by key, raw-record cache, Transformer training or new native rule was inserted.

Cycle1 executed2 technical record applications (one active, one passive); Cycle2 executed1024; Cycle3 executed816 (544active,272passive). Total**1,842 technical record applications /9,210native events**, including passive controls; **zero formal scientific jobs**, zero host optimizer updates, zero addition runs. Five events per four-byte-plus-EOS observation. The runs' timed diagnostic bodies used8.24CPU seconds total (imports, artifact reduction and delivery overhead excluded); peak reported process RSS below280MiB. These are microdiagnostics, not end-to-end hybrid cost estimates. Seeds202610226101–103 are development data, never confirmation evidence.

Queries read a private BEGIN-reset native snapshot without teaching or inter-slot ticks. Cycle1 verified persistent-state readonly and passive zero storage. Raw nine-slot proposals and all margins are saved; a ninth non-EOS proposal is a truncation failure, not an allowed ninth payload byte in the prospective8-byte host API. The future host must enforce that output cap explicitly. There is no real host decoder in these tests.

## Audit and limits

[Artifact reducer](reduce.py) recomputes outputs, exact scores, bit accuracy and EOS diagnostics without importing native learning or rerunning trajectories. It verifies each cycle's locked runner and design hashes and record counts. An initial reducer syntax error was fixed; its log is preserved, and no model run was repeated. These cycles were self-reviewed; the separate reducer process is not an independent reviewer. Scalar engine parity inherits published prior qualification rather than being revalidated here. The full signed cross-write matrix and actual learned-address qualification remain pending.

Only1seed per cycle and one canonical anatomy, with correlated checkpoints/controls: no significance claims or general capacity estimate. An exact-key table can solve the task after one observation, so successful E1 storage would still not establish arithmetic-rule learning or engineering advantage. Strong bit signal does justify investigating a decoder, but it does not authorize a large campaign or guarantee a hybrid can reconstruct the values.

See [postcycle revised design](REVISED_DESIGN.md), [raw/derived summary](SUMMARY.json), [artifact audit](FINAL_AUDIT.json) and the per-cycle plans/reviews. Frozen parent design/results and arithmetic pointer are unchanged. No GitHub publication or formal run was launched.
