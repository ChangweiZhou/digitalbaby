# Independent audit of completed V42 results

**Verdict: PASS for result integrity, fixed-history reconstruction, arithmetic, and the declared scope. No numerical or reporting blocker found.** This is an engineering and interpretation verdict; the protocol defines no new scientific qualification gate or confirmatory claim.

Source SHA-256: `7f50146e0e8980db8dc7ff461a4185b12cdfd9f02111de7e171a51bb340b8dce`.

## Coverage and independent reconstruction

- All four locked histories, seeds 41101–41104, have completed result envelopes. Their checksums, plan identities, source digest, protocol digest, and reference-input digests agree.
- Every completed record reports **zero training updates and zero new stream presentations**. The histories contain 15,344 byte presentations each, totaling **61,376 reused historical presentations**. The execution record reports 2,436 newly processed evaluation units; these are aggregate/probe evaluation units, not training examples.
- Independently recomputed the online counts, correct counts, loss sums, and authority-read counts for every policy and each whole/target/nontarget scope directly from the original V41 common-stratum groups and the locked five masks. All match the completed records.
- Independently reconstructed all **12,000 policy probe rows** from their original historical or terminal READ-HYBRID/SUPER partners and mask-selected sources. Every correctness and loss value matches.
- Checked all **980 per-seed metric rows**, all **245 paired-contrast rows**, four-seed means, Student interval endpoints, and exact sign-reversal calculations. Counts, source values, pairing, and reported statistics agree.
- Baseline reconstructions all pass. The disjoint-mask loss identity holds in every seed and byte scope to floating-point roundoff.

## Primary measured effects

Differences are in bits per whole-stream byte; a negative difference favors the first policy. Intervals are exploratory paired Student 95% intervals with three degrees of freedom.

| Contrast | Four seed differences (41101, 41102, 41103, 41104) | Mean difference | 95% interval | Exact sign-reversal p |
|---|---|---:|---|---:|
| SLOT-ONLY minus READ-HYBRID | −0.05628336, −0.05576041, −0.06723181, −0.05109688 | −0.05759311 | [−0.06847022, −0.04671601] | 0.125 |
| SLOT-ONLY minus SUPER | +0.01331666, +0.05579943, +0.04208064, +0.08474055 | +0.04898432 | [+0.00173815, +0.09623049] | 0.125 |

The corresponding parametric two-sided Student p-values are 0.00045513 and 0.04574971. The report correctly distinguishes these assumption-dependent values from the four-seed sign-reversal resolution and from independent confirmation.

SLOT-ONLY has lower whole-stream loss than READ-HYBRID in each reused history, but higher loss than SUPER in each history. The observed benefit is therefore a measured tradeoff, not a universal policy win.

| Endpoint, equal-weight mean over four seeds | READ-HYBRID | SLOT-ONLY | ANCHOR-ONLY | SUPER |
|---|---:|---:|---:|---:|
| Whole-stream bits/byte | 2.43964989 | 2.38205677 | 2.32017612 | 2.33307245 |
| Old seen recall | 98.4375% | 98.4375% | 98.4375% | 76.5625% |
| Old held-out recall | 98.4375% | 98.4375% | 70.3125% | 70.3125% |
| Same identity: E1 and all delayed probes correct | 98.4375% | 98.4375% | 67.1875% | 67.1875% |
| Final revised-label held-out recall | 100% | 100% | 90.625% | 90.625% |

All five policies have 100% measured E1, E2, and final current-cohort seen accuracy. These endpoints do not distinguish them here. The old held-out and joint endpoints each use 16 identities per seed; old seen recall uses 32 contexts per seed. Final revised-label held-out recall uses eight contexts per seed.

The equality of SLOT-ONLY and READ-HYBRID on certified never-trained held-out contexts follows structurally from disabling only exact-trace authority. The report identifies this limitation. The seen-context equality above is a measured terminal result and does not recover unlogged intermediate seen-context behavior.

## Attribution and denominator correction

Independently recomputed every pooled numerator from the seed stratum effects. Every pooled whole-byte denominator is **61,376**, including the bytes of any completed history with zero events in a category. The current observed categories happen to occur in all four histories; the separate engineering regression covers an absent-category history.

For each of READ-HYBRID minus SUPER and SLOT-ONLY minus READ-HYBRID:

1. Category event counts sum to all 61,376 historical bytes.
2. Category contributions sum to the independently reconstructed whole-policy loss difference.
3. Per-affected-event changes divide by category events; whole-stream contributions divide by all historical bytes.

Removing exact-trace authority improves whole-stream loss by approximately 0.057337 bits/byte from ordinary-byte reads and 0.000256 bits/byte from target-byte reads, totaling 0.057593 bits/byte. Supported-similarity ordinary-byte reads retain an approximately 0.061881 bits/byte cost relative to SUPER while preserving the stronger held-out memory endpoints. Exact consolidated-slot target reads contribute approximately −0.013032 bits/byte relative to SUPER. The report's attribution table agrees with these quantities.

## Archive and execution checks

The audited results ZIP passes CRC verification, exact manifest coverage, and every member SHA-256 check. Its snapshot at audit time had 30 members and 865,630 bytes, with SHA-256 `01030cb413f456754fc558c57f263e74b5a2f5feb9af8149e7a965513811ddf0`. Completed checkpoint payloads are omitted from that compact archive. Later report-only or completed-run verification may legitimately add invocation records and replace the archive; this digest identifies the snapshot inspected here, not a requirement to preserve incidental invocation metadata.

The completed scientific execution reports peak process RSS of 151,744,512 bytes, approximately 144.7 MiB, with no stop request. Actual SIGINT/SIGKILL recovery, checksum failures, concurrent-writer exclusion, and completed-run zero work were separately verified against this final source before the audit execution.

The evidence supports the declared post-hoc fixed-history conclusion. It does not establish new-stream learning behavior, reduced state storage, standalone runtime savings, or fresh statistical replication.
