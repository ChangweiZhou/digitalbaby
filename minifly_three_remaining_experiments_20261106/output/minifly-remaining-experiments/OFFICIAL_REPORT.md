# MiniFly official results for the three remaining experiments

**November 6, 2026 · Development campaigns**

The three protocols tested frozen encoder initialization, downscaling during rest, and attractor topology with worked counting. Results belong to separate learners and tasks. [Locked design](../minifly-whole-repo-audit/MINIFLY_THREE_REMAINING_EXPERIMENTS_DESIGN_20261106.md) · [Engineering cycles](CYCLES.md)

## Frozen pretrained encoder

All 48 local histories finished. They consumed 196,608 graph episodes and 10,813,440 delivered symbols. The six trained encoder births consumed 6,291,456 next-byte targets. All frozen encoder tensor checks passed during life.

| Encoder | Literal taught A | Held B before teaching | Held B after 2,048 | Correct first byte at B end | EOS at B end | Consistent A B area | Mixed A B area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 0/384 | 0/384 | 0/384 | 5/384 | 19/384 | 0.00% | 0.00% |
| shuffled | 0/384 | 0/384 | 0/384 | 0/384 | 0/384 | 0.00% | 0.00% |
| random | 0/384 | 0/384 | 2/384 | 2/384 | 120/384 | 0.20% | 0.20% |
| stock | 10/384 | 0/384 | 3/384 | 13/384 | 220/384 | 0.29% | 0.29% |

Counts pool the six consistent histories per encoder. The best final B score of a single consistent history was 2/64. **0/24** consistent histories reached the literal taught floor of 52/64. The forward-transfer gate failed. The local acquisition scores govern interpretation of the B curves. [Encoder summary](official/encoder/SUMMARY.json) · [Encoder lock](official/encoder/LOCK.json)

## Mod 97 downscaling

3/3 seeds exceeded the 50% early-transfer ceiling at the common plateau. Early-transfer seeds remained in the matched shrinkage comparison and were excluded from the delayed-rise test. Their early warmup checkpoint was recovered by deterministic continuation and checked against the original taught and monitoring receipts. All sealed scores were generated after the branch checkpoints were frozen.

| Seed | Fork update | Taught at fork | Monitoring held at fork | Sealed at fork | No shrinkage | Periodic | Smooth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 73129 | 1,536 | 4681/4704 | 84.7% | 2008/2354 (85.3%) | 2130/2354 (90.5%) | 2021/2354 (85.9%) | 2034/2354 (86.4%) |
| 93683 | 1,536 | 4692/4704 | 88.0% | 2097/2356 (89.0%) | 2204/2356 (93.5%) | 2069/2356 (87.8%) | 2079/2356 (88.2%) |
| 116089 | 1,536 | 4669/4704 | 82.4% | 1912/2355 (81.2%) | 2063/2355 (87.6%) | 1910/2355 (81.1%) | 1920/2355 (81.5%) |

The no-shrinkage arms reached **87.6%–93.5%** exact on the sealed split. Both shrinkage schedules scored below no shrinkage in 3/3 seeds. In the no-shrinkage arms, the shared path alone reached 86.3%–91.6%; the associative residual alone reached 0.9%–1.6%.

Taught and held operands were split by unordered pair, so the swapped form of a held pair was also withheld. Exact accuracy requires two correct bytes and native EOS. The pathway diagnostic temporarily muted either the shared path or the associative residual. Its scores, raw native sealed outputs and shortcut ceilings are in the receipts. The conditional shuffled-teaching comparison was tied to a positive delayed-rise result. Interpret the early transfer as held behavior on this finite task with its own evidence boundary. [Mod 97 summary](official/mod97/SUMMARY.json) · [Path and shortcut audit](official/mod97/PATH_AND_SHORTCUT_AUDIT.json)

## Attractor and worked counting

All 18 histories completed 16,384 groups. Each group contained a count, addition and auxiliary record. The roster processed 294,912 groups and 17,104,896 delivered symbols. The behavioral gate required at least 60/75 taught and 20/25 held in each seed.

| Topology | Teaching | Taught exact out of 75 | Held exact out of 25 | Held native EOS out of 25 | Stable localized supports |
| --- | --- | --- | --- | --- | --- |
| ring | answer | 75 · 75 · 75 | 0 · 0 · 0 | 25 · 25 · 25 | 5 · 0 · 8 |
| ring | shuffled | 75 · 75 · 75 | 0 · 0 · 0 | 24 · 24 · 24 | 0 · 0 · 0 |
| ring | worked | 75 · 75 · 75 | 0 · 0 · 0 | 25 · 25 · 25 | 3 · 5 · 2 |
| rewired | answer | 75 · 75 · 75 | 0 · 0 · 0 | 25 · 25 · 25 | 0 · 0 · 0 |
| rewired | shuffled | 75 · 75 · 75 | 0 · 0 · 0 | 24 · 24 · 24 | 0 · 0 · 0 |
| rewired | worked | 75 · 75 · 75 | 0 · 0 · 0 | 25 · 25 · 25 | 0 · 0 · 0 |

The highest held score was **0/25**. Values are in seed order 73129, 93683, 116089. The count-only control and B continuation were conditional on the behavioral gate. Stable localized states and the tally-pulse and readout-muting diagnostics describe the mechanism separately from exact addition. [Attractor summary](official/attractor/SUMMARY.json) · [Mechanism and gate receipt](official/attractor/DECISION.json)

## Accounting

Aggregate main worker CPU was 13.5 minutes for pretraining and local life, 35.4 for modular training and branch monitoring, and 134.4 for attractor training. Sealed postprocessing and separate diagnostics have their own receipts. [Receipt hashes](official/RECEIPT_HASHES.json) · [Reproduction commands](REPRODUCE_OFFICIAL.md)

## Published raw files

- [Frozen encoder raw receipts](official/encoder/RAW_RECEIPTS.zip)
- [Mod 97 raw receipts](official/mod97/RAW_RECEIPTS.zip)
- [Attractor raw receipts](official/attractor/RAW_RECEIPTS.zip)
- [Local checkpoint checksums](official/CHECKPOINT_HASHES.json)

The three receipt archives expand into the corresponding campaign directories under `official`.
