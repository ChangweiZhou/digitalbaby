# MiniFly ordered byte front end — development result

**The native reasoning screen failed.** After 100,000 A records, the ordered, stock, and order-removed arms each scored **0/32 held quartets in both blocks**. The ordered arm learned more repeated teaching prompts than stock and produced its own EOS on every held A prompt. These gains did not extend to reliable answers on new renderings of the taught topologies or on the two held topologies.

## Benchmark revision and its limits

The six-edge panel from the engineering cycles was replaced with two non-isomorphic eight-edge families. Both teaching and probe records used eight link rows. Every primary case paired prompts with the same byte counts, endpoint opcode counts, and final eight bytes. The new panel mixes cases where both laws flip the answer, cases where the designated route is identical between prompts, and cases where that route changes while one law's answer stays fixed. The graph oracle solved **32/32** and ruled out literal and canonical recitation as sources of held coverage.

The designated-route parity shortcut dropped to **6/32** and the preregistered oracle-role comparator with any fixed three-link subset reached **19/32**. The locked screen therefore required **23/32** in each block and four more than stock. After locking the run, I found a stronger evaluator that chooses later links based on earlier opcodes. With up to three adaptive queries it scored **30/32**, so a positive result on the selected quartets would have required further adjudication. The native zeroes and the separate exhaustive assignment probe give the negative result here. [Campaign lock](campaign/LOCK.json) · [Adaptive shortcut audit](campaign/ADAPTIVE_AUDIT.json)

## Native scores after 100,000 A records

| Birth | Encoder | Taught A exact | Held A first byte | Held A exact | Held A EOS | Held quartets at 10k → 100k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 68071 | stock | 78/128 | 4/128 | 3/128 | 84/128 | 0/32 → 0/32 |
| 68071 | ordered | 93/128 | 15/128 | 15/128 | 128/128 | 0/32 → 0/32 |
| 68071 | bag | 91/128 | 11/128 | 8/128 | 108/128 | 0/32 → 0/32 |
| 80413 | stock | 79/128 | 7/128 | 4/128 | 96/128 | 0/32 → 0/32 |
| 80413 | ordered | 94/128 | 10/128 | 10/128 | 128/128 | 0/32 → 0/32 |
| 80413 | bag | 87/128 | 15/128 | 13/128 | 104/128 | 0/32 → 0/32 |

Each taught and held exact cell pools 64 outputs per history. A joint item requires both prompts to be correct, with the model's own EOS, under both teaching histories. Correct first-byte and EOS counts are shown separately.

The ordered arm's strongest result was **187/256** exact on repeated taught prompts across the two blocks. This is E1 acquisition. The held primary scores did not establish E3 transfer. The matched inconsistent-teaching histories were conditional on a positive native screen and were not triggered. The two development births leave uncertainty about other curricula, update rules and seeds.

## Continuation and matched edit

| Birth | Encoder | B preserved after edit | B preserved after no edit | Previously correct flipped A corrected | Same corrections in no edit |
| --- | --- | ---: | ---: | ---: | ---: |
| 68071 | stock | 0/0 | 0/0 | 0/1 | 0/1 |
| 68071 | ordered | 0/8 | 0/8 | 1/8 | 1/8 |
| 68071 | bag | 0/7 | 0/7 | 0/4 | 0/4 |
| 80413 | stock | 0/0 | 0/0 | 0/0 | 0/0 |
| 80413 | ordered | 4/14 | 3/14 | 2/10 | 2/10 |
| 80413 | bag | 0/6 | 0/6 | 0/5 | 0/5 |

Retention uses the B items that were exact immediately before branching. Correction uses flipped A items that were exact immediately before branching. An empty denominator means the earlier stage supplied no correct responses on that subset. Across both ordered blocks, the edit branch corrected **3/18**, and the no-edit branch happened to produce the same new-law answer on **3/18**. These counts give no evidence of a correction advantage. [Per-history retention, corrections and obsolete answers](campaign/SUMMARY.json)

## Frozen-checkpoint diagnostics

| Birth | Encoder | Taught topology, familiar actors | Taught topology, fresh actors | Held topology, familiar actors | Held topology, fresh actors | Held quartets, familiar actors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 68071 | stock | 4/128 | 2/128 | 7/128 | 3/128 | 0/32 |
| 68071 | ordered | 12/128 | 13/128 | 14/128 | 15/128 | 0/32 |
| 68071 | bag | 13/128 | 7/128 | 7/128 | 8/128 | 0/32 |
| 80413 | stock | 6/128 | 1/128 | 4/128 | 4/128 | 0/32 |
| 80413 | ordered | 10/128 | 18/128 | 15/128 | 10/128 | 0/32 |
| 80413 | bag | 6/128 | 1/128 | 13/128 | 13/128 | 0/32 |

These are fresh, unreinforced byte renderings. The familiar-actor contrast changes only the actor byte vocabulary on the same held cases. The ordered arm gave 10–12/128 exact on new renderings of the taught topologies using familiar actors, compared with 93–94/128 on the literal teaching renderings. Familiar actor bytes did not yield a successful held quartet. This points to a failure of presentation reuse before the harder question of graph composition. [Vocabulary and topology diagnostic with raw outputs](campaign/VOCABULARY_DIAGNOSTIC.json.gz)

The all-assignment probe scores every one of the 256 opcode masks on each of the two held families. It weights correct move and stay responses equally and averages both opposite teaching histories. The strongest three-adaptive-query comparator reached **83.1%** and **89.3%** on these families. The full graph oracle gets 100%.

| Birth | Encoder | Family 1 balanced exact | Family 2 balanced exact |
| --- | --- | ---: | ---: |
| 68071 | stock | 2.9% | 2.4% |
| 68071 | ordered | 11.8% | 9.8% |
| 68071 | bag | 7.4% | 9.7% |
| 80413 | stock | 1.4% | 1.6% |
| 80413 | ordered | 12.5% | 12.6% |
| 80413 | bag | 4.2% | 5.1% |

The model scored 9.8–12.6% balanced exact with the ordered input. This check separates the observed negative result from the shortcut present in the selected quartets. Both supplements ran on frozen checkpoints after the primary protocol was locked. [All-assignment native outputs](campaign/ALL_MASK_MODEL_SCORES.json.gz)

## Run receipts

All **12 histories and their matched continuations** completed, covering **1,560,000 training updates and 85,800,000 predicted symbols**. The parallel main run used 76.7 aggregate CPU minutes and took 13.8 minutes for its longest worker. Peak observed RSS was 483 MiB for one worker. These figures exclude qualification and the supplementary read-only jobs. The cache preserved bitwise native update parity at preflight and did not persist learned state.

The development result does not support promoting the ordered input as a reasoning improvement or carrying it into the reasoning scaling matrix on this evidence. Independent review remains pending. [Locked design](CAMPAIGN_DESIGN.md) · [Main run summary](campaign/SUMMARY.json) · [Reproduction commands](REPRODUCE.md)
