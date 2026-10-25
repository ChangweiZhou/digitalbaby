# V148 — PRETRAIN-PROBE V1: what does MiniFly extract from a small amount of raw text?

25 October 2026. **Design document only. Nothing in it has been implemented or run.** It is to be refined in three cycles (refine, audit, test run, exercise) before a lock. Every number about earlier work cites its source; every prediction is written as three sentences (expectation, basis, what would weaken it).

## 1. Why this round, and why not arithmetic

Arithmetic was a probe of whether MiniFly can pre-train, that is, extract reusable structure from experience rather than store items. The probe did not separate the possible causes.
- Every route that could memorise the taught facts stayed near the copy baseline on held-out sums. This held for V143 mod-10, V144–V146, and V147's acquisition stop.
- The dev diagnostics in `minifly_root_cause_20261025/NOTE.md` found two effects:
  - per-item error learning showed no preference for the rule;
  - the front end's max-normalisation distorted an exactly additive trace.
- Codex's review rightly limits those diagnostics:
  - they used class outputs, not MiniFly's own output;
  - preprocessing used all 100 inputs;
  - a same-answer relational signal is a useful bias, not a necessary one;
  - several components changed between versions, so this was never one fly losing an ability.
- The relational teaching signal needs labelled "same answer" pairs that real, noisy data does not supply.

The program's long-term aim is learning from data, and real data is noisy. So this round measures the pre-training question directly, on small natural text. Arithmetic returns later as a downstream transfer test (section 10).

## 2. The question

> Trained by next-byte prediction on a small amount of raw English text, unchanged and with gradual warmup, how well does the existing native fast memory predict *held-out* text? How does that compare with simple counting predictors trained on the same bytes?

This round adds **no mechanism**. It establishes what MiniFly does now, under a metric that tolerates noise. It also sets the bar that any later component must clear on the same data.

## 3. System under test (unchanged)

`minifly_cls_dev_20261017/fastmem.py` `FastMemory`, with default settings:
- FE-PERM front end;
- fresh adaptation at each record;
- one-sided KC homeostasis;
- error-corrective (`delta`) native teaching with temperature 2;
- 16 canonical native Full151 compartments.

Its output is the fixed 257-class C16 code: `logits(values) = CODE · (values − mean) / √16`. That gives a native score for every byte and for EOS. No learned layer, no map, no relational rule and no other new component is added.

The source and native assets are pinned by hash, from the runtime closure computed with `sys.modules` plus the loaded assets. The front end's max-normalisation (`read() = p / max(p)`) is kept as it is. Changing it is a later, separate step (section 9).

## 4. Data

**Corpus.** One public-domain English text from Project Gutenberg. The proposal is *Alice's Adventures in Wonderland* (about 150 KB of plain text). It is natural, irregular, small, and has reusable regularities: letter order, frequent words, spacing and punctuation.
- The Gutenberg header and licence footer are removed by fixed line markers.
- The bytes are kept as they are (ASCII; any non-ASCII bytes stay as bytes).
- The file is pinned by SHA-256. If the download is unavailable, cycle 1 records the reason and uses another named public-domain text; no synthetic text is substituted.

**Records.** Each line of text (split on newline) is one record. Lines longer than 128 bytes are cut into 128-byte pieces. Empty lines are skipped.
- A record is `train(prompt=b"", answer=line)`. FastMemory therefore teaches every byte from the context of the preceding bytes in the line, then EOS at the line's end.
- EOS stands for "line ends here". This is the model's own stopping decision, with no external length.
- The front end and adaptation reset at each record, as the memory was built to do.

**Split.** Contiguous blocks, never interleaved, so that held-out text is genuinely unseen:

| Block | Position in the text | Size | Use |
|---|---|---|---|
| Train pool | from the start | first 40 KB of record bytes (nested prefixes of 5, 10, 20 and 40 KB) | teaching |
| Calibration | immediately after the train pool | 5 KB | one temperature only (section 5); never taught |
| Held-out test | the final chapter(s), at the end of the text | 10 KB | evaluation only; never taught |

Small data is intended. The largest training set is 40 KB, about 40,000 byte events. If the cost measured in cycle 1 makes 40 KB too expensive, the largest prefix is reduced *before* any held-out score is computed, and the change is recorded.

**Development text.** A different public-domain book, about 15 KB, is used for cycles 1–3. The test text is not scored until the lock.

## 5. Measurements

All evaluation runs on clones. The memory's state hash must be identical before and after every probe, and held-out and calibration text is never taught.

**Primary metric: held-out bits per byte.**
- For each held-out byte, the memory reads its own output for the current context exactly as at recall. It converts the 257 native logits to probabilities with a softmax at temperature T, and records −log₂ p(true next symbol). That symbol is the next byte, or EOS at the line end.
- The context is the actual preceding text bytes of that line (teacher forcing), which is standard for measuring prediction.
- T is a single scalar fitted to minimise bits/byte on the calibration block, with the memory frozen. The uncalibrated value at T = 1 is also reported, and so is T itself.

**Secondary metrics.**
1. **Top-1 next-symbol accuracy** on held-out text. It needs no calibration.
2. **Learning curve.** Held-out bits/byte and accuracy after the 5, 10, 20 and 40 KB prefixes, in one continuous training history with no restarts.
3. **Memorisation gap.** Bits/byte on the *taught* text (first and last 5 KB of the train pool, read after training) against held-out. This separates storing from generalising and shows retention across the training stream.
4. **Own generation.** For 50 held-out line prefixes (the first 10 bytes of each), the memory's greedy own continuation up to its own EOS or 64 bytes. These are reported as examples with descriptive statistics only: the fraction of real words, line length, and repetition. This is not a scored endpoint.
5. **Cost.** CPU seconds per byte event, peak memory, and the total including failed attempts.

## 6. The reference ladder (simple counting predictors)

Bits/byte means little on its own, so MiniFly is compared with the simplest predictors that use the same training bytes and the same 257-symbol alphabet (256 bytes plus EOS at line end):

| Reference | How it predicts the next symbol | What beating it would show |
|---|---|---|
| Uniform | 1/257 for every symbol | MiniFly learned anything |
| Unigram | symbol frequency in training (add-one) | MiniFly uses context |
| Bigram | counts of what followed the previous byte (interpolated with unigram) | MiniFly uses more than one byte of context |
| PPM-C, orders 2, 3, 4, 5 | counts after the last k bytes of the line, with escape to shorter contexts (method C) | MiniFly extracts structure beyond counting recent contexts |

- These are plain counting rules. They are evaluator-side measuring sticks, not learners, and they never touch MiniFly.
- They are trained on exactly the same record bytes and frozen at test, like MiniFly. They reset at line starts, like MiniFly's records.
- Each is checked in cycle 1 against hand-computed probabilities on a toy string.

## 7. Predictions (three sentences each)

1. **Held-out level.**
   - *Expect:* held-out bits/byte between the bigram and PPM order-3 rungs, and not better than order 3.
   - *Because:* the stored codes are fixed random expansions of a decaying, max-normalised recent-context trace, and each teaching event writes an association for one context.
   - *Weakened if:* MiniFly beats PPM order 3 or above on held-out text at the same training size.
2. **Memorisation gap.**
   - *Expect:* bits/byte on taught text clearly below held-out.
   - *Because:* the error-corrective native write stores context-to-byte associations, and G1 showed 84–88% taught recall on byte-stream facts (`minifly_cls_dev_20261017/REPORT.md`, G1).
   - *Weakened if:* taught and held-out bits/byte are close, meaning little is memorised, or interference erases early text.
3. **Learning curve.**
   - *Expect:* held-out bits/byte improves from 5 to 40 KB by less than the PPM rungs improve.
   - *Because:* similarity-based recall generalises only through overlap of context codes, and that overlap does not grow with data the way context counts do.
   - *Weakened if:* MiniFly's curve is as steep as PPM's or steeper.

These are written before any code. A prediction that fails is reported as failed.

## 8. Three cycles before the lock

1. **Implement and audit.**
   - Build the corpus loader with header and footer removal, the SHA-256 pins, the split and the record builder.
   - Build the probe: clones, the 257-way softmax, and calibration of T on a 1-D grid.
   - Build the four-rung reference ladder.
   - Unit checks:
     - probes leave the state hash unchanged;
     - softmax probabilities sum to 1;
     - the ladder reproduces hand-computed toy values;
     - the held-out and calibration blocks never reach `train`;
     - the record builder is deterministic;
     - the runtime closure is pinned.
   - Measure cost per byte event on 1 KB of the dev text.
2. **Dev test run.** The full measurement on the dev text at reduced size. If cost requires it, fix the largest prefix and the generation sample size here. Check that the numbers make sense against the ladder. No scoring of the test text.
3. **Rehearsal and lock.** A dress rehearsal of the exact protocol on the dev text, then a lock of the source hashes, corpus hash, split offsets, prefixes and evaluation code. The locked run then evaluates the test text once.

**Reporting rules.**
- Failed attempts are counted.
- Results on the dev text are called development results, never confirmation.
- The report states the measured values and the ladder positions. It claims no capability beyond them.

## 9. What comes after this round (one step at a time; not part of V148)

Each later step changes one component and is measured on the same data, split and ladder. A component counts as an improvement only if it lowers held-out bits/byte against both the V148 baseline and the best ladder rung at the same training size.

- **9a.** The raw front-end trace in place of the max-normalised one. This is the only part of the arithmetic diagnostics that applies to any data.
- **9b.** A learned layer between trace and expansion, with a self-supervised local update: contexts that are followed by the same next byte move towards a shared code. This is the general, label-free form of the relational signal, and its value on real text is untested.
- **9c.** Replay from the fast store into a slow learner (consolidation). CLS v0 tested replay for retention, never for compression.

## 10. Arithmetic, later

Once some component lowers held-out bits/byte below the counting rungs, arithmetic returns as a **transfer test**. Does pre-training on text make later learning of a small task faster or more general than learning it from scratch? That makes arithmetic a downstream measurement, not the gate.

## 11. Scope

This round measures. It does not adopt anything, change the native core, or claim progress on the arithmetic goal. If the measurement fails, a negative or partial result is still the deliverable.
