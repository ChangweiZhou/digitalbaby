# Memo: representation and generalization, round 2 (2026-11-03)

**Owner ruling applied:** anything is allowed except backpropagation and attention, including attention-equivalent content-addressed reads.

**Scope of the evidence:**
- Every result is a local development run on synthetic streams: 1 seed for the store runs, 6 seeds for the byte-relation runs.
- These are `BOUNDED_DEV_POSITIVE` or failure results, not sealed E2/E3 confirmations.

**Files:** `minifly_review_experiments/rep_gen_20261103/`
- Code: `rep.py`, `bytecyc.py`, `bytecyc2.py`.
- Results: `L256_*.json`, `bytecyc_results.json`, `bytecyc_persistence.json`.

---

## 1. Representation: a teaching channel solves the write-allocation half

**Setup:**
- The ruling permits an environmental "teach" marker. The association store therefore writes only on bytes the environment marks as answers. This is the fly's dopamine-gated KC→MBON plasticity.
- Front end: fixed lag code plus a position-free hashed byte bigram/trigram code.
- Store: a 16,384-unit sparse superposed store, 819 active units, local delta rule.
- Load: 256 facts in 4 waves.
- Probes: untouched wave-A keys after wave D, in renderings never seen in training.

| writes | trained form | new header/sep | key shifted 1 byte | shifted 2 | new wrapper | key moved left |
|---|---|---|---|---|---|---|
| **teach-gated** | **100%** | **96.9%** | **90.6%** | **84.4%** | **93.8%** | **92.2%** |
| random writes, same rate (1/14) | 70.3% | 43.8% | 0% | 0% | 1.6% | 0% |
| all bytes (load 64, earlier) | 100% | 100% | 0% | 0% | 63% | 31% |

**Reading:**
- An invariant code combined with environment-gated writes gives position- and wrapper-tolerant recall of 256 facts that survives three interference waves.
- The rate-matched random control fails, so the effect is *which* events write, not *how many*.

**Caveat:** the association store predicts only answers; whole-stream bits/byte for this store alone is 7.7. A deployed learner pairs it with a separate all-byte predictor, as in the dual-store design. That pairing is not yet measured here.

---

## 2. Generalization: cyclic relations transfer from a locally learned byte-successor map

The earlier symbolic result has been moved into bytes.

**Stream:**
- Prose-like background runs of letters `a`–`l` in cyclic order, e.g. `efghij kl…`.
- Interleaved with taught task records `cmp xy=` followed by an answer of `>` or `<`, then a newline.

**Learner (all local):**
- A TD(0) successor map over bytes, learned from the whole stream.
- A fixed sparse KC expansion (2,000 units, 100 active) of [SR(byte at lag 3), SR(byte at lag 2)].
- A teach-gated delta readout over 256 outputs.
- A bigram count model that emits the end-of-record byte.

**Scoring:** the model's own answer byte, and the answer followed by its own newline/EOS.

| holdout (6 seeds) | SR features: answer | SR: answer + own EOS | one-hot byte features | SR from shuffled-cycle stream |
|---|---|---|---|---|
| random 50% of pairs | 77.5% (min 70) | 77.5% | 36.4% | 43.9% |
| **whole distances 2 and 10 never taught** | **97.2%** (min 91.7) | 97.2% | 54.9% | 53.5% |
| **letter `f` never taught in any pair** | **83.3%** (min 70) | 83.3% | 48.3% | 47.5% |

All arms reach about 100% on taught pairs. The never-taught transfer comes only from the structure-bearing successor map, so the controls show it is not lookup or byte statistics.

### Persistence across relations

**Setup:**
- Second half of life: teach a second relation `par xy=` (same/different parity) on the same letters.
- The operator byte routes writes to its own KC population, a compartment-like split.

**Results:**
- **cmp is retained:** taught pairs 96.2%, never-taught distances 97.2% after par training. There is essentially no loss.
- **par does not transfer: 2.1% on held-out distances.** This is systematically inverted, not chance.
  - The successor map of a cycle has little power at the alternating (period-2) frequency.
  - Parity of a held-out distance is not linearly available.

**Lesson:** generalization appears exactly where the *representation* already contains the relevant structure, and not elsewhere. A single fixed representation will not cover every relation.

---

## 3. Where this leaves the three stages

| stage | mechanism now supported (development level) | open |
|---|---|---|
| memorization | sparse superposed store; capacity grows with units | size contract; int8 storage |
| representation | lag + n-gram invariant code **plus teach-gated association writes**: 256 facts, shift/wrapper recall 84–97% | pair with an all-byte predictor and measure whole-stream bpb; prose wrappers; larger loads |
| generalization | locally learned successor map + KC conjunction + teach-gated readout: **never-taught relation transfer with own answer + EOS**; persistence across two relations | relations not expressed by the current representation (parity: 2%); multi-step composition; arithmetic |

---

## 4. Next ideas (all within the rules)

1. **Richer, still-local representations**, to close the parity-type failures.
   - Use several successor maps at different discounts γ, plus maps over *skip-transitions* (x→x+2) learned locally.
   - This gives multiple "frequencies" of structure, so a single KC conjunction can express more relations.
   - **Test:** parity and "two steps ahead" on held-out distances.
   - **Falsifier:** still at chance. Then representations must be learned from task feedback itself, e.g. local Hebbian updates gated by prediction error over the conj features.
2. **Composition without attention.**
   - Chain modules: the output byte of module 1 is fed back into the stream and becomes the input of module 2.
   - First target: "count-on" addition over digits, where `a+b` means b successor steps from a. Read it out via repeated successor-map application, or via an iterated loop that emits intermediate bytes.
   - Run an engineered positive control first, plus a shortcut ladder (operand-marginal, nearest-neighbour, frequency).
3. **Integrated learner.** Combine in one byte stream:
   - an all-byte predictor;
   - the teach-gated association store over the invariant code (facts);
   - the successor-map conjunction head (relations).
   
   Run the A→B→corrected-A lifetime with prose wrappers. Measure own byte/EOS answers, retention, revision, transfer and whole-stream bpb together.
4. **Scale and data checks:**
   - A capacity grid with the invariant code (it raises different-key overlap from 0.14 to 0.18).
   - A cleaner background stream versus a noisy one, where task records pollute the successor statistics.
   - Several seeds for every store run.

## 5. Confidence

- **Medium-high:** the teach-gating result. The contrast is large (90% vs 0% on shifted keys at a matched write rate), but it is 1 seed.
- **Medium:** the byte relational transfer. It holds over 6 seeds with clear controls, on a clean, purpose-built stream.
- **Not shown:**
  - natural text;
  - relations whose structure is absent from the successor map;
  - multi-step reasoning;
  - integration with whole-stream prediction.
- The lag-position role code assumes a fixed answer-cue position (`xy=`). Varying the format is part of idea 3.
