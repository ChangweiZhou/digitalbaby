# S0 explore after V148: why MiniFly sits below the unigram rung

**Status: DEV_NOT_EVIDENCE.** 25 October 2026. Research-loop stage S0 (explore, tier T0), following `.claude/skills/minifly-research-loop` (origin/claude/vibrant-cerf-sje726 at 66925b04).
* No learner run produced a score. Everything below is evaluator-side arithmetic on corpus counts and the fixed output codebook, plus one cost profile that records timing only.
* Only the dev texts were used (Bryant, Buster). The main text (Alice) was not touched, so it stays clean for later comparisons.
* About 0.3 CPU-hours in total, including reruns.

## 1. Knowledge-base status (`MFC kb status`)
* There are 28 open incidents. They concern earlier arithmetic deliveries, program-level practice and the kit's own held-out canaries; none concerns text prediction.
* There is no sealed G series yet.
* Findings relied on, both provisional:
  * F-UF-11: "The FE code barely encodes byte order."
  * F-UF-13: an external readout fits training bytes at 100% but held bytes at 5.39–9.13%.
* Nearest delivery: V148 (`minifly_v148_pretrain_probe_dev_20261025`, protocol lock d3a7f815). It is a legacy delivery with no ledger row and no family id. Its predictions came out P1 not as predicted (worse than bigram), P2 not as predicted (early text erased) and P3 as predicted.

## 2. Falsifier 1: what the fixed output code allows, before memory or context
**Readout.** MiniFly's readout is `logits = CODE @ (v - mean v) / 4` (`minifly_cls_dev_20261017/fastmem.py:47-48`). CODE is the fixed C16 code: 257 distinct 16-bit words, each with exactly eight 1-bits, with a nearest neighbour usually 2 bits away.
* This readout is a product of independent bits restricted to the 257 code words.
* It can push one symbol's probability towards 1, by scaling the values along that symbol's code word.
* It cannot spread probability over several symbols without also giving it to the other code words that share their bits.
* Text needs spread-out predictions: after "th", several letters are likely.

`readout_ceiling.py` computes, from counts alone, the best values any memory could store. These are maximum-likelihood vectors for this readout, with no context or with contexts of 1–3 previous bytes. Every number in the table is computed in this folder; the files are listed at the end. Bits per symbol on held-out text:

| Context | C16 readout, best possible values (Bryant 40 KB / Buster 10 KB) | Free readout = counting ladder (Bryant / Buster) |
|---|---:|---:|
| none | **7.04** / 7.04 | 4.53 / 4.47 (unigram) |
| 1 previous byte | 6.10 / 5.93 | 3.71 / 3.54 (bigram) |
| 2 previous bytes | 5.50 / 5.24 | 3.36 / 3.13 (PPM-2) |
| 3 previous bytes | 5.03 / 5.15 | 3.20 / 3.01 (PPM-3) |
| *strict bound, fitted on the held-out text itself:* none | 7.03 / 7.03 | entropy 4.49 / 4.39 |
| *strict bound, fitted on the held-out text itself:* 1 byte | 5.88 / 5.58 | conditional entropy 3.43 / 3.18 |

**Comparison with what V148 measured** (Bryant: `../minifly_v148_pretrain_probe_dev_20261025/results/c3_rehearsal/SUMMARY.md`; Buster: `../minifly_v148_pretrain_probe_dev_20261025/results/c2_dev_passes/SUMMARY.md`):
* At 40 KB, MiniFly scored 7.03 held-out on Bryant. The best context-free vector scores 7.04.
* MiniFly put 0.605 of its probability on never-seen symbols. That vector puts 0.602 there.
* On Buster at 10 KB the figures are 6.95 against 7.04, and 0.659 against 0.635.
* Renormalised over seen symbols, MiniFly scored 5.47; the context-free vector scores 5.62.

So MiniFly sits at, or a little below, what a perfect memory with *no* context would achieve through this readout. The never-seen mass and most of the gap to unigram follow from the output code alone.

**Consequence for design section 9.** The planned next steps (9a raw trace, 9b self-supervised layer, 9c replay) change what reaches the readout, not the readout.
* Through C16, even perfectly used 3-byte context reaches only about 5.0 held-out with the best estimator computed here. That is above unigram on both texts.
* With no context or 1 byte, the strict bounds say the same without any estimator choice.
* None of the three can meet section 9's criterion (beat the best counting rung) while the C16 port stays.
* Caveat: contexts longer than 3 bytes were not computed. In the free readout, PPM-4 and PPM-5 add almost nothing over PPM-3 on these texts (rehearsal: 3.198, 3.191, 3.230).

## 3. Falsifier 2: does a wider output fix it? Only together with the teaching target
`code_width.py` repeats the context-free and previous-byte ceilings for random balanced codes of 16–128 bits (two random codes per width). It also computes where MiniFly's own delta rule settles. That rule teaches compartment j with (tanh(z_j/2) − y_j)/2 (`fastmem.py:160-163`), so it settles at tanh(z_j/2) = the mean target bit, each bit separately. One temperature is fitted on calibration. Bryant, held-out:

| Output port | Best possible values: none / 1 byte | Delta rule's fixed point: none / 1 byte |
|---|---:|---:|
| C16 (real) | 7.04 / 6.10 | 7.13 / 6.34 |
| 32 bits | 6.58 / 5.31–5.40 | 6.91–6.94 / 6.34–6.35 |
| 64 bits | 5.22–5.55 / 4.18–4.40 | 6.50–6.69 / 6.04–6.26 |
| 128 bits | 4.52 / 3.72 | 6.44–6.46 / 6.07–6.16 |
| one compartment per symbol (257), `onehot.py` | 4.52 / 3.74 | 4.54–5.04 / 3.79–4.37 |

**Two separate limits:**
1. **The code width limits what can be expressed.** The texts use 65 distinct symbols, and 44 cover 99% of the taught text. A bit code needs roughly that many independent dimensions; 128 bits reaches the counting rungs.
2. **The per-bit teaching target settles in the wrong place for a wide code.** Each compartment learns its own bit's mean independently, while the readout needs the bits fitted jointly. At 128 bits the rule's fixed point is still 6.4. Two things avoid this:
   * A teaching error taken from the readout's own 257-way output: (expected code bit under the current output) − (target bit). Its fixed point is exactly the best-possible column.
   * A one-compartment-per-symbol port, where each compartment's bit is "this symbol next". Its log-odds approximate log-probabilities, so the current rule settles near the rungs.

The range 4.54–5.04 for the one-hot port depends on how strongly a stored value can say "this symbol never comes" (clip 1e-4 to 1e-2). Whether the native stores have that range is untested.

## 4. Cost (`cost_profile.py`, timing only)
On 60 Buster records, 93% of the time per teaching event is per compartment: the native teach call plus the per-compartment readout. Encoding the context once is 3.5%, and the rest is other overhead. Cost therefore scales almost linearly with the number of compartments: about 0.07 + 0.93 × (compartments / 16) times today's cost.
* Today, with 16 compartments: 0.036 CPU-s per event and 0.012 per read symbol.
* 128 compartments (about 7.5×): about 0.27 per event and 0.09 per read symbol.
* 257 compartments (about 15×): about 0.54 per event and 0.19 per read symbol.

These are linear estimates, not measurements.

## 5. Options (neutral, each with cost and distance from the question V148 asked)

| | Option | What the analysis says it can reach | Cost | Distance from the question |
|---|---|---|---|---|
| A | One compartment per symbol (257), native per-bit delta rule unchanged | Fixed point 4.54–5.04 (no context), 3.79–4.37 (1 byte), against 4.53 / 3.71; depends on the native value range | about 15× per event. A 10 KB one-pass history with one evaluation (2 KB calibration, 4 KB held-out, two 5 KB taught slices) is about 2.3 CPU-h. 40 KB with one pass and one evaluation is about 7.5 CPU-h. The full V148 protocol (passes 1, 2 and 4, seven evaluations) is about 33 CPU-h per text | Same question and metric as V148, one component changed (the output port) |
| B | 128-bit code plus teaching from the readout's own 257-way output | Fixed point = best possible: 4.52 / 3.72 | about 7.5× per event | Two components changed (port and teaching signal). The new teaching signal is the gradient of the readout's log-likelihood with respect to the 128 output values: one layer, no hidden-layer credit assignment. Whether that is acceptable under "no back-propagation" is the owner's decision |
| C | Keep C16 and continue design section 9 (9a, 9b, 9c) | Bounded by section 2: about 5.0 even with perfect 3-byte context (strict: at least 7.03 with no context, at least 5.58 with 1 byte); cannot meet section 9's criterion | as designed, about 2–3 CPU-h per step | Same question, but its success criterion is unreachable |
| D | Keep C16 and measure what a bit code can express: exact recall or exact continuation of taught lines | Not computed here | low | Leaves noisy-text prediction, the reason V148 was run |
| E | Stop the text line here and decide the next direction in discussion | — | none | — |

**Recommendation (shown separately from the options).** A is the smallest change that the analysis says could reach the counting rungs: one component, with the native rule unchanged.
* Its own cheapest falsifier comes before any full run: a 257-compartment memory on about 2 KB of dev text, about 0.5 CPU-h. It reads two things: how negative the stored values for never-taught symbols become (the range condition above), and held bits/byte against unigram.
* A 40 KB history exceeds the default grant (2 CPU-h, 5,000 records). Any run beyond the falsifier needs a grant with numbers in Mr. Zhou's own words.

## 6. What this does not show
* **The ceilings are bounds, not predictions.** They assume the values for each context can be set independently. The native memory stores by similarity of KC codes, so it can stay above (worse than) these ceilings.
* **The bounds cover only the context definitions computed.** The strict ones (fitted on the held-out text itself) cover no context and 1 byte; the estimator ceilings cover up to 3 bytes. MiniFly's own context is a decaying trace of the whole line.
* Everything is on dev texts. The cost figures are linear extrapolations from one profile.

## 7. Proposals for the knowledge base (for the other agent family to check and activate)
* **Candidate finding.** Through the fixed C16 output code, held-out bits/byte on English text cannot go below 7.03 with no context or 5.58–5.88 with one previous byte, even with values fitted on the held-out text itself (dev texts Bryant and Buster). MiniFly's V148 dev results sit at the context-free bound.
* **Candidate incident, FM-16 ("no check of whether the next stage works on a perfect upstream").** V148 registered P1 ("between bigram and PPM-3") without computing the readout ceiling. That ceiling was computable from the codebook at design time and implies P1 could not hold. The V148 measurement itself stands as reported.

## Files
* `readout_ceiling.py` produces `readout_ceiling_{bryant,buster}.json`, with logs `bryant.log` and `buster.log`.
* `code_width.py` produces `code_width_bryant.json` and `.log`.
* `onehot.py` produces `onehot_bryant.json` and `.log`.
* `cost_profile.py` produces `cost_profile.log`.
* All scripts run from this folder with the V148 venv. They import only `corpus.py` and `ladder.py` from the V148 folder, plus the codebook JSON; the profile alone also imports `probe.py` to run the memory for timing.
