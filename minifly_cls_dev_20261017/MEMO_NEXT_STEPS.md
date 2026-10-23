# Memo: next steps without backpropagation, and what human neuron research suggests (2026-10-18)

Discussion memo, no new runs. Builds on `REPORT.md` (MiniFly-CLS v0 gates) and
`addition_difficulty_diagnosis_dev_20261017/MEMO.md`. Project stance: no backpropagation as a learning mechanism
(its biological evidence is slim); propagating error through a few layers may be acceptable.

## 1. What v0 changed, in plain terms

**Kept (the native fly):** the canonical newborn flies and their input-to-memory-cell (PN→KC) wiring; the sparse
memory code (top ~3.5% of 5,177 cells); the native fast, slow and adaptation stores and their equations; the native
readout and native signed write event. The 16-fly output port (one bit each of a 16-bit code over 257 answers) came
from the earlier lineage comparison.

**Changed around the fly (no backprop):**

| Change | What it is | How native |
|---|---|---|
| Order-keeping front end | Replaces FE0; each new byte first shifts the old trace through a fixed shuffle of the 88 input channels; fixed, no learning | new front end |
| Memory-cell homeostasis | cells firing for almost every prompt get a higher threshold before top-k selection | new mechanism in the coding step (identical to native when off) |
| Adaptation reset | cell adaptation reset to "recovered" at the start of every record | protocol override of native dynamics; the most hack-like change |
| Error-correcting write | the teaching signal comes from the memory's own current answer; only wrong bits are written, through the native signed write | new teaching signal (the fly has output-to-dopamine-neuron feedback, but the native model does not) |

These four produced all of the memory gain (0–8% → 84–88% of taught facts; 81–85% after a day) and the order result
(100% vs 75% on order-contrast prompts).

**Added next to the fly (not MiniFly):** a 64-unit leaky recurrent network with its own output and silent thinking
steps, trained by backpropagation through time (reference) or e-prop (local approximation), with weight decay as
sleep-time scaling-down; a replay buffer of raw byte records outside the fly; and a rule choosing between the two
systems' answers.

**Where results came from:** memorisation and order from the fly plus the four changes; keeping addition while
learning multiplication from the replay buffer plus the fly's error-correcting write; generalisation to unseen sums
only from the recurrent network with backprop, and only 13%. e-prop: 0%. The fly contributed nothing to
generalisation, consistent with it being a lookup memory.

## 2. What our results imply for dropping backprop

e-prop failed for a specific reason: the two operands arrive several steps apart, and e-prop cannot carry credit back
through the recurrent connections that link them. But the order-keeping front end already presents the whole prompt
at once at "=" — that is why the memory recalled 100% of the order-contrast prompts.

So the slow core need not be a recurrent network trained through time. It can be a **one-step learner**: the
88-channel trace at "=" → one or two plastic layers → the answer. Credit then travels at most two layers, within the
"few layers" that may be acceptable. Local rules that work at that depth:

* **Three-factor rule:** Hebbian change × a broadcast teaching signal — what the mushroom body already does, applied
  one layer earlier.
* **Feedback alignment:** errors return through fixed random weights; no weight symmetry needed.
* **Instructive-signal plasticity:** as in hippocampus, where a dendritic plateau tells a neuron what to learn over
  seconds (behavioural-timescale plasticity).

Two-layer backprop would serve only as the yardstick.

**Unchanged risk.** Generalising needs a learned digit representation plus a pressure towards reuse (weight decay gave
+38 points in the diagnosis). Shallow networks on position-coded inputs do learn addition with weight decay in the
machine-learning literature, but this has not been tested here with local rules.

## 3. What human neuron research suggests

### 3.1 Arithmetic does not run on the language network

* Algebra and arithmetic do not engage the brain's language-selective network (Monti, Parsons & Osherson 2012;
  Amalric & Dehaene 2016; Fedorenko et al.).
* Varley et al. (2005, PNAS, "Agrammatic but numerate"): three men with severe agrammatic aphasia could not tell
  "the lion hunted the man" from "the man hunted the lion", yet handled 52 − 11 vs 11 − 52 and solved bracket
  equations; recursion and structure-dependent operations stayed intact.
* In humans, order and brackets for arithmetic come from a separate number and sequence system, not from grammar.
  This cuts against "learn arithmetic, then language, in the same core". A fairer framing: both need the same basic
  capacities (keeping order, building structure, holding a rule), possibly on separate substrates.

### 3.2 Numerals are mapped onto a magnitude code

* Single neurons in the human medial temporal lobe encode numbers (Kutter et al. 2018, Neuron; 585 neurons).
* Parietal cortex carries a magnitude code shared by dot quantities and digits (Piazza et al. 2007, Neuron).
* Monkeys trained to link arbitrary shapes with quantities: prefrontal neurons come to respond to a sign and its
  quantity alike (Diester & Nieder 2007, PLoS Biology). Symbol meaning is learned by reward-based association, without
  backprop, onto a magnitude code that already exists.
* This is in honest tension with the scalability concern about grounding: brains do ground numerals. The scalable
  middle ground is a magnitude code learned from symbol-only sequences (e.g. the count list), but the biology uses a
  pre-existing quantity system.

### 3.3 The operator is held as a separate, abstract rule

* Kutter et al. (2022, Current Biology) recorded single neurons in the human medial temporal lobe during addition and
  subtraction. Neurons coded the rule abstractly — the same whether shown as "+"/"−" or as words. The code was steady
  in hippocampus (persistently rule-selective neurons) and changed rapidly in parahippocampal cortex.
* For us: the operator should become a lasting context that selects how the operands are combined, not one more byte
  in a hashed memory key. This targets the G4 problem directly: addition and multiplication collided because they
  share the same digit cells (same-pair key overlap ~0.30).

### 3.4 Sentence structure is built step by step and closed in chunks

* Activity builds up word by word within sentences but not in word lists (Fedorenko et al. 2016, PNAS).
* It grows with the number of open phrases and drops when a phrase completes (Nelson et al. 2017, PNAS).
* Cortex tracks phrase and sentence rhythms even without acoustic cues (Ding et al. 2016, Nature Neuroscience).
* Cai, Williams, Cash and colleagues (Nature, June 2026, "Mapping the neuronal building blocks of human language with
  language models"): 579 single neurons in frontotemporal cortex recorded while people built sentences; some tracked
  parts of speech, others how words group into phrases and fit the sentence structure, up to a second before the word
  was spoken.
* For `(5+3)*2`: a working state that opens at `(`, closes at `)`, and replaces the bracketed part with its result.
  Our trace keeps order but cannot close chunks. Synaptic working memory (held in fast synapses rather than sustained
  firing; Mongillo, Barak & Tsodyks 2008) is a plausible substrate, and the fly's fast store is that kind of store.

## 4. Next steps (no code yet), in order

1. **One-step slow layer.** Replace the recurrent network with trace → 1–2 plastic layers → answer, trained by a
   three-factor or feedback-alignment rule plus scaling-down; two-layer backprop only as the yardstick. Re-run the
   unseen-sums gate (G3). Cheap; answers whether no-backprop generalisation is feasible at all.
2. **Operator as context.** Check whether operator-selective units emerge, as in human rule neurons; then test gating
   the plastic layer by them and measure addition/multiplication interference (G4).
3. **Order from symbol-only experience.** Add count-list streams; check whether the hidden layer develops number-line
   tuning (like number neurons) and whether that helps unseen sums.
4. **Chunking for brackets.** Only after 1–3: a fast-synapse working state holding an intermediate result, tested on
   `(a+b)*c` vs `a+b*c`.

Throughout: keep the mushroom body as the exact memory with replay into both systems; retire the backprop recurrent
network to a reference role.

**Open decision (affects step 3):** is a pre-existing quantity code acceptable as innate structure (the biological
route), or must order and magnitude come from symbol sequences only?

## Sources

* Kutter et al. (2022), Current Biology — https://homepage.uni-tuebingen.de/andreas.nieder/Kutter%20et%20al.%20(2022)%20CurrBiol.pdf
* "Math Neurons Identified in the Brain", Neuroscience News — https://neurosciencenews.com/math-neurons-20067
* University of Bonn release, 14 Feb 2022 — https://www.uni-bonn.de/en/news/028-2022
* Kutter et al. (2018), Neuron — https://homepage.uni-tuebingen.de/andreas.nieder/Kutter%20et%20al.%20(2018)%20Neuron.pdf
* Varley et al. (2005), PNAS — https://pmc.ncbi.nlm.nih.gov/articles/PMC552916 ; https://eprints.whiterose.ac.uk/337/1/varleyra1.pdf
* Cai et al. (2026), Nature, coverage — https://mgriblog.org/project/researchers-find-individual-neurons-that-capture-the-building-blocks-of-human-speech/ ;
  https://www.nih.gov/news-events/news-releases/researchers-discover-single-cell-brain-activity-underlies-human-speech ;
  https://carta.anthropogeny.org/node/333799
* Other works cited by name from prior knowledge (not re-checked online in this session): Monti et al. 2012; Amalric &
  Dehaene 2016; Piazza et al. 2007; Diester & Nieder 2007; Fedorenko et al. 2016; Nelson et al. 2017; Ding et al.
  2016; Mongillo et al. 2008.
