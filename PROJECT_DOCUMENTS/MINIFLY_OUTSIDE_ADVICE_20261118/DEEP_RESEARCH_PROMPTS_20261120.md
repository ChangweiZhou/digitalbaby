# Prompts for a deep research tool: the outside knowledge this programme lacks (20 November 2026)

## 1. What this file is and how to use it

**What it is.** Twenty prompts for an external web-research tool, drafted by a helper working inside the coordinating session of the outside-advice work, at Mr. Xu's request. His words, as given to the helper in the coordinating session's brief (the helper found them in no file of the work folder at the time of writing): "Send an agent to draft prompts for all external information you need from the deep research agent that we have not yet covered, just to avoid blind spots. I will run the deep search later myself"

**Status.** A proposal; it decides nothing. Nothing was run and no web search was made for it. No checker gate, seal or review is claimed. The names of papers, programs and researchers inside the prompts come from memory (the helper's and the coordinating session's) and are unverified; every prompt says so and asks for them to be checked. The verdicts of the coverage table and the ranking are the helper's reading.

**Dated** 20 November 2026, the day of the request; finished in the night that followed (21 November by the clock).

**Read by the coordinating session** in full before it was committed. Four changes were made to the helper's text, all in the paragraphs of facts: in prompt 3 the margins over xz were attributed to the right variants of the system; in prompt 7 the orders of the reference tables were corrected (3 to 6); and two results of the same night were added, each marked as one or two orders of the streams (prompt 1: a first trial of kept copies of the mixer weights; prompt 9: which counts spoil a stream read late, and a first trial of counts that halve at a bound). Nothing else was changed. Mr. Xu's request reached the coordinating session only when he repeated it; the first sending is in no record of the session.

**Source.** The work folder `PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/` at commit `c708b0de8` of the outside-advice session's working branch (the state page, the roadmap's sections 1 to 25, the report of `nano/suite`, the two plans of `nano/lives`, the two readings of 20 November), and the facts of the night as the coordinating session's brief gave them, which are not yet in those files. Every figure quoted in a prompt is development level: programs written and checked by sessions of the same kind, one run a stream unless said, repeated by no outside party.

**How to use it.** Each prompt stands alone. Paste one whole fenced block into one run of the tool, and nothing else. The prompts carry no name of a person, repository, branch or file of this project. They are ordered by how much the answer would change what the programme does next. **Prompts 1 to 5 are the minimum set**; if only one is run, prompt 1. Prompts 8, 11, 14, 17 and 18 touch ground that the reports and prompts of 15 November went over for the earlier learner (section 2, E): whoever holds those reports should look at them before running these five.

**How the list was made.** The helper first wrote its own list of what the programme decides without knowing what the literature knows, then read the coordinating session's list, and kept what was in either and is not already held. The last column of the index says where each came from.

## 2. What the project already holds from outside, and what it covers

**The reports** (found by reading the checkout, not taken from the brief):

* **A.** "Cells of the Human Brain: Types, Wiring, Local Roles, Global Functions, and Mechanisms Beyond Connectionism" (given on 20 November; set against the programme in `BRAIN_CELLS_REVIEW_AGAINST_PROGRAMME_20261120.md`). Biology.
* **B.** "Biological Circuit Mechanisms Literature Survey" (6 November), with its backing conversation and its executive summary: eleven circuits. Text copies in `output/minifly-mechanism-rescreen-20261107/sources/`. Biology.
* **C.** In the same folder: "Fly Brain Learning Mechanisms", its "Evidence Map and Verification Synthesis", and "Neuronal Human Learning Mechanisms Research". Biology.
* **D.** The human-learning audit of 24 October (`PROJECT_DOCUMENTS/HUMAN_LEARNING_AUDIT_20261024/MEMO.md` and `REVISED_AUDIT.md`). It drew on a report about teaching practice (a text copy is in the same sources folder) and on a second report, and takes back most of both (its section 8).
* **E. Three prompts and two reports of 15 November that the brief to the helper did not list.** The discussion memo of that day (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/DISCUSSION_MEMO_20261115.md`, on the default branch and not in this checkout) holds three prompts written by the reviewer of that time (its items 36, 42 and 54) and records that a report was uploaded after each of the first two (items 39 and 43). The reports themselves are not files in the checkout; what is said of them here rests on the text of the prompts and on that reviewer's summaries (items 40 and 44). The first prompt asked which mechanisms without end-to-end backpropagation have beaten a tuned backpropagation baseline (predictive coding, equilibrium and target propagation, forward-only schemes, Hebbian learning with a read-out, eligibility traces, feedback alignment), about fixed structure in place of gradients, fast memories beside a slow learner, outer loops without gradients, and theory for random features; all for small recurrent networks on byte-level algorithmic streams. The second asked how a map into a fixed circuit can be learned by a local rule, and what beats online gradient descent with a small replay buffer on single-pass streams under equal budgets. The third asked about learners whose way of learning changes with their own history: definitions of continual learning under bounded resources and the evaluation they prescribe, which weights may change (step sizes for each weight, protection by utility), which samples teach, when to rehearse, and meta-gradients. Neither that memo nor the memo of 16 November records a report for it.
* **F. The project's own literature checks**, made by worker sessions that opened the sources (not by the research tool): `PROJECT_DOCUMENTS/LOCAL_LEARNING_LITERATURE_20261025/` (44 papers: local credit, online recurrent credit, local self-supervised learning, continual learning, compositional benchmarks); `PROJECT_DOCUMENTS/LIT_CHECK_20261110/` (local credit through time; several timescales; delayed observation); `PROJECT_DOCUMENTS/LIT_CHECK_20261113/` (gates and approximate credit; learned feedback; time constants and delays).

**What follows from this (the helper's reading).** The brief supposed that everything held is biology or human learning. A to D are. E and F are machine learning, but all of it concerns the earlier learner, a network taught by local credit rules. On the present system, tables of counts joined by a mixer, the project holds nothing from outside. A search of A to F and of the discussion memo for the words of each prompt's topic (compression, context mixing, expert advice, change point, step size, smoothing, sketch, minimal pair, regret, hardware, scaling law) finds none of them treated as a topic; the passing mentions that bear on a prompt are entered in the table below. The programme's own plans name three things from memory and unsourced: the match model (`nano/suite/PLAN_3.md`), AdaGrad (`nano/suite/PLAN_4.md`) and escape estimated from the contexts' own record (`nano/table/PLAN_23.md`).

**Coverage table.** Rows are the topics of the prompts; columns are the reports A to F; "yes", "part" or "no" says whether a report covers the topic. The last column gives one line of evidence, a heading or a sentence that can be pointed to. For E the evidence is the prompt's wording or the reviewer's summary, since the reports were not read; "asked, no report on record" refers to the third prompt.

| No. | Topic | A | B | C | D | E | F | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | Recurring regimes without labels, for an online mixture | no | no | no | no | part | part | E, second prompt: "single-pass, non-stationary streams evaluated online"; summary: "Plain replay is the baseline to beat under matched budgets." F, list of 25 October: "D. Selective memory and continual learning (6)". Both for networks; nothing on mixtures of experts or pools of models. |
| 2 | Gated linear networks; networks of local online learners | no | no | no | no | part | part | E, first prompt asks for "forward-only schemes"; F, check of 10 November: "Q2. A target at every layer". No report names gated linear networks. |
| 3 | What context-mixing compressors do | no | no | no | no | no | no | No heading or sentence in any. The programme's own plan: "It is known art (the match model of compressors that mix)" (`nano/suite/PLAN_3.md`). |
| 4 | Scaling from a megabyte to a terabyte | no | no | no | no | no | part | F, check of 10 November, verdict 2: "Block-local training of 125M and 400M language models now exists". Nothing on count-based models or on data size. |
| 5 | Tests of general learning other than compression and arithmetic | no | no | no | no | part | part | E, first prompt, question 1(a): "held-out or out-of-distribution accuracy, including systematic or compositional generalization and length extrapolation"; F, list of 25 October, entries P39 and P40 (two compositional benchmarks). Nothing on tests scored by probability alone. |
| 6 | Telling where the stream is without over-splitting | part | no | no | part | no | part | A, key finding 4: astrocytes "integrate noradrenergic "failure" signals over time and trigger passivity" (a switch of state, no rule); D, `REVISED_AUDIT.md` section 9: "Latent causes / context-sensitive learning (normative model)"; F, entry P32: "Context selectivity with dynamic availability enables lifelong continual learning". |
| 7 | Baselines and fair comparison | no | no | no | no | no | no | None. The roadmap's section 3 gives the reference network's published figures "as I recall them" and has them measured locally. |
| 8 | Step sizes for a life that never becomes stationary | no | no | no | no | no | part | E, third prompt: "per-weight step-size adaptation (IDBD, Autostep and successors)": asked, no report on record. F, check of 11 November on timescales, verdict 3: "A one-variable metaplastic rule matches EWC on six permuted-MNIST tasks without task boundaries". Weights at several rates; nothing on rules for the step of an online convex learner. |
| 9 | Inside a context: counts to probabilities, forgetting, sharing | no | no | no | no | no | no | None. The reading of A says where to look: "the forms ... are in the literature of compression" (`BRAIN_CELLS_REVIEW_AGAINST_PROGRAMME_20261120.md`, section 5). |
| 10 | A bounded store | part | no | no | no | no | no | A, summary: "microglia prune synapses"; its table under Q4: "Adult neurogenesis ... Adding and removing units". Names only; nothing on a store of contexts. |
| 11 | Generalising beyond contexts met exactly, with local rules | no | part | no | no | part | part | B: "Circuit 4: Cerebellum as a Supervised Learner" (a fixed random expansion read by one learning layer); E, summary: "Fast memory stores and does not generalise, by both the record and the report."; F, source check of 11 November: "P-TNCN, 1.70 against 1.38 BPC". Local rules for networks are held; class-based and count-based similarity are not. |
| 12 | Recall of earlier occurrences, and attention | no | part | no | no | no | part | B: "Circuit 11: Attention as Learned Routing in Cortex"; F, entry P41: "Linear Transformers Are Secretly Fast Weight Programmers". Nothing on match models, cache models or nearest-neighbour models. |
| 13 | Units and levels found by the system | no | no | no | part | no | no | D, `REVISED_AUDIT.md` section 1, names "sequence chunking" among the mechanisms of an earlier design proposal. A name; no algorithm on text. |
| 14 | Structure found by search | no | no | no | no | part | part | E, first prompt, question 4: "Non-gradient outer loops."; summary: "Evolutionary search cannot cover an interface that includes a parser inside about a million weights."; F, entries P25 and P37 (plasticity rules found by an outer loop). Third prompt, question 3, "Adapting the learning procedure within one lifetime": asked, no report on record. Nothing on search of contexts in count models. |
| 15 | Theory of mixing and limits of context models | no | no | no | no | part | no | E, first prompt, question 5: "Theory that bounds the search." (random features on modular arithmetic). Nothing on regret of mixing or on formal limits of context models. |
| 16 | From predicting to acting | no | part | no | no | no | no | B: "Circuit 5: Basal Ganglia as an Actor–Critic with Dopamine as Reward-Prediction Error". The biology of reward; nothing on agents built on predictors. |
| 17 | Self-made lessons; never-taught cases | no | part | no | part | part | part | B: "Circuit 9: Sleep-Dependent Consolidation at the Circuit Level"; D, `MEMO.md`: "Complementary learning systems with offline consolidation and replay"; E, summary: "the only large out-of-range effects come from fixed group codes"; F, entry P36. Third prompt, "Which samples teach" and "When to rehearse": asked, no report on record. Nothing on self-training or co-training. |
| 18 | Evaluation method | no | no | no | no | part | no | E, second prompt: "evaluated online (each input scored before it is learned from) under an explicit, equal budget"; summary: "No budget-matched comparison exists for sequence streams". Third prompt, question 1, "Frameworks and evaluation": asked, no report on record. Nothing on error bars or on reuse of test data. |
| 19 | Hardware and the cost model | part | no | no | no | no | no | A, heading "Recommendations (for neuromorphic and computational modelling)"; under Q4, item 12: learnable delays "map naturally onto neuromorphic hardware". No measured result. |
| 20 | Context and reopening of learning, as algorithms | part | part | part | part | no | part | A, under Q2: "Noradrenaline raises gain and signals arousal and unexpected uncertainty."; B: "Circuit 2: Prefrontal Cortex with Basal-Ganglia Gating of Working Memory"; C, the report on human neurons: "metaplasticity—the activity-dependent modification of the threshold"; D: latent causes, as in row 6; F: Benna–Fusi chains in the check on timescales. Named in five, with no rule for a stream; the word "remapping" occurs in none. |

## 3. Index

Ordered by how much the answer would change what the programme does next (the helper's ranking). "From" says whose list the need came from: the helper's own, written first, the coordinating session's, or both.

| No. | Title | The decision it informs | Minimum set | From |
|---|---|---|---|---|
| 1 | Recurring regimes without labels: keeping the weights of an online mixture | What to build next against forgetting across kinds of stream, after the first remedy failed: a small pool of mixer weight vectors under a rule with a guarantee, or something else. | **yes** | both |
| 2 | Gated linear networks and other networks of local online learners | Whether the programme's layer of mixers is prior art; whether to take gating by side information and try depth again; whether a recipe for learning without task labels already exists for this family. | **yes** | both |
| 3 | What context-mixing compressors already do, component by component | Which parts of the system are known art, so that claims are worded rightly; which tested components to take next (weight sets by context, counters that forget, secondary estimation); a first estimate, from published tables, of the distance to that family. | **yes** | both |
| 4 | Scaling from a megabyte to a terabyte: count-based mixing against transformers | Whether a family built on tables is worth carrying beyond megabytes: the growth of the store, the order of cost against transformers, and how the bitter lesson reads for it. | **yes** | both |
| 5 | Tests of general learning for a byte-stream predictor, other than compression and arithmetic | What to measure now that addition is no longer the yardstick and log loss is the only one left: a test that separates memory of contexts from a rule that was learned. | **yes** | helper |
| 6 | Telling where the stream is, without labels and without over-splitting | The rule by which a new regime is created, or the choice to give up creating regimes in favour of soft mixtures; why the first remedy split source code into dozens of regimes. | no | both |
| 7 | Baselines and fair comparison: published numbers, programs that can be obtained, one-pass neural predictors | Which baseline to install and run on the programme's own streams, and what may be written about the comparison with a small transformer. | no | both |
| 8 | Step sizes for a life that never becomes stationary | Whether to replace the AdaGrad-like rule of the mixer, whose steps only shrink; how to remove the last number set by hand; the scaling by the number of active inputs. | no | both |
| 9 | Inside a context: from counts to probabilities, forgetting, and sharing between regimes | The last number still set per kind of data (the smoothing of counts); whether counts should fade; how short contexts shared by all kinds of stream can stop polluting one another. | no | both |
| 10 | A bounded store: which contexts to create, keep and drop | The rule for a store of fixed size (eviction, birth at a second occurrence, or a sketch), for a learner meant to live long. | no | both |
| 11 | Generalising beyond contexts met exactly, with local rules | Whether and how the programme can get what a table cannot give (an answer in a context never met): classes of contexts, similarity built by counting, or a small local-rule network as one more input. | no | both |
| 12 | Recall of earlier occurrences: match models, cache models, nearest-neighbour models, and attention | Whether to carry recall beyond exact matches; and, for the owner to rule on with the literature in hand, whether recall by exact match stands under the project's own rule against attention-equivalent reads. | no | helper |
| 13 | Units and levels found by the system: segmentation and hierarchy without labels | How to replace word ends given by hand with units the system finds; whether a second level deserves another attempt, and in what form. | no | helper |
| 14 | Structure found by search: which contexts and features to try | Whether and how to go on with the lines that search for context types and for mixer contexts; what removal of a feature costs in retention. | no | both |
| 15 | Theory: what online mixing guarantees and what finite-context models cannot do | Which observed limits are known theorems and which are faults of the build; what the present family cannot reach whatever is added. | no | both |
| 16 | From predicting to acting | The route from a predictor to the agent that is the aim; which of the project's earlier criteria (complete answers of the system's own) can be tested on the present system. | no | both |
| 17 | Self-made lessons: self-training, co-training and generated cases | Whether lessons the system composes for itself, and replay from its own store, are worth carrying beyond sums, and with what safeguards. | no | coordinating session |
| 18 | Evaluation method: error bars, reuse of scored bytes, changes of regime | What may be claimed from one deterministic run and from bytes scored dozens of times; the protocol for a life over several streams. | no | both |
| 19 | Hardware and the cost model | Whether any chip other than a plain processor with large memory is worth planning for; the statement of cost to set beside the references. | no | both |
| 20 | Context and the reopening of learning: algorithms from neuroscience and cognitive science | Whether any model from neuroscience gives a rule that prompts 6 and 8 do not; to be run only if those two come back thin. | no | coordinating session |

## 4. The prompts

Each fenced block is one run. Paste it whole. Every block ends with the same three paragraphs (what to give for every question, the standards, what to return); they are repeated so that each block stands alone. The line in italics above a block is for the project's readers and is not part of the prompt.

### Prompt 1. Recurring regimes without labels: keeping the weights of an online mixture (minimum set)

*Informs:* What to build next against forgetting across kinds of stream, after the first remedy failed: a small pool of mixer weight vectors under a rule with a guarantee, or something else. *Already held:* Nothing on mixtures. The second report of 15 November covered streams for networks with replay.

```text
Topic. Keeping what an online mixture of predictors has learned when earlier regimes return, with no labels.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. In standard terms it is online context mixing. For each context order 0 to 6 a table entry is created when a context first occurs and counts the bytes that follow. A mixer takes a softmax over the 256 byte values of a weighted sum of the entries' smoothed log-probabilities (a geometric mixture). Weights are selected by order and by a confidence bin of the entry (total count, distinct followers); they learn online from the log-loss gradient with an AdaGrad-like step per weight. Added: match models, a predictor of the last byte each context saw, and a second mixer whose weights also depend on each of the last six bytes.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). One system read the first nine tenths of six streams of 1 to 2 MB (English, source code, a table, markup) in turn. Then each stream's last tenth was scored while learning went on (test-then-train). Mean loss at the end: 0.881 to 0.929 nats per byte over six orders, against 0.836 for six separate systems; one stream lost 0.17, and in one order a loss had not shrunk 64,000 bytes into the return. Restoring the mixer weights saved just after a stream removed about two thirds of its loss (0.160 became 0.051). The rest sits in counts of short contexts shared by all streams. A constant step was no better. The loss gives no sign of a change of kind. A first trial (one order): a copy of the mixer weights kept every 65,536 bytes, all copies answering, combined by fixed share, removed 38 to 66% of the loss on the returning streams and was never worse.

Questions.
1. Prediction with expert advice when earlier regimes return: fixed share, mixing past posteriors (reported: Bousquet and Warmuth 2002), sleeping experts and specialists. What does each guarantee at a return? Which carry over from a convex mixture of fixed experts to a log-linear mixer trained by gradient steps?
2. Which still hold when the experts themselves keep learning and share state (shared count tables)?
3. Partition tree weighting, context tree switching and the forget-me-not process (reported: Veness and colleagues; Milan and colleagues 2016): algorithms, cost, and measurements on real sequences.
4. Pools of stored models, or of stored mixer weight vectors, selected or mixed online (Bayesian online change-point detection with a pool; growing ensembles of experts; anything in compressors): rules, the weight a new or returning member receives, results.
5. Language-model adaptation before neural networks (mixtures of domain models re-weighted on recent text): gains when domains alternate.
6. Published failures of these methods on real data.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 2. Gated linear networks and other networks of local online learners (minimum set)

*Informs:* Whether the programme's layer of mixers is prior art; whether to take gating by side information and try depth again; whether a recipe for learning without task labels already exists for this family. *Already held:* Nothing. The literature checks of 10 November treat layer-local objectives for networks; none names gated linear networks.

```text
Topic. Networks in which every unit predicts the target and learns from its own loss, with nothing passed backward between layers.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. In standard terms it is online context mixing. For each context order 0 to 6 a table entry is created when a context first occurs and counts the bytes that follow. A mixer takes a softmax over the 256 byte values of a weighted sum of the entries' smoothed log-probabilities (a geometric mixture). Weights are selected by order and by a confidence bin of the entry (total count, distinct followers); they learn online from the log-loss gradient with an AdaGrad-like step per weight. Added: match models, a predictor of the last byte each context saw, and a second mixer whose weights also depend on each of the last six bytes.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Each unit of this system is a geometric mixer trained on its own log loss. In the one form tried, further mixers stacked on the first added nothing beyond one wide mixer: each only added terms to the same weighted sum of the same inputs. A second mixer whose weights depend on the last six bytes lowered the loss by 0.017 to 0.120 nats per byte. Reading six streams of different kinds in turn, the system forgets the earlier ones, mostly in its mixer weights.

Questions.
1. Gated linear networks (reported: Veness and colleagues, 2017 to 2021): write out the unit, the gating by side information (half-space gating is reported), the local loss, the update, initialisation and clipping. Is anything passed backward between layers?
2. What was measured (data, metric, numbers, sizes, passes, baselines), and was it reproduced by others? Any result on byte or character prediction or on compression?
3. Depth: do stacked units gain from depth, by how much, and why? What if every unit sees the same side information? What is proved (universality), under which assumptions?
4. Forgetting: what is claimed, on which benchmark, with which numbers? "A combinatorial perspective on transfer learning" (reported 2020) is said to join these networks with the forget-me-not process, for continual learning without task labels: method, results, limits.
5. Relation to the mixers of PAQ and cmix (weight sets selected by context, two-layer mixing): what is the same, what is new?
6. Present state: extensions, failures, criticism, and whether the work was continued.
7. Other learners with no backward pass between layers (stacking, cascade-correlation, online boosting, deep forests): one-pass forms, and results on sequence prediction.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 3. What context-mixing compressors already do, component by component (minimum set)

*Informs:* Which parts of the system are known art, so that claims are worded rightly; which tested components to take next (weight sets by context, counters that forget, secondary estimation); a first estimate, from published tables, of the distance to that family. *Already held:* Nothing. Three plans name the match model, AdaGrad and learned escape from memory.

```text
Topic. The published rules of compressors that mix context models, and the measured worth of each part.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. In standard terms it is online context mixing. For each context order 0 to 6 a table entry is created when a context first occurs and counts the bytes that follow. A mixer takes a softmax over the 256 byte values of a weighted sum of the entries' smoothed log-probabilities (a geometric mixture). Weights are selected by order and by a confidence bin of the entry (total count, distinct followers); they learn online from the log-loss gradient with an AdaGrad-like step per weight. Added: match models, a predictor of the last byte each context saw, and a second mixer whose weights also depend on each of the last six bytes.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Test-then-train loss on the last tenth of six streams of 1 to 2 MB, each after its first nine tenths (two English texts, Python, C headers, a table of numbers, XML), nats per byte: this system 1.33, 1.39, 0.94, 0.69, 0.34, 0.28; xz at its highest preset 1.75, 1.79, 1.13, 0.72, 0.38, 0.43. The margin is thin on two streams (about 0.02 against xz with tuned settings; 0.007 for the variant without age-weighted recency); on 13 further streams the system is below xz on 10. No context-mixing compressor has been run: the distance to that family is unknown. A larger variant adds skip contexts and contexts keyed on the distance since each byte value last occurred; recency predictors are weighted by age. Match models and the last-byte predictor were added from memory, without sources.

Questions.
1. For PAQ8 and its descendants, lpaq, ZPAQ, cmix, PPMd/PPMII and context tree weighting, write out: how a context's history is stored, made to forget and turned into a probability; the mixer (inputs, update, learning rate, selection of weight sets by context, two layers); secondary estimation (SSE, APM); match models (minimum length, how confidence enters); sparse, word, record, distance and indirect contexts.
2. For each component, the gain measured in a published ablation.
3. Bitwise against bytewise prediction: reasons and measured differences.
4. Archives that mix kinds of file in different orders: order effects, one kind polluting another's statistics, and what the programs do (reported: type detection, weight sets per type, resets).
5. On standard small text and source files (Calgary and Canterbury corpora): bits per character of PPMd, lpaq, paq8 and cmix beside xz and bzip2.
6. Which parts of the system above have no counterpart in this literature?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 4. Scaling from a megabyte to a terabyte: count-based mixing against transformers (minimum set)

*Informs:* Whether a family built on tables is worth carrying beyond megabytes: the growth of the store, the order of cost against transformers, and how the bitter lesson reads for it. *Already held:* Nothing on count-based models. The check of 10 November has one result on block-local training of large language models.

```text
Topic. How count-based predictors scale with data, memory and time, against neural networks.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Over 6.3 MB of English read once (one set of books, one order), the number of table entries grew as the bytes read to the power 0.5 to 0.6 (0.8 with more context types), to 0.43 stored counts per byte; the loss while learning fell from 1.45 to 1.34 nats per byte. Cost on one core: 2 to 7 seconds per stream of 1 to 2 MB for the small system, 2 to 3 minutes for the largest; per byte, the number of context types times 256, whatever the context length. Stopped after 1 MB of English it is about level with a 10.7-million-parameter character transformer on the same text (one run). The project takes the "bitter lesson" as a steer: general methods that transfer, no fixes for one task.

Questions.
1. Published curves of bits per character against the amount of training text (1 MB to 1 GB and beyond) for PPM, context mixing and n-gram models, and for neural networks on the same data (enwik8, enwik9, text8). At what data and model size do neural models overtake?
2. Extrapolation laws for compression rate against text length (Hilberg's conjecture; stretched-exponential fits, reported: Takahira, Tanaka-Ishii and Dębowski 2016). What do they predict at 1 TB?
3. Growth of the number of distinct contexts of each order with text length: measured exponents. Does theory give a lower bound on the memory an optimal predictor of natural language needs?
4. n-gram models at very large scale (reported: "stupid backoff"; suffix-array models of unbounded order): quality against neural models, memory per n-gram, gains of hybrids.
5. Order of computation and memory per byte of context mixing against transformers of equal loss; measured speed and memory (reported: the Large Text Compression Benchmark).
6. Reported: n-gram models lost to neural networks as data grew. The measured causes.
7. The furthest any learner with no global backward pass has reached on byte or character prediction at 100 MB or more.
8. The "bitter lesson" essay (Sutton 2019): quote its claims. By its own criteria, is a table-based learner a method that scales with computation?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 5. Tests of general learning for a byte-stream predictor, other than compression and arithmetic (minimum set)

*Informs:* What to measure now that addition is no longer the yardstick and log loss is the only one left: a test that separates memory of contexts from a rule that was learned. *Already held:* Nothing on tests scored by probability alone. The list of 25 October names two compositional benchmarks.

```text
Topic. Tests that show whether a next-byte predictor has learned something general, which need no labels and no generation.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). The aim is a learner that keeps what it learned, revises what has become false, and uses it on cases it was never shown. So far ideas were judged on two-digit sums never shown in training and on log loss over six streams of different kinds (English, source code, a table, markup). The project no longer holds on to addition and seeks general tests. A table of contexts can score well on log loss by memory and recency alone, so log loss does not show whether a rule was learned.

Questions.
1. Tests that need only the probabilities a model gives to strings: minimal-pair grammar tests (reported: BLiMP, Zorro, SyntaxGym). How are they scored for a byte- or character-level model? Published scores of n-gram baselines, recurrent networks and transformers, by amount of training text.
2. Small-data benchmarks for language learners (reported: the BabyLM challenge, 10 and 100 million words): tasks, baselines, and the best entries that are not neural or use no backpropagation, if any.
3. Formal-language suites graded by the Chomsky hierarchy with tests of length generalisation (reported: Delétang and colleagues 2023): tasks, protocol, results, and what finite-context learners reach.
4. Tests of learning inside a stream: a regularity introduced at one point and tested later on new cases. Existing benchmarks that do not assume a neural network.
5. Task solving by compression: classification by cross-entropy under class models, normalised compression distance (reported: Teahan; Bratko and colleagues 2006; Cilibrasi and Vitányi 2005). Method, and results against standard classifiers.
6. Tests of retention, and of revising facts that change over time, for stream predictors.
7. Generation: how the quality of text sampled from count-based models has been measured.
8. Which of these tests has a published control showing that a pure n-gram table fails where a learner of rules passes?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 6. Telling where the stream is, without labels and without over-splitting

*Informs:* The rule by which a new regime is created, or the choice to give up creating regimes in favour of soft mixtures; why the first remedy split source code into dozens of regimes. *Already held:* Names only: latent causes in the audit of 24 October; a switch of state by glia in the report on brain cells.

```text
Topic. Inferring which regime a byte stream is in, online and without labels, without splitting one regime into many.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). One system reading six streams of different kinds in turn forgets the earlier ones (mean loss 0.88 to 0.93 nats per byte at the end against 0.84 for separate systems). The loss gives no sign of a change of kind. A first remedy failed (one order): regime detectors that each hold a histogram of byte values, a belief over them updated by Bayes with a small share for switching, and a new detector created when an empty one would have coded the last block of 1,024 or 4,096 bytes more cheaply than every existing one; each regime owned a separate complete predictor. Python and C source were split into dozens of regimes (a cap of 64 was reached) and the result was worse than no remedy (0.945 and 0.922 against 0.918).

Questions.
1. Over-segmentation in nonparametric mixtures: the sticky HDP-HMM (reported: Fox and colleagues 2008), the inconsistency of Dirichlet-process mixtures for the number of components (reported: Miller and Harrison 2013), split-and-merge moves. What is established, the remedies, and which have one-pass forms?
2. Segmentation by description length: the correct test of "new model" against "reuse an old one" (a fresh model is always cheap on a short block), penalties, hysteresis, block length, merging.
3. Recurring concept drift in data-stream mining: detectors (ADWIN, Page-Hinkley), pools of past models, tests that two concepts are the same. Rules, results, open implementations.
4. Recognising the kind of data by compression (cross-entropy under per-class models): how sharp is the signal on windows of 1 to 4 KB, and what context order does it need? How do PAQ-family compressors detect data types?
5. Soft against hard assignment: mixtures in which all components contribute and share statistics, against one active component with private statistics.
6. New regimes whose benefit shows only when an old regime returns: delayed decisions, candidates on probation, later merging.
7. Context inference in cognitive models (latent-cause models; hippocampal remapping as hidden-state inference, reported: Sanders, Wilson and Gershman 2020): algorithms, and tests on real streams.
8. Published cases where regime detection made a predictor worse.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 7. Baselines and fair comparison: published numbers, programs that can be obtained, one-pass neural predictors

*Informs:* Which baseline to install and run on the programme's own streams, and what may be written about the comparison with a small transformer. *Already held:* Nothing from outside; the references were measured locally.

```text
Topic. The right baselines for an online byte predictor at 1 to 2 MB, and what a fair comparison with a small neural network requires.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Comparators run so far, on the project's own streams: xz, bzip2, gzip; smoothed n-gram tables (Kneser-Ney, orders 3 to 6); and the widely used six-layer, 10.7-million-parameter character transformer on the 1.1 MB "tiny Shakespeare" text. On that text's last tenth: transformer 1.454 to 1.456 nats per byte (one run, shorter than the published schedule); this system 1.467 with learning stopped, and 1.32 to 1.34 when it goes on learning through the test. No PPM or context-mixing compressor is installed. The machine can install from standard package indexes only.

Questions.
1. Per-file results in bits per character for xz, bzip2, PPMd, lpaq, ZPAQ, paq8 variants, cmix and neural compressors on the Calgary, Canterbury (reported to hold a Shakespeare play and C source), Silesia and enwik8/enwik9 sets: versions, options, memory, time.
2. Which PPM or context-mixing implementations can be had from standard package indexes (Python or system packages) or as one small source file? Licences, strongest options, and how to get the cost of only the last part of a file after priming on the first part.
3. Neural predictors trained online in one pass, each symbol predicted before it is learned from (reported: NNCP; the recurrent model inside cmix): methods, sizes, and results on files of about 1 MB against large files.
4. Dynamic evaluation of neural language models (reported: Krause and colleagues 2018): gains on enwik8 and text8. Is it the accepted comparator for a learner that goes on learning during the test?
5. "tiny Shakespeare": published validation losses of character models, the best reported, and any result of compressors or n-gram models on that file. Is it documented that strong statistical compressors match small neural models at about 1 MB?
6. Accepted practice for comparing stored counts with parameters, one pass with many passes, and compute.
7. What a careful reviewer would require before "level with a small transformer at 1 MB" may be written.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 8. Step sizes for a life that never becomes stationary

*Informs:* Whether to replace the AdaGrad-like rule of the mixer, whose steps only shrink; how to remove the last number set by hand; the scaling by the number of active inputs. *Already held:* Nothing on step-size rules. The third prompt of 15 November asked about step sizes for each weight; no report is on record. The check on timescales (11 November) treats weights held at several rates.

```text
Topic. Step-size rules for an online log-linear mixer whose data never become stationary.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. In standard terms it is online context mixing. For each context order 0 to 6 a table entry is created when a context first occurs and counts the bytes that follow. A mixer takes a softmax over the 256 byte values of a weighted sum of the entries' smoothed log-probabilities (a geometric mixture). Weights are selected by order and by a confidence bin of the entry (total count, distinct followers); they learn online from the log-loss gradient with an AdaGrad-like step per weight. Added: match models, a predictor of the last byte each context saw, and a second mixer whose weights also depend on each of the last six bytes.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). One number (0.03) sets every step. On single streams this rule matched or beat two hand-set schedules. Its steps only shrink. Reading six streams of different kinds in turn, neither it nor a constant step kept the earlier ones; two thirds of the loss was in the mixer weights. With hundreds of inputs the number had to be divided by the root of the count of active inputs. Starting the sum at zero, a first gradient of rounding size made a full step. Inputs are strongly correlated (nested contexts); new inputs start at weight zero.

Questions.
1. Step-size adaptation for tracking: IDBD, Autostep, SMD, vSGD ("no more pesky learning rates"), averages of recent squared gradients, restarts. Write each rule. Which need no number set per task, recover after a change, and still settle? Comparisons on nonstationary streams.
2. Parameter-free and scale-free online learning (coin betting; normalised updates): rules, guarantees, behaviour with sparse inputs.
3. Normalising by the number or norm of active inputs (normalised least mean squares): theory, and relation to the scaling found here.
4. AdaGrad in practice: initial accumulator, discounted forms, per-coordinate FTRL (reported: McMahan and colleagues 2013); behaviour when the data change.
5. Many inputs of which few matter: multiplicative updates (Winnow, exponentiated gradient) against additive ones; L1 or truncated gradients as a principled way to drop inputs.
6. Strongly correlated inputs: second-order updates (online Newton step; reported for PAQ-like mixing by Knoll and de Freitas 2012): cost and gain.
7. Loss of plasticity in continual learning (reported: Dohare and colleagues): does it occur for convex online learners?
8. Learning rates driven by an estimate of change (reported: Nassar and colleagues 2010; volatility models): rules and tests.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 9. Inside a context: from counts to probabilities, forgetting, and sharing between regimes

*Informs:* The last number still set per kind of data (the smoothing of counts); whether counts should fade; how short contexts shared by all kinds of stream can stop polluting one another. *Already held:* Nothing. One plan notes from memory that compressors came to estimate escape from the contexts' own record.

```text
Topic. How a count-based context model should turn counts into probabilities when the source changes and several regimes share it.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Each table entry adds a constant to every count (0.02 on text; two-digit sums needed 2): the one number that still differs between kinds of data. A learned replacement: entries of each order and count bin measure how often such an entry was followed by a byte it had never counted (45% after one count; 0.4 to 2% after 64 or more) and leave that share for unseen bytes. It costs 0.002 to 0.005 nats per byte against the tuned constant on English. Counts never fade. A second, fading copy of the counts helped while learning went on and hurt once learning stopped. After one continual run over six kinds of stream about one third of what was lost sits in the counts of short contexts that all kinds share. On concatenated C headers, dropping entries at random beat keeping all. A stream read after other kinds also costs more than for a fresh system (0.353 against 0.326 on the markup stream read last, one order); emptying the counts of contexts of 0 to 3 bytes before the new stream removed most of that (two orders), resetting the mixer weights did not. A second copy of those short contexts whose counts halve at 32 or 512 lowered the loss on every stream (one order; 1 to 13%).

Questions.
1. The zero-frequency problem: PPM escape methods A to D and X, secondary escape estimation (PPMZ, PPMII), the Krichevsky-Trofimov estimator, Good-Turing missing mass, Kneser-Ney and its modified form, Pitman-Yor. Write the rules, with measured differences on standard files. Is the learned rule above one of them?
2. Learned maps from a small history state to a probability (indirect modelling and adaptive state maps in PAQ and ZPAQ): rules and gains.
3. Counts when the source changes: halving at a limit, exponential forgetting, forgetting rates learned per context, two rates mixed. Rules, theory, measured gains.
4. Sharing between regimes: hierarchical Bayesian sharing of n-gram statistics across domains (reported: Wood and Teh 2009), the sequence memoizer and any online or forgetting forms, short contexts conditioned on a domain variable. Rules, cost, results.
5. Starting a new context from its parent's statistics (reported: information inheritance in PPMII): rule and gain.
6. Deterministic contexts and the last symbol seen (LZP, symbol ranking): how PPM and context mixing use them.
7. Which of these use only the entry's own record and its parent's?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 10. A bounded store: which contexts to create, keep and drop

*Informs:* The rule for a store of fixed size (eviction, birth at a second occurrence, or a sketch), for a learner meant to live long. *Already held:* Nothing on stores of contexts; pruning appears only as biology.

```text
Topic. Keeping the table of contexts of a count-based predictor within fixed memory at the least loss.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Entries are created at a context's first occurrence; a half to four fifths have been met once; their number grows as the bytes read to a power of 0.5 to 0.8. Measured on six streams of 1 to 2 MB with the small system (0.3 to 0.6 million entries unbounded): half the store, evicting the least counted and oldest entry, costs at most 0.007 nats per byte; a quarter 0.006 to 0.025; an eighth 0.011 to 0.057; evicting at random at half costs up to 0.073 on English; creating a lasting entry only at a context's second occurrence gives 46 to 66% fewer entries for 0.004 to 0.016.

Questions.
1. What PPMd, the PAQ family, ZPAQ and context-tree-weighting programs do when memory is full (reported: restart, cut-off, hashed buckets with check bits and replacement by priority). Rules, and the measured loss against memory size for one program at several settings.
2. Which contexts deserve to exist: prediction suffix trees and variable-order Markov models (reported: Ron, Singer and Tishby 1996), Rissanen's Context algorithm, context-tree maximising, pruning by BIC, the cloning thresholds of dynamic Markov compression. Rules, one-pass forms, results.
3. Count cut-offs and entropy pruning in n-gram modelling (reported: Stolcke 1998): size against loss. Is creation at a second occurrence the same as dropping singletons?
4. Approximate stores: count-min sketches with conservative update, Bloom-filter language models, lossy counting and space-saving with error bounds. Bits per entry and measured loss.
5. Bytes per stored context in PPMd, PAQ hash tables and compact n-gram stores.
6. Suffix arrays or suffix trees as the store (all orders implicit): memory per byte of text, and use for online prediction.
7. Eviction that takes account of entries whose old counts now mislead: any published rule beyond least-recent and least-frequent.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 11. Generalising beyond contexts met exactly, with local rules

*Informs:* Whether and how the programme can get what a table cannot give (an answer in a context never met): classes of contexts, similarity built by counting, or a small local-rule network as one more input. *Already held:* Partly: local-learning methods for networks (checks of 25 October and 10 to 13 November; first report of 15 November). Nothing on class-based or count-based similarity.

```text
Topic. Ways for a count-based predictor to answer in contexts it has never met, without backpropagation through layers.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). A table of contexts cannot answer in a context it has never met. Networks get this from representations learned by backpropagation, which the constraints exclude. Found so far: contexts whose predictions agree were merged into classes by greedy search and further contexts keyed on the classes; with one setting this gained 0.010 to 0.016 nats per byte on English books and, on two-digit sums, formed classes that are the carry; it cost 4 to 7 times the entries and 30 times the compute. An earlier small network trained by a local rule (the output error relayed over learned feedback connections) answered 95% of sums never shown but reached only 1.72 to 1.77 nats per byte on English, where the count tables reach 1.47 to 1.52.

Questions.
1. Class-based n-gram models and clustering of contexts (Brown clustering; exponential class models such as "Model M"; agglomerative information bottleneck): algorithms, online forms, measured gains.
2. Histories made equivalent by their predictions: state merging for probabilistic automata (ALERGIA, MDI), causal-state splitting reconstruction, online or spectral learning of hidden Markov models, distributional learning of substitutable languages, ADIOS. Rules, learnability results, scale reached, results on natural text.
3. Similarity without gradients: random indexing and hyperdimensional vectors built by accumulation; count-based embeddings (pointwise mutual information with SVD; Hebbian online PCA); locality-sensitive hashing of contexts; memory-based learning (nearest neighbours with back-off). Language-modelling results.
4. Sparse distributed memory, the sequence memory of hierarchical temporal memory, Tsetlin machines: rules, and best results on character prediction against n-gram and backpropagation baselines.
5. Local-learning neural methods (forward-forward, predictive coding, equilibrium propagation, target propagation, feedback alignment, eligibility traces): one table of their best character-level language-modelling results (enwik8, text8, Penn Treebank) against backpropagation at equal size and data. The project already holds checked notes on these methods for other tasks; only these numbers are needed.
6. A count-based predictor with a small local-rule network as one more input: measured gains.
7. Which of these answer unseen combinations in a controlled test, and which only smooth?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 12. Recall of earlier occurrences: match models, cache models, nearest-neighbour models, and attention

*Informs:* Whether to carry recall beyond exact matches; and, for the owner to rule on with the literature in hand, whether recall by exact match stands under the project's own rule against attention-equivalent reads. *Already held:* Nothing on match or cache models; attention appears only as a cortical circuit and as a comparator.

```text
Topic. Predicting from earlier occurrences of the present context, and how the literature relates this to attention.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. In standard terms it is online context mixing. For each context order 0 to 6 a table entry is created when a context first occurs and counts the bytes that follow. A mixer takes a softmax over the 256 byte values of a weighted sum of the entries' smoothed log-probabilities (a geometric mixture). Weights are selected by order and by a confidence bin of the entry (total count, distinct followers); they learn online from the log-loss gradient with an AdaGrad-like step per weight. Added: match models, a predictor of the last byte each context saw, and a second mixer whose weights also depend on each of the last six bytes.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). The match model here: a table from the last eight bytes to the places where they last occurred (up to three); it predicts the byte that followed there; its mixer weight is indexed by the length of the agreement (8 to 2,048 bytes). Gain: 0.01 nats per byte on English, 0.07 to 0.12 on source code, a table and markup. Matching is exact. An earlier rule of the project excluded attention and any content-addressed read equivalent to it; whether recall by exact match falls under such a rule has not been examined.

Questions.
1. Match models in PAQ, lpaq, ZPAQ and cmix; LZP; PPM* and other unbounded contexts: rules (minimum length, verification, recovery after a miss, how confidence enters the mixer, several candidates) and measured gain by type of data.
2. Cache language models, continuous caches, pointer mechanisms and nearest-neighbour language models (reported: Kuhn and De Mori 1990; Grave and colleagues 2017; Khandelwal and colleagues 2020): rules, gains, and which need representations learned by backpropagation.
3. Approximate recall without learned representations: matches with gaps or edits, locality-sensitive hashes of recent bytes, longest match by suffix array. Published results.
4. "Induction heads" in transformers (reported: Olsson and colleagues 2022) are described as match-and-copy. How does the literature state the relation between exact-match recall, n-gram models of unbounded order and attention? Is exact-match recall formally a special case of attention? Sources on both sides.
5. What attention does that exact or approximate match-and-copy cannot (soft matching on learned features, composition over layers): controlled comparisons.
6. Memory cost of recall over a long life, and standard ways to bound it.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 13. Units and levels found by the system: segmentation and hierarchy without labels

*Informs:* How to replace word ends given by hand with units the system finds; whether a second level deserves another attempt, and in what form. *Already held:* Nothing; chunking is named once as a human mechanism.

```text
Topic. Finding units (words, chunks) and higher levels in a raw byte stream, online and without labels, for use in prediction.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Contexts that advance once per word, with word ends given by hand (space, line end), gain 0.015 to 0.025 nats per byte on English. Ends found by the system from changes in its own uncertainty keep 90% of that gain where a byte marks the end, and one fifth when every space is removed. A second counting level keyed on the first level's best guess gained 0.005 on English and nothing on two-digit sums. Mixers stacked on mixers added nothing.

Questions.
1. Segmentation from predictability: successor counts (Harris), branching entropy, transitional probabilities, the Voting Experts algorithm. Rules, accuracy on text without spaces, online forms.
2. Bayesian word segmentation and the nested Pitman-Yor language model (reported: Goldwater and colleagues 2009; Mochihashi and colleagues 2009): algorithms, whether one pass and local, results, cost.
3. Grammar-based compression as hierarchy found online: Sequitur, Re-Pair, byte-pair merging. Rules, the hierarchy found, and measured gains when its units serve as contexts for prediction.
4. Word models and dictionary preprocessing in PPM and context mixing: gains in bits per character on text, and on data that are not text.
5. Count models with several levels (bytes, chunks, chunks of chunks): any that beat flat context mixing, by how much, on what.
6. Chunking models from cognitive science (reported: PARSER): rules, and tests on real text.
7. Source code, tables and markup: what units unsupervised methods find there, and whether they help prediction.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 14. Structure found by search: which contexts and features to try

*Informs:* Whether and how to go on with the lines that search for context types and for mixer contexts; what removal of a feature costs in retention. *Already held:* Only outer loops over plasticity rules for networks (list of 25 October; first report of 15 November).

```text
Topic. Online, gradient-free search for the contexts and features of a count-based predictor.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Tried so far: (a) context types drawn at random (subsets of recent positions), each new one starting at weight zero and dropped if its weight stays small: better than random types never changed, level with hand-chosen types on two-digit sums and worse on two English books of three, and dropping harmed what had been learned earlier; (b) the type with the largest weight extended by one more position, chosen on the last 50,000 bytes, its entries started from pooled statistics of the parent: 0.010 to 0.015 nats per byte on six English books, at 5 to 8 times the compute; (c) contexts for the second mixer's weights drawn at random and kept by what they earned: 0.69 to 0.92 of the gain of hand-given ones.

Questions.
1. Generate-and-test of features under an online linear learner (reported: Mahmood and Sutton 2013; unit replacement in continual backpropagation): utility measures, replacement rules, results, effect on retention.
2. Free choice of the next conditioning position: free-tree coding and adaptive templates in bilevel image coding, decision-tree language models (reported: Bahl and colleagues 1989), Hoeffding trees (a split test in one pass). Rules and results.
3. Context sets found by search in compressors: evolutionary or greedy optimisation of sparse context masks and of model sets. Gains, and whether found sets transfer between kinds of data.
4. Greedy feature induction for log-linear models (reported: Della Pietra, Della Pietra and Lafferty 1997) and its online forms.
5. Choosing context sets by description length: criteria, and cost of search per byte.
6. Evidence that more search keeps helping as compute grows, for gradient-free online search of structure.
7. Induction of programs or grammars by search from examples (reported: Levin search, genetic programming, library learning): what has scaled, and results on sequence prediction without gradients.
8. Changing the learning rule itself within one run, with no gradient taken through the learning process (reported: the success-story algorithm; self-modifying learners): results against a tuned fixed learner, and known pitfalls.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 15. Theory: what online mixing guarantees and what finite-context models cannot do

*Informs:* Which observed limits are known theorems and which are faults of the build; what the present family cannot reach whatever is added. *Already held:* Only theory for random features on modular arithmetic (first report of 15 November).

```text
Topic. Guarantees for online mixing of context models, and the formal limits of such models.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. In standard terms it is online context mixing. For each context order 0 to 6 a table entry is created when a context first occurs and counts the bytes that follow. A mixer takes a softmax over the 256 byte values of a weighted sum of the entries' smoothed log-probabilities (a geometric mixture). Weights are selected by order and by a confidence bin of the entry (total count, distinct followers); they learn online from the log-loss gradient with an AdaGrad-like step per weight. Added: match models, a predictor of the last byte each context saw, and a second mixer whose weights also depend on each of the last six bytes.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Observations to be explained or bounded: mixers stacked on mixers added nothing; a context chain that learns to skip ignorable bytes made a task with fillers between operands learnable; contexts keyed on the system's own guesses of the next two or three bytes let count tables answer 91% of two-digit sums never shown (77% without; five seeds); on a closed set of training items the system grew sure that unseen items do not occur.

Questions.
1. Regret bounds for geometric (logistic) mixing under log loss (reported: Mattern 2012 to 2016; exp-concavity; online Newton step): statements, assumptions, constants, for bitwise logistic and bytewise softmax forms, and with weight sets selected by context.
2. What is proved for networks of local mixers (gated linear networks).
3. Redundancy bounds for context tree weighting, PPM and switching methods; whether and why dictionary coders lose to context models on small files.
4. Limits of finite-context models in formal terms: strictly local, strictly piecewise and tier-based strictly local languages, regular and context-free languages. What skip contexts and a chain that skips ignorable symbols add. Learnability of the tier (reported: Jardine and Heinz 2016).
5. Long-range dependence, copying, counting, carrying in arithmetic: which are provably out of reach of finite-context mixing, and which come within reach with a match model or with contexts keyed on the system's own earlier predictions?
6. The theoretical account of where transformers gain over n-gram models (transformers representing n-gram models; in-context learning as implicit Bayesian mixing; statistical against computational arguments).
7. Theory of forgetting for convex online learners whose weights are shared across regimes; and of missing-mass estimation on a closed set.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 16. From predicting to acting

*Informs:* The route from a predictor to the agent that is the aim; which of the project's earlier criteria (complete answers of the system's own) can be tested on the present system. *Already held:* Only the biology of reward learning (the circuit survey).

```text
Topic. Building an agent that acts and answers on top of an online sequence predictor made of counts.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). The aim is a persistent learning agent; the present system only predicts the next byte. The project's own criteria: complete answers produced by the system itself with its own end mark; knowledge kept through later learning; obsolete knowledge revised; use on cases never taught. One result so far joins a model with replay: a small local-rule network kept a skill through a second task when it replayed examples sampled from the count tables (94% of never-shown sums kept against 2% without).

Questions.
1. Agents built on count-based predictors: the Monte-Carlo AIXI approximation on context tree weighting (reported: Veness and colleagues 2011), "compress and control", utile suffix memory and U-Tree (reported: McCallum). Algorithms, tasks, sizes, results, limits.
2. Dyna and model-based replay (reported: Sutton 1990): relation to replay from the learner's own predictive model; known failures when the model is wrong.
3. Discrete active inference: learning as accumulation of Dirichlet counts, structure learning by Bayesian model reduction, recent claims of learning games in one pass. Methods, results, independent replication.
4. Producing answers from a sequence predictor: decoding from PPM or context-mixing models, question answering from a memory of counts. Published attempts and their quality.
5. "Compression as a route to intelligence" (the rationale of the Hutter Prize; "Language modeling is compression", reported 2024): claims, evidence, counter-arguments.
6. In-context learning as Bayesian mixture prediction: does explicit count-based mixing reproduce it? Evidence.
7. The smallest environments in which a lifelong byte-stream agent has been evaluated, and their protocols.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 17. Self-made lessons: self-training, co-training and generated cases

*Informs:* Whether lessons the system composes for itself, and replay from its own store, are worth carrying beyond sums, and with what safeguards. *Already held:* Partly: never-taught algorithmic cases for learners without backpropagation (first report of 15 November); replay as biology. Nothing on self-training or co-training.

```text
Topic. A learner that makes its own training cases, and two learners of different kinds that teach each other, without outside labels.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). On one family of tasks (two-digit sums; five seeds): a count-based mixer learns fast and then grows sure that items it has not met do not occur; a small local-rule network learns slowly and answers 95.5% of sums never shown. Joined as two inputs to one mixer they did not add up (79.0% after 400,000 examples). They did when, at intervals, the network composed new examples and the mixer's weights were moved toward the network's answer wherever the two already agreed: 98.8%. Elsewhere, examples that the system itself had completed were stored for replay; one sixth of them were wrong and stayed wrong.

Questions.
1. Self-training and pseudo-labels: when they help and when they lock in errors; theory and measurements.
2. Co-training and learning by agreement between learners with different biases (reported: Blum and Mitchell 1998): assumptions, and results outside classification, above all in sequence prediction.
3. Pseudo-rehearsal and generative replay (reported: Robins 1995; Shin and colleagues 2017): replay sampled from the learner's own model, accumulation of errors, results in continual learning with no stored data.
4. A fast memory-based learner and a slow generalising learner teaching each other (complementary learning systems as algorithms; distillation): measured results, and any in which neither part uses backpropagation through layers.
5. Generalisation to longer or never-taught cases on algorithmic tasks by learners with no global backward pass: results published since late 2025. The project holds earlier reports on this; only new or missed results are needed.
6. Choosing the generated cases on which the learner's loss is largest: relation to prioritised replay and maximally interfered retrieval; results.
7. How to detect that self-made lessons are making the learner worse.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 18. Evaluation method: error bars, reuse of scored bytes, changes of regime

*Informs:* What may be claimed from one deterministic run and from bytes scored dozens of times; the protocol for a life over several streams. *Already held:* Partly: online scoring under equal budgets for networks (second report of 15 November). Nothing on error bars or on reuse of test data.

```text
Topic. Sound evaluation of online byte predictors: uncertainty of small differences, repeated use of test data, and streams whose regime changes.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). Practice so far: the systems are deterministic and each was run once per stream (six streams of 1 to 2 MB). Differences as small as 0.007 nats per byte have been read. The same last tenth of one text was scored for about 75 variants, and each of four other streams 26 times, with later ideas shaped by earlier scores. Two measures are used: learning stopped before the test, and learning going on through it. Keeping is measured by reading six streams in turn and testing the earlier ones at the end. Plans are written before runs, with predictions and their chances; only the count that held is reported.

Questions.
1. Test-then-train (prequential) evaluation: definitions, fading factors and sliding windows under drift (reported: Gama and colleagues 2013), pitfalls.
2. Error bars for the difference in mean log loss of two predictors on the same dependent sequence: the Diebold-Mariano test, block bootstrap, sequential tests with e-values. How to apply them to per-byte losses, and the length needed to detect 0.005 nats per byte.
3. Repeated use of the same test bytes: theory (adaptive data analysis, the reusable holdout), the measured size of the effect in benchmarks (reported: Recht and colleagues 2019), and how compression benchmarks guard against tuning to the corpus.
4. Accepted practice for comparing a learner that goes on learning during the test with one that is trained and then fixed.
5. Standard streams with changes of regime and returns, for sequence prediction on text, code and tables. If none exist, how others have built them.
6. Continual-learning measures (forgetting, backward and forward transfer, the "stability gap", reported: De Lange and colleagues 2023) restated for prequential log loss.
7. Formal definitions of continual learning under bounded memory and compute (reported: computationally constrained reinforcement learning; the "big world" hypothesis; the Alberta Plan): the evaluation each prescribes, and any use on sequence prediction.
8. Predictions registered with probabilities before a run: how to score them (Brier or log score, calibration), and accepted practice for preregistration in machine learning.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 19. Hardware and the cost model

*Informs:* Whether any chip other than a plain processor with large memory is worth planning for; the statement of cost to set beside the references. *Already held:* One clause in the report on brain cells; no measured result.

```text
Topic. What hardware suits a learner made of sparse hashed lookups and local updates, by measured results only.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). For each byte the system makes one hashed table lookup for each of 17 to 355 context types (random access to a store of 0.3 to 4 million entries per stream of 1 to 2 MB), updates a few numbers in each entry, and takes a softmax over 256 values. There is no backward pass. Measured on one processor core: 2 to 7 seconds per stream for the small system, 2 to 3 minutes for the largest.

Questions.
1. The cost model on ordinary processors: cache misses per lookup; how PAQ-family and PPMd programs lay out their tables to limit them; measured speed and memory of lpaq, ZPAQ, PPMd and cmix.
2. Hardware forms of adaptive context modelling: the context-adaptive binary arithmetic coder of video standards; FPGA or ASIC versions of PPM, context tree weighting and context mixing. Throughput, energy, table sizes.
3. Content-addressable memory and in-memory computing for longest-context match and for count updates: measured results.
4. Neuromorphic chips with learning on the chip: which local rules they support, any sequence-prediction or language result with on-chip learning, energy per learning event.
5. Does the absence of a backward pass bring a measured hardware advantage (memory for activations, latency, learning on the device)?
6. Graphics processors and many cores: can hashed sparse lookups with a strict byte-by-byte dependence be run in parallel (many streams at once; speculative lookups)?
7. Energy and time per byte against a small transformer on the same processor, for prediction and for learning. Measured figures only.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 20. Context and the reopening of learning: algorithms from neuroscience and cognitive science

*Informs:* Whether any model from neuroscience gives a rule that prompts 6 and 8 do not; to be run only if those two come back thin. *Already held:* Partly, as names without rules: unexpected uncertainty, prefrontal gating, consolidation, metaplasticity, latent causes. Remapping occurs in none.

```text
Topic. Computational models from neuroscience and cognitive science of how a learner infers its present context and reopens learning after a change, as algorithms only.

Setting. A small research project builds a learner that reads a byte stream once and keeps learning all its life. The present system is online context mixing: count tables for context orders 0 to 6, combined in the logistic domain by a mixer whose weights learn online from the log loss, with match models and a second, context-dependent mixer.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). Each unit learns only from what reaches it. One pass; each byte is predicted before it is learned from. No task or regime labels. Nothing tuned for one kind of data.

Facts (development level: one team's programs and checks, one run per stream unless said). The project already holds literature reports on brain cell types, neuromodulators as slow signals sent to many cells, prefrontal and basal-ganglia gating, complementary learning systems and sleep. They name "unexpected uncertainty" and metaplasticity without giving a rule. Tried here with little or no effect: one signal sent to all units that scales every step by recent error; units that stiffen with age. What is missing is an algorithm for inferring the present context with no labels, and for reopening learning when the stream changes, that can be run on byte streams.

Questions.
1. Hippocampal remapping as hidden-state inference (reported: Sanders, Wilson and Gershman 2020) and latent-cause models (reported: Gershman, Blei and Niv 2010): the algorithm, the prior that makes contexts persist, how a new context is created, tests.
2. Unexpected uncertainty and noradrenaline (reported: Yu and Dayan 2005), delta rules driven by change-point probability (reported: Nassar and colleagues 2010), volatility models (reported: Behrens and colleagues 2007; the hierarchical Gaussian filter): update equations, whether each unit can compute them from its own signals, tests on real data streams.
3. Metaplasticity and consolidation as algorithms (cascade models, the Benna-Fusi chain, synaptic tagging and capture): any test on sequence prediction beyond random patterns. The project holds a checked note on the theory; only tests on streams are needed.
4. Models of prefrontal gating that infer task sets without labels (reported: Collins and Frank 2013): algorithm and results.
5. Which of these have been run as machine-learning methods on data that are not toys, with numbers?
6. Evidence against: cases where such a model was shown to be equivalent to a simpler statistical method, or not to scale.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

## 5. What no prompt covers, and why

Things the helper considered and left out, or kept only as a single question. These are the helper's judgements.

1. **The biology of cells, circuits and the fly, and human learning as teaching practice.** Five reports are held (A to D). The reading of 20 November found that such a report gives "names more than forms". Only algorithms for context and for reopening learning are asked again (prompt 20), and last; the two leading ones are also named as starting points in prompts 6 and 8, so prompt 20 may not be needed.
2. **Local credit rules for networks in general** (feedback alignment, predictive coding, equilibrium propagation, eligibility traces, layer-local objectives). Held in F and in the first report of E. Only their language-modelling figures are asked for (prompt 11, question 5).
3. **Online continual learning for networks with replay under equal budgets.** Asked on 15 November (E, second prompt). Prompts 1, 6 and 18 ask about the same ground only for mixtures of count-based predictors.
4. **Generalisation to longer sums as a topic of its own.** The steer is away from addition, and the first report of E covered learners without backpropagation on algorithmic tasks. Kept as one question on what is new (prompt 17, question 5) and as general test suites (prompt 5, question 3).
5. **Online meta-gradients and learned optimisers.** Part of the third prompt of 15 November, for which no report is on record. They need a gradient taken through the learning process itself, which the constraints exclude between units; only the gradient-free forms are asked for (prompt 14, question 8).
6. **Arithmetic coding and the engineering of real compressors.** The programme predicts; it does not have to emit a code.
7. **Data that are not text-like** (images, sound, executables). The test-bed holds none yet, as the report of `nano/suite` says of itself. When it does, prompt 3 should be run again with those kinds named.
8. **Large pretrained language models as comparators.** Another scale and outside the constraints; they appear only as the upper end of prompt 4.
9. **Questions the project must settle itself, which no search can.** (a) Whether recall by exact match stands under the rule of the research-loop skill, section B, against "renaming an attention-equivalent content-addressed read"; prompt 12 gives only how the literature classifies such recall. (b) Whether a weight shared by every entry of one context type is "local" as the project's constraints mean it; the reading of the brain-cells review raised it ("no synapse is shared"). (c) What counts as an agent for this programme, and which of the earlier criteria still bind after the steer away from addition.
10. **Questions an experiment answers faster than a search.** The distance from the programme's system to a PPM or context-mixing compressor on its own six streams takes minutes to measure once such a program is at hand; prompt 7, question 2, asks only how to get one, and prompt 3, question 5, asks for published ratios in the meantime.
11. **The reports of 15 November themselves.** They are not in the checkout, and the prompts here were written from the reviewer's summaries of them. A copy placed in the work folder would let prompts 11, 17 and 18 be trimmed before they are run.

**A check of this file against its brief** (made once by the helper after writing): no prompt names a person, repository, branch, file or product of the project; each prompt carries the setting, the constraints and the facts with their caveats; each asks for the rules written out, the measurements with data and unit, a table of sources, at most three one-day experiments with what would count against each, the strongest counter-argument, and one paragraph on what was not asked; the project's own words for its parts ("cell", "trust", "answering cell", "sender") occur in no prompt, the standard terms being used throughout, and "kind" and "goes on learning" occur only in their ordinary sense; each prompt is between 350 and 700 words, counted as strings between spaces.
