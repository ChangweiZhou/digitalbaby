# Prompts for a deep research tool, second round: what is still not covered, for persistent learning (21 November 2026)

## 1. What this file is and how to use it

**What it is.** Ten prompts for an external web-research tool, drafted by a helper working inside the coordinating session of the outside-advice work. They follow the twenty prompts of `DEEP_RESEARCH_PROMPTS_20261120.md` (round one), which Mr. Xu ran, and the report that answered them, "Outside literature for the twenty research prompts" (822 lines; called "the report" below). They ask only for what that report left open and for what the work of the night of 20 to 21 November made newly worth knowing.

**Mr. Xu's words**, as the coordinating session's brief gave them to the helper (the helper found them in no file of the work folder at the time of writing). The request: "Send an agent to draft prompts for all external information you need from the deep research agent that we have not yet covered, just to avoid blind spots. I will run the deep search later myself". And, on the night of 21 November: "No need, our goal is persistent learning, not info compressing alone." and "我们的目标始终是持续学习，这一点从未改变" (the coordinating session's translation: "Our goal has always been persistent learning; this has never changed.").

**What follows from those words for this file** (the helper's reading, not Mr. Xu's wording). Every prompt is about persistent learning: a learner that goes on learning all its life and (a) keeps what it learned of A through learning B, (b) has it when A returns, (c) revises what has become false, (d) uses it on cases never shown, (e) does not learn the next thing worse for what it already holds; and that ultimately acts as an agent. Loss in nats per byte appears only as the yardstick. No prompt asks for a better compressor or for a compressor to compare with, and each prompt says so to the tool.

**Status.** A proposal; it decides nothing. Nothing was run and no web search was made for it. No checker gate, seal or review is claimed. The names of papers and researchers inside the prompts come from the helper's memory and are unverified; every prompt says so and asks for them to be checked. The verdicts of the coverage table, the ranking and the choice of what to drop are the helper's reading.

**Dated** 21 November 2026.

**Source.** The work folder `PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/` on the outside-advice session's working branch at commit `d409621ee`: round one; `nano/lives/PLAN_1.md` to `PLAN_10.md`; `ROADMAP_SECOND_REVISION_20261120.md`, sections 26 and 27. The report was read in full from the copy Mr. Xu uploaded to the session; it is not a file of the work folder. Where a plan's outcome was not yet entered at the time of reading (plans 3 and 7 to 10), the figure used is the one the later plans or the brief give, and is marked. `OUTSIDE_LITERATURE_AGAINST_THE_PROGRAMME_20261121.md`, another helper's reading of the same report, did not exist when this file was written and was not read. Every figure quoted in a prompt is development level: programs written and checked by sessions of the same kind, mostly one run for each order of the streams, repeated by no outside party.

**How to use it.** Each prompt stands alone. Paste one whole fenced block into one run of the tool, and nothing else. The prompts carry no name of a person, repository, branch or file of this project, and no name of an AI product. Each begins from what round one established, in a sentence or two, so that the tool does not repeat it. They are ordered by how much the answer would change what the programme does next for persistent learning. **Prompts 1 to 4 are the minimum set**; if only one is run, prompt 1.

**How the list was made.** The helper first wrote its own list of what is still unknown, from four sources: what the report says it could not find or verify; questions its answers raise that nobody asked; what the night's results make newly relevant to a decision; and the properties round one touched least, (c), (d) and acting as an agent. It then dropped what the report already answers well enough (section 2), and what an experiment or the owner settles faster than a search (section 5).

## 2. What round one and its report now cover, and what they left open

Rows are the twenty prompts of round one. "Covers" is what the report settled well enough for the programme's next decisions; "left open" is what bears on persistent learning and is still unknown; the evidence is one sentence of the report with the heading under which it stands. The last column says what round two does with the row. "Dropped" means that no round-two prompt follows it, either because the report answered it or because, after Mr. Xu's words above, it serves compression and not persistent learning.

| Round one | The report now covers | Left open for persistent learning | Evidence (heading of the report) | Round two |
|---|---|---|---|---|
| 1. Recurring regimes | The guarantees of fixed share, mixing past posteriors and specialists for a linear mixture of frozen members; partition tree weighting; forget-me-not; the weight a new member receives. | Any measurement on bytes; a bound when members share learned counts; members that go on learning; a pool over a long life; whether the worth of the pool is an artefact of single returns. | "Prompt 1", Answers, Q2: "no rule can recover loss caused by later streams changing the counts that the copies read". Strongest counter-argument: "the protocol (in which each stream returns once, for its last tenth) may overstate the value of a memory over time". | Prompts 1, 4, 5. |
| 2. Gated linear networks | The unit; depth helps only through different gating; a label-free recipe exists, tested on images only. | Keeping by gating when no side information separates the regimes. | "Prompt 2", Strongest counter-argument: "the resistance of GLNs to forgetting rests on side information that separates tasks, which the programme's shared streams do not have". | Prompt 7. |
| 3. Context-mixing compressors | Which components are known art. | Nothing for persistent learning. | "Prompt 3", Strongest counter-argument: "Compression margins against this family may therefore be unable to demonstrate new learning." | Dropped. |
| 4. Scaling to a terabyte | Where networks trained by backpropagation overtake count tables. | Nothing asked again; the choice it bears on is the owner's. | "Executive summary", Prompt 4: "The record argues against tables alone." | Dropped. |
| 5. Tests of general learning | Minimal pairs, formal-language tasks, their artefacts, the need for held-out lexical items. | A test of never-shown cases inside a stream, and after later learning; revision of changed facts by a learner that is not a neural network. | "Prompt 5", Answers, Q4: "No benchmark of in-stream rule learning for non-neural predictors exists." Q6 names benchmarks "for neural models" only. | Prompts 2, 9. |
| 6. Where the stream is | Why the detector over-split (no persistence prior, no merging, a block compared with each regime and not with their mixture); the correct test. | The sharpness of a label-free signal on short windows. | "Prompt 6", Answers, Q4: "How sharp an online per-class cross-entropy signal is at 1 to 4 KB has not been measured anywhere found." | Dropped as answered; the open signal is one question of prompt 7. |
| 7. Baselines | Which programs to run and what a fair comparison requires. | Nothing for persistent learning. | "Prompt 7", Strongest counter-argument: "A small transformer is not the relevant bar." | Dropped. |
| 8. Step sizes | That no step rule removes forgetting; a floor, an offset, a cap. | Nothing that a search settles faster than a run. | "Prompt 8", Strongest counter-argument: "no step rule can keep two streams in one weight vector". | Dropped. |
| 9. Inside a context | The learned escape is secondary escape estimation; bounded and discounted counts are known art. | Sharing of short-context statistics between regimes without labels; forgetting rates learned for each context. | "Prompt 9", Answers, Q3: "Forgetting rates learned per context were not found." Q4: the one model of sharing across domains "does not discuss online inference". | Prompts 2, 6. |
| 10. A bounded store | Memory is not where the loss lies at 1 to 2 MB. | Stale entries, as a matter of revision and not of capacity. | "Prompt 10", Strongest counter-argument: "the real issue is stale counts, not capacity". | Dropped; staleness is asked in prompt 2. |
| 11. Beyond contexts met exactly | Class and similarity models smooth; few numbers exist for local rules on characters. | How to test what was gained (prompt 9), not how to gain it. | "Prompt 11", Answers, Q7: "class and similarity models mainly smooth". | Dropped. |
| 12. Recall and attention | How the literature relates exact-match recall to attention. | Only the owner's ruling. | "Prompt 12", Answers, Q4: "The ruling therefore hinges on whether the rule bans learned soft similarity, or any key-based lookup." | Dropped (section 5). |
| 13. Units and levels | Segmenters; no gain from hierarchy shown. | Nothing for persistent learning. | "Prompt 13", Strongest counter-argument: "hierarchy has never been shown to beat flat CM on a standard benchmark". | Dropped. |
| 14. Structure by search | Generate-and-test and its two guards. | Nothing that a search settles faster than a run. | "Prompt 14", Answers, Q1: "The programme's dropping rule lacks the maturity guard and the contribution utility". | Dropped. |
| 15. Theory | Bounds for gradient-trained mixing; the formal limits of finite contexts. | A theory of forgetting and of interference for learners that share state. | "Prompt 15", Answers, Q7: forgetting theory is limited to linear regression and to "switching-regret bounds, which say nothing about recovery time". | One question of prompt 6. |
| 16. Predicting to acting | Agents on context-tree predictors; a good predictor that was a bad simulator. | Exploration; how reward enters a stream; any agent that lives one long life. | "Prompt 16", Answers, Q7: "No lifelong byte-stream agent was found." Strongest counter-argument: "Exploration, and how reward enters a byte stream, were not asked about". | Prompt 10. |
| 17. Self-made lessons | When self-training locks errors in; the standard guards. | Consolidation of a pool of stored states. | "Prompt 17", Answers, Q7: "There is no established detector." | Dropped; one question of prompt 5. |
| 18. Evaluation method | Prequential scoring; error bars for dependent sequences; reuse of test data. | A stream with returns; published measures of transfer for log loss (the report's are its own restatement); the definitions, read only in part. | "Prompt 18", Answers, Q5: "No standard stream with designed switches and returns across text, code and tables exists". Q6: "the research team's restatement". Q7: "The Alberta Plan was opened only as an abstract." | Prompts 3, 4, 8. |
| 19. Hardware | No special chip is worth planning for now. | Nothing. | "Prompt 19", Strongest counter-argument: "Specialisation pays only after the algorithm is fixed." | Dropped. |
| 20. Context from neuroscience | The models are approximations of change-point detection and of mixtures that statistics already has. | Recognising a return, and telling it from a change of fact. | "Prompt 20", Strongest counter-argument: "The decisive question that was not asked is how to recognise a *return* and restore what was learned". | Prompts 1, 2, 7. |

**What the report itself lists as not found, and where it goes.** From its section "What remains unverified or not found", under "Not found anywhere searched": "Byte-stream measurements of mixing past posteriors, specialists or forget-me-not; a bound for pools of predictors that share learned counts" (an experiment, section 5; the theory is asked in prompts 1 and 5); "an in-stream rule-learning benchmark for non-neural predictors" (prompt 9); "An online per-class cross-entropy signal at 1 to 4 KB" (prompt 7); "learned per-context forgetting rates" (prompt 2); "An online, label-free, domain-shared HPYLM" (prompt 6); "A lifelong byte-stream agent" (prompt 10). The other items of that list concern compressors, hardware or networks and are not asked again.

**What the night's work adds that the report could not know** (`nano/lives`, development level). Stored copies taken within one regime are twins, and shares between twins carry no evidence (prompt 1). A copy stored every 65,536 bytes makes a pool that grows without bound (prompt 5). Stored copies and lifetime counts are built to bring the old back, and nothing yet separates that from a failure to revise (prompt 2). The loss on streams read late sits in the counts of short contexts that all kinds share, while one stream gains from what was read before it (prompt 6). Every result so far rests on streams that return once (prompt 4).

## 3. Index

Ordered by how much the answer would change what the programme does next for persistent learning (the helper's ranking). Prompt 10 stands last because nothing in the next steps waits on it, not because acting matters least: it is the aim.

| No. | Title | The decision it informs | Minimum set |
|---|---|---|---|
| 1 | Members of a pool that go on learning: who learns, and the problem of twins | The next step against forgetting, planned and not yet run: a stored copy that goes on learning while it holds the belief. Which members should learn, and what evidence may wake or overwrite one. | **yes** |
| 2 | Revising what has become false: a changed fact against a returning regime | The design and the pass mark of the planned test of revision (a frequent word replaced throughout); and how stored copies and lifetime counts, built to bring the old back, are to be kept from bringing back what is no longer true. | **yes** |
| 3 | What counts as persistent learning: definitions, trivial baselines, known critiques | The wording of the aim and of any claim; which reference systems to run beside "six separate systems" (a learner that stores everything; an oracle that knows the boundaries); and which resource must be bounded for the problem not to be trivial. | **yes** |
| 4 | Lives with returns: how regimes recur in natural streams, and benchmarks with repetition | The protocol of the lives planned next (twelve kinds; streams woven in blocks): how often and after how long a regime should return, and whether published work says that natural repetition already removes most forgetting. | **yes** |
| 5 | A bounded pool over a long life: which stored states to keep, merge and drop | The rule that keeps the pool of stored copies within fixed memory once lives grow; whether a stored state should hold counts as well as weights; whether to consolidate between streams. | no |
| 6 | Learning the next thing no worse: interference in shared short contexts, and sharing without labels | Whether counts that halve at a bound are the whole answer to streams read late, or whether short-context statistics should be shared between regimes in a principled way that keeps the transfer now observed. | no |
| 7 | Keeping by selection: weights and counts chosen by a slow summary of the input, against stored copies over time | Whether to add selection by a slowly varying, label-free summary of the input (the remedy the earlier report prefers) beside or in place of copies stored over time; and what such selection cannot do. | no |
| 8 | Measuring a life: forward and backward transfer, savings, and the reference systems | The measures and reference systems the reports of lives should carry, so that keeping, return, revision and "no worse at the next thing" are each a number a reader from the field recognises. | no |
| 9 | Never-shown cases that survive later learning: test designs | The test of property (d) on general streams, which the programme lacks since it stopped holding on to addition: a design that fits inside a life and is scored again after later streams. | no |
| 10 | Persistent learning for an agent built on counts: reward, exploration, and the smallest test-beds | Nothing in the next few steps; it sets the first test-bed and the first design for acting when the programme turns to the agent that is its aim. | no |

## 4. The prompts

Each fenced block is one run. Paste it whole. Every block carries the same setting, aim and constraints, and ends with the same three paragraphs as round one (what to give for every question, the standards, what to return); they are repeated so that each block stands alone. "An earlier search" in a block means round one and its report. The line in italics above a block is for the project's readers and is not part of the prompt.

### Prompt 1. Members of a pool that go on learning: who learns, and the problem of twins (minimum set)

*Informs:* The design of the stored copy that goes on learning, and the rule that replaces "more than half of the belief" as evidence. *Follows* round-one prompt 1. *Properties:* (a), (b).

```text
Topic. A pool of stored predictors whose members go on learning: who should learn at each step, and what to do about members that are near-copies of one another.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search established that fixed share, mixing past posteriors and specialists guarantee returns only for a linear mixture of frozen members; nothing was found measured on bytes. Here one system reading six kinds of stream (1 to 2 MB each) in turn ends 5 to 11% worse than six separate systems (five orders); three quarters of the loss on returning streams is in the mixer weights. Frozen stored copies remove 38 to 66% of it (one order). Overwriting the living weights with a copy holding over half of the belief failed (173 to 201 such returns inside one stream of one kind): copies from one regime are near-twins, and belief wanders among them by chance. Planned, not run: the copy with the largest share learns on beside the living weights.

Questions.
1. Pools whose members learn: who learns at each step (all; in proportion to posterior weight; the leader only; those "awake")? Which guarantees of expert-advice theory survive when a member's future depends on its own weight?
2. Near-identical members: posterior weight wandering among them (reported: "dilution" in Bayesian model averaging); remedies (hysteresis, tempered posteriors, sequential tests on loss differences before acting).
3. Winner-take-all collapse in mixtures of experts and multiple choice learning (reported: Jacobs and colleagues 1991; Guzman-Rivera and colleagues 2012): remedies; one-pass forms with no backward pass between units.
4. Online ensembles for drifting streams that weight and train their members (reported: dynamic weighted majority, Kolter and Maloof 2007; Learn++.NSE; follow-the-leading-history): which freeze old members; results on recurring concepts.
5. Task-free continual learning with growing mixtures (reported: Aljundi and colleagues 2019; Lee and colleagues 2020): birth and update rules, what needs backpropagation, results.
6. Published cases where a stored member, woken on the wrong regime, lost what made it worth storing; safeguards.
7. For each: what has been shown for persistent learning specifically (keeping, return), not accuracy on a stationary benchmark?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 2. Revising what has become false: a changed fact against a returning regime (minimum set)

*Informs:* The planned test of revision and the rule that separates "keep both" from "overwrite". *Follows* round-one prompt 5 (question 6) and prompt 9 (question 3). *Property:* (c), and its conflict with (b).

```text
Topic. How an online count-based predictor should revise what has become false, and how it can tell a changed fact from a returning regime.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search found tests of changing facts only for neural language models on time-ordered text, and no forgetting rate learned per context. Here aims (b) and (c) pull against each other: stored copies exist to bring old weights back when a regime returns; counts that halve at 32 or 512 let recent evidence outvote old; lifetime counts never fade. When old predictions begin to fail, nothing yet tells "another regime: keep both" from "the world changed: overwrite". No revision has been measured. Planned, not run: a frequent word replaced by another through the rest of a stream, then tested in contexts where the new word was never shown.

Questions.
1. The formal distinction between a change of regime and a change of fact: real against virtual concept drift (reported: Gama and colleagues 2014); latent-cause accounts of extinction, where small errors update an old memory and large ones create a new one (reported: Gershman and colleagues 2010, 2017). Rules; tests on real streams.
2. Interference paradigms, A-B then A-C (reported: Barnes and Underwood 1959; McCloskey and Cohen 1989): measures; which learners keep A-B while answering A-C.
3. Change detection inside one context of a count model: run-length posteriors (reported: Adams and MacKay 2007), discounting triggered by surprise, what PPM-family models do when a deterministic context breaks. Rules, cost, measurements.
4. How fast a belief with n confirmations should yield to k contradictions: the Bayes-optimal answer under a hazard rate; human data.
5. Measures from knowledge-editing work, as measurement designs only (reported: efficacy, locality, generalisation; "ripple effects", Cohen and colleagues 2023): any use on non-neural predictors.
6. In pools of stored models (mixing past posteriors, forget-me-not): how an obsolete member is kept from resurging; published failures.
7. What has been shown for a learner that must both keep and revise over one long life?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 3. What counts as persistent learning: definitions, trivial baselines, known critiques (minimum set)

*Informs:* The frame of the whole line: what may be claimed, against which trivial baselines, under which bound. *Follows* round-one prompt 18 (question 7). *Properties:* all.

```text
Topic. What "persistent" (continual, lifelong) learning requires by its published definitions, which trivial baselines already meet them, and what makes the problem hard.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search found one formal objective (average reward, or negative log loss, under a compute budget per step; reported: Kumar and colleagues 2023); the "big world" paper and the Alberta Plan were not read. Here one system reading six kinds of stream in turn ends 5 to 11% worse than six separate systems (five orders); stored weight copies and bounded counts each remove part of that. A whole life so far is under 12 MB, so keeping the raw stream would be cheap; the project has not decided whether that is admissible.

Questions.
1. Definitions (reported: Ring 1997; Thrun and Mitchell 1995; Hadsell and colleagues 2020; Abel and colleagues 2023; Kumar and colleagues 2023; Javed and Sutton 2024; the Alberta Plan): which of (a) to (e) does each require? Does any argue that forgetting what does not recur is optimal under bounded resources?
2. Trivial baselines: store everything and retrain or look up (reported: "GDumb", Prabhu and colleagues 2020; nearest neighbours over the stored stream, Prabhu and colleagues 2023); one model per segment given by an oracle. Their benchmark scores, and the constraint under which each stops being admissible.
3. Is it established that memory-based (non-parametric) learners do not forget? What do they fail at instead (generalisation, revision, cost)?
4. Critiques of continual-learning evaluation (reported: Farquhar and Gal 2018; Hammoud and colleagues 2023).
5. Proven tensions among the properties: stability against plasticity; hardness of optimal continual learning (reported: Knoblauch and colleagues 2020).
6. Online Bayesian learning is said not to forget in principle: where does it fail in practice (misspecification, shared parameters, growing confidence)?
7. What would a careful reviewer require before "learns persistently" may be written of a system tested on six to twelve streams of megabytes?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 4. Lives with returns: how regimes recur in natural streams, and benchmarks with repetition (minimum set)

*Informs:* The schedule of returns in the planned lives, and how much weight the single-return result may bear. *Follows* round-one prompt 18 (question 5) and the counter-argument of prompt 1. *Properties:* (a), (b).

```text
Topic. How regimes recur in natural streams, what continual learning with repetition has shown, and how to build a life of byte streams with returns.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search found no standard stream with designed switches and returns across text, code and tables, and warned that a protocol in which each stream returns once may overstate the worth of a memory over time. Here six kinds of stream of 1 to 2 MB are read in turn (five orders), and each returns once, for its last tenth. A life costs 5 to 11% against separate systems. At a return the excess loss over the first 4,000 bytes is 2.7 times that over the whole return: recovery takes tens of thousands of bytes. Planned, not run: twelve kinds; the six kinds woven in blocks of 65,536 bytes.

Questions.
1. Measured statistics of recurrence: return times of topics, words, files or tasks (reported: Church and Gale 1995; Altmann and colleagues 2009; need as a power function of recency and frequency, Anderson and Schooler 1991). What schedule of returns do they imply?
2. Continual learning with repetition (reported: Stojanov and colleagues 2019; Cossu and colleagues 2022; Hemati and colleagues 2023): does natural repetition shrink forgetting until a plain learner suffices? Numbers.
3. Blocked against interleaved schedules in humans and in machines (reported: Flesch and colleagues 2018; the spacing effect): what each predicts for retention.
4. Generators and real data sets with recurring concepts in data-stream mining: how recurrence is injected; known critiques.
5. Any multi-domain character or byte stream with returns used for sequence prediction since 2020.
6. Design: block length, number of returns, orders, fresh text at each return; excluding exact repeats, which reward copying, not learning.
7. Which results concern persistent learning over one life, and which only a sequence of tasks with resets?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 5. A bounded pool over a long life: which stored states to keep, merge and drop

*Informs:* The pool's size rule and the content of a stored state. *Follows* round-one prompts 1 (questions 3 and 4) and 10. *Properties:* (a), (b), over a long life.

```text
Topic. Keeping a pool of stored predictor states within fixed memory over a long life: which to keep, merge and drop, and what a stored state should contain.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search established: the forget-me-not process caps its pool by thresholds on reservoir subsamples (about 30 members); with a few dozen members a switch costs little, so which copies are kept matters more than the combining rule (an inference); storing short-context counts with each copy was proposed, not measured. Here a copy of the mixer weights is stored every 65,536 bytes, so the pool grows linearly with the bytes read; copies from one regime are near-twins; about a quarter of the loss on a returning stream is in counts, not weights (five orders).

Questions.
1. Fixed-budget forms of mixing past posteriors, specialists and follow-the-leading-history: pruning rules, and the regret lost.
2. Merging near-duplicate members: criteria (divergence between predictions on a reservoir; correlation of losses), what to keep after a merge (reported: mixture reduction, Runnalls 2007), measured effects.
3. Which members to keep when returns cannot be foreseen: recency and frequency rules from cache replacement (reported: ARC, LRFU) and from the rational analysis of memory (reported: Anderson and Schooler 1991); any use for pools of models.
4. Pools for recurring concepts in data-stream mining: caps, eviction rules, pool sizes reached on long streams.
5. Stored state larger than weights: versioned count tables (copy-on-write, deltas, persistent data structures); cost per snapshot; published uses in online learners.
6. Consolidation: reducing many stored members to few, offline between streams and without backpropagation (moment matching; refitting on samples from the learner's own store). Results, and damage done.
7. Evidence beyond a few dozen regimes or returns.
8. For each: what has been shown for persistent learning specifically (what is still there at a late return)?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 6. Learning the next thing no worse: interference in shared short contexts, and sharing without labels

*Informs:* What to build after bounded counts for streams read late. *Follows* round-one prompts 9 (question 4) and 15 (question 7). *Property:* (e).

```text
Topic. Why what a count-based learner already holds makes it learn the next thing worse, and how statistics can be shared between regimes without labels so that they help and do not hurt.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search established that bounded and discounted counts are known art, and found hierarchical sharing across domains only in batch form with labels. Here a stream read late in a life is learned 0.02 to 0.06 nats per byte worse than by a fresh system; emptying the counts of contexts of 0 to 3 bytes before that stream removes 0.88 of the loss at the median (three orders), resetting the mixer weights 0.14. Counts that halve at a bound, kept beside the lifetime counts, cut it by two thirds (five orders). Yet one stream (C headers) is learned 0.02 to 0.05 better after other kinds than alone: there is transfer worth keeping.

Questions.
1. Negative forward transfer (proactive interference) in online learners: theory by overlap of features or parameters (reported: Riemer and colleagues 2019; Evron and colleagues 2022); the cost of unlearning a prior of n wrong counts.
2. Shared-plus-private estimators, task identity unknown: online multitask learning (reported: Cavallanti and colleagues 2010); n-gram models conditioned on a latent topic inferred online (reported: Gildea and Hofmann 1999; Bellegarda 2004). One-pass forms; results when domains alternate.
3. The past as a prior whose strength is learned: empirical-Bayes concentration (reported: MacKay and Peto 1995), power or discounted priors, stabilised forgetting (reported: Kulhavý and Zarrop 1993).
4. Overconfidence of count learners late in life, as an analogue of loss of plasticity: evidence, remedies.
5. Learners shown to learn each new stream faster as a life goes on, with no gradient taken through the learning process.
6. What fast-fading short memories lose (the positive transfer above): measured trade-offs.
7. For each: what has been shown for persistent learning specifically?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 7. Keeping by selection: weights and counts chosen by a slow summary of the input, against stored copies over time

*Informs:* A second route to keeping, which needs no decision about where a regime begins. *Follows* round-one prompts 2 (question 4) and 6 (question 4). *Properties:* (a), (b), (e).

```text
Topic. Keeping what was learned by selecting weights and counts with a slowly varying summary of the input, in place of or beside copies stored over time.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search established: compressors of this family handle mixed data by weight sets selected by observable context, not by a memory over time; gated linear networks resist forgetting only when side information separates the tasks; the sharpness of an online, label-free signal of the kind of data at 1 to 4 KB is unmeasured. Here a detector of regimes built on byte histograms split source code into dozens of regimes (its cap of 64 was reached) and did worse than no remedy; stored weight copies removed 38 to 66% of the loss on returning streams (one order).

Questions.
1. Context-dependent gating against forgetting (reported: Masse and colleagues 2018, with task labels): label-free forms in which the gate is computed from the input alone (reported: dendritic gated networks, Sezener and colleagues 2021; active dendrites, Iyer and colleagues 2022). Which use no backward pass between units? Results.
2. Sparse or local codes and interference (reported: French 1991; tile coding; Ghiassian and colleagues 2020): measured retention against dense codes.
3. Slow summaries of a byte stream computed without labels (running histograms of byte classes, sketches of n-grams, slow feature analysis, random projections): how well each separates kinds of data on short windows.
4. Granularity: finer partitions keep more and starve each cell. Theory; soft selection that lets neighbouring cells share.
5. Direct comparisons of selection by input with pools over time, on recurring tasks.
6. Regimes that differ in rule but not in surface statistics: what selection by input cannot do; published failures.
7. For each: what has been shown for persistent learning specifically (keeping, return, no harm to the next thing learned)?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 8. Measuring a life: forward and backward transfer, savings, and the reference systems

*Informs:* The tables of every later report on lives. *Follows* round-one prompt 18 (question 6), where the restatement for log loss was the earlier report's own. *Properties:* (a), (b), (e); measures for (c) and (d).

```text
Topic. Measures of forward and backward transfer and of savings, and their reference systems, for a one-pass stream learner scored by log loss.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search gave prequential evaluation, error bars for dependent sequences, and its authors' own restatement of forgetting, backward transfer and the stability gap for log loss, with no published source for that restatement. Here retention is the loss on a stream's last tenth at the end of a life less the same right after the stream was read; "learning the next thing worse" is the loss at a stream's own stage less that of a fresh system; the reference is six separate systems; five orders of six streams; the systems are deterministic.

Questions.
1. Original definitions (reported: Lopez-Paz and Ranzato 2017; Chaudhry and colleagues 2018, forgetting and intransigence; Díaz-Rodríguez and colleagues 2018), and any published restatement for prequential log loss or code length (reported: Bornschein and colleagues 2022).
2. Reference systems: separate learners, one learner trained jointly, a fresh start. What each controls for; pitfalls (capacity, memory).
3. Savings (faster relearning) as a measure of retention (reported: Ebbinghaus; Hetherington and Seidenberg 1989): definitions, use in continual learning, relation to the loss over the first bytes of a return.
4. Orders of tasks: how many are needed, Latin squares, measures of sensitivity to order (reported: Yoon and colleagues 2020).
5. Aggregating over streams of very different entropy: differences, ratios, ranks.
6. Tests of "no worse than separate systems": non-inferiority margins and their sequential forms.
7. The stability gap in learners that are not neural networks.
8. Which published measures capture revision of what became false, and use on never-shown cases, for persistent learning specifically?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 9. Never-shown cases that survive later learning: test designs

*Informs:* A test of (d) inside a life. *Follows* round-one prompt 5 (questions 4 and 8). *Property:* (d), and its keeping under (a).

```text
Topic. Tests of whether a stream learner uses what it learned on cases never shown, and whether that ability survives later learning.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search established: no benchmark of rule learning inside a stream exists for predictors that are not neural networks; at 10 million words n-gram models score within three points of recurrent networks on minimal pairs, with known artefacts; held-out lexical items are needed. Here the only evidence on never-shown cases is two-digit sums, from an earlier system; on general streams nothing has been measured, and nothing on whether such ability is kept through later streams.

Questions.
1. Artificial-language paradigms that can be scored by probabilities (reported: Saffran and colleagues 1996; Marcus and colleagues 1999; Gómez 2002; Reber's artificial grammars; the "wug" test, Berko 1958): which do n-gram, chunking and count models pass, and under what controls?
2. Compositional benchmarks that can be presented as byte streams (reported: SCAN, Lake and Baroni 2018; COGS, Kim and Linzen 2020): results of learners without backpropagation (grammar or rule induction).
3. Never-shown cases tested after an intervening, different stream (reported: Li and colleagues 2020, compositional continual language learning). Are rules forgotten more slowly than items? Human evidence (gist against verbatim memory).
4. Controls that separate a rule from memory of contexts and from copying by exact match: held-out items, novel combinations, frequency-matched foils.
5. The number of exposures needed, in humans and in models, for the smallest tests that fit inside a stream of megabytes.
6. Does later learning ever improve earlier never-shown performance (backward transfer of rules)? Evidence.
7. For each: what has been shown for persistent learning specifically, as opposed to one training run followed by one test?

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

### Prompt 10. Persistent learning for an agent built on counts: reward, exploration, and the smallest test-beds

*Informs:* The first step from predictor to agent. *Follows* round-one prompt 16, whose report named exploration and the entry of reward as not asked. *Property:* acting as an agent, with (a) to (e) restated for it.

```text
Topic. What persistent learning means for an agent built on counts, and the smallest settings in which it can be tested.

Setting. A small research project builds a learner that reads a byte stream once and goes on learning all its life. It is online context mixing: count tables keyed by contexts of orders 0 to 6 (lifetime counts and, beside them, counts that halve at a bound); match models; a log-linear mixer over the 256 byte values, its weights selected by context type and a confidence bin and learned online from the log loss. Copies of the mixer weights are stored every 65,536 bytes; all copies answer, combined by fixed share.

Aim. Persistent learning: (a) keep what was learned of A through learning B; (b) have it when A returns; (c) revise what has become false; (d) use it on cases never shown; (e) learn the next thing no worse for what is already held; ultimately, act as an agent. Nats per byte are a yardstick only: a better compressor is not the aim.

Constraints. No global backward pass: no gradient passes from unit to unit, through layers or through time (a backward pass inside one unit is allowed). One pass; each byte is predicted before it is learned from (test-then-train). No task or regime labels. One setting for all data.

Facts (development level: one team's programs and checks, mostly one run per order of the streams). An earlier search established: agents built on context-tree predictors reach optimal play on small domains and stall beyond them; a count model good at estimating values was no better than random as a simulator for planning; no lifelong byte-stream agent was found; exploration, and the way reward enters a stream, were not asked about; a count learner grows sure that what it has not seen does not occur. Here the system only predicts the next byte; nothing about acting has been run.

Questions.
1. Reward and action inside one byte stream: action-conditional prediction; prediction conditioned on return (reported: "upside-down" reinforcement learning, Schmidhuber 2019); value from density models. Forms with no backward pass between units; results.
2. Exploration from counts: bonuses (reported: Strehl and Littman 2008), pseudo-counts from density models (reported: Bellemare and colleagues 2016, with a context-tree model), posterior sampling with Dirichlet counts (reported: Osband and colleagues 2013). Behaviour when the environment changes.
3. Non-stationary and recurring environments: switching bandits (reported: Garivier and Moulines 2011); definitions and benchmarks of continual reinforcement learning (reported: Khetarpal and colleagues 2022; Abel and colleagues 2023). Results of tabular or linear agents.
4. The smallest environments in which keeping, return, revision and never-shown cases can each be scored for an agent: protocols, baselines.
5. Interference in tabular and linear agents (tile coding): measurements.
6. State beyond a finite context without backpropagation (reported: utile suffix memory; predictive state representations): what has scaled.
7. For each: evidence over one long life, as opposed to episodes with resets.

For every question: the exact rule or algorithm; what was measured (data, metric, number, year); what is established, what is only argued, what you could not verify.

Standards. Primary sources (papers, code, benchmark tables) before surveys and blogs. Each number with its data set and unit, said each time (bits per character or nats per byte; ours are nats per byte, 1 nat = 1.443 bits). Never present a method that needs backpropagation through layers or through time as meeting the constraints; say where it uses it. Where sources disagree, show both. Write "not measured" rather than infer. Names given here are from memory: starting points to check, not facts. Plain words.

Return. (a) A direct answer to each question. (b) A table of key sources: authors, year, data, metric, number, link. (c) The rules as equations or pseudocode. (d) At most three experiments that fit one day on a four-core machine with 15 GB of memory, each with the result that would count against it. (e) The strongest argument against the direction these questions presume. (f) One paragraph on what such a project is probably missing on this topic that was not asked.
```

## 5. What no prompt covers, and why

These are the helper's judgements.

**Left out because of the steer to persistent learning.**

1. **Better compression and compressor baselines.** Secondary estimation, bit-history states, the distance to packaged context-mixing programs, published ratios. The report answers them (its sections on prompts 3 and 7), and after Mr. Xu's words quoted in section 1 no prompt has them as its aim.
2. **Scaling to gigabytes and beyond, hardware, units and hierarchy, search for contexts, step-size rules.** The report answers each well enough for now (section 2), and none decides the next step for persistent learning.

**Left out because the report already answers them.**

3. **Why the regime detector over-split, and the correct test for a new regime.** Answered under prompt 6 (a persistence prior, a merge step, a sequential code compared with the mixture of regimes). What remains is to run it.
4. **The guarantees of fixed share and of mixing past posteriors for frozen members.** Answered under prompt 1. Round two asks only about members that learn (prompt 1) and about a bounded pool (prompt 5).
5. **Context and the reopening of learning in neuroscience, as a topic of its own.** The report finds that those models approximate methods statistics already has. Two of them return as starting points inside prompt 2 (latent causes and extinction) and prompt 4 (blocked against interleaved learning), where they bear on a decision.

**Left out because an experiment answers faster than a search.** The report proposes most of these under "Experiments for one day"; several are already planned in `nano/lives`.

6. Whether the worth of stored copies survives when streams are woven in blocks and not returned once (`PLAN_10.md`, written, not run).
7. Whether a stored copy that goes on learning closes the gap that a frozen copy leaves (`PLAN_8.md`, written, not run). Prompt 1 asks what the literature knows of the design; the run decides.
8. Whether storing the counts of short contexts with each copy removes the part of the return loss that sits in counts (proposed by the report under prompt 1; not planned).
9. Whether the copies' predictions are combined in probability space or in logit space: a reading of the program, as the report's addendum (item 2) asks.
10. Whether the findings hold over twelve kinds (`PLAN_9.md`, written, not run), and the test of revision itself, a frequent word replaced throughout (planned, not written as a plan at the time of reading). Prompt 2 asks how such a test should be read; it does not replace the run.
11. How sharp a label-free signal of the kind of data is on windows of 1 to 4 KB: the report proposes the offline check under prompt 6. Prompt 7 asks only what others have used as the summary.

**Questions only the owner can settle.** No search can answer them, and the helper attributes no answer to anyone.

12. **What must be bounded.** Whether the learner may keep the raw stream and read it again, or keep everything it has counted. A life so far is under 12 MB. If keeping everything is admissible, property (a) is cheap and the hard properties are (c), (d) and (e); if it is not, the bound has to be stated before "keeps" means anything. Prompt 3 gathers how the field has drawn this line.
13. **Whether work between streams is admissible under "one pass".** Merging stored copies or refitting them from the learner's own store while no stream is read (prompt 5, question 6) is not a second pass over the data, but it is not test-then-train either.
14. **Which of (b) and (c) rules when they conflict.** When a regime returns after one of its regularities has changed, bringing the old state back is keeping, and it is also a failure to revise. The project's working agreement speaks of preserving "the entire A→B→corrected-A lifetime"; what the corrected A should look like on general streams is not written anywhere the helper read.
15. **Whether a summary of the input computed by a fixed recipe** (for example a running histogram of byte classes, prompt 7) is consistent with "no task or regime labels" and with "one setting for all data". The helper's reading is that it is, since nothing is given from outside; the ruling is the owner's.
16. **What counts as an agent for this programme**, and which earlier criteria (complete answers of the system's own, with its own end mark) still bind. Left open by round one (its section 5, item 9) and still open; prompt 10 can inform it but not settle it.
17. **Whether recall by exact match stands under the project's rule against attention-equivalent reads.** Left open by round one; the report gives the literature's view (under prompt 12) and says the ruling "hinges on whether the rule bans learned soft similarity, or any key-based lookup".

**Not done, and said so.**

18. **Checking the report's own unverified items.** Its section "What remains unverified or not found" lists some eighty sources as recalled and not re-fetched. Checking them is verification, not a new question, and is not a prompt here. Those that bear most on persistent learning are named again as starting points to check inside the prompts: the latent-cause model of 2010 (prompt 2), the theory of forgetting in linear regression (prompt 6), follow-the-leading-history and the forget-me-not process (prompts 1 and 5).
19. **The other helper's reading of the report** did not exist when this file was written. If it finds that the report settles something asked here, the prompt concerned should be trimmed before it is run.
20. **The reports of 15 November**, which round one could not read (its section 5, item 11), were not read for this file either. The third prompt of that day asked about definitions of continual learning under bounded resources; if a report for it exists, prompt 3 should be set against it first.

**A check of this file against its brief** (made once by the helper after writing): no prompt names a person, repository, branch, file or product of the project, or any AI product; each prompt carries the setting in standard terms, the aim, the constraints, and the facts with their caveats, and says what round one established; each asks for the rules written out, the measurements with data and unit, a table of sources, at most three one-day experiments with what would count against each, the strongest counter-argument, and one paragraph on what was not asked; each asks what has been shown for persistent learning specifically; none has a better compressor as its aim; each is between 350 and 700 words, counted as strings between spaces.
