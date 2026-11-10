# Roadmap note: what is reachable, what is not, and how to stop the muddle

- **From:** the cloud review session of 9–10 November 2026. Companion to `MEMO_TO_FABLE_WORKER_20261109.md` in this folder.
- **Status:** an opinion, written on request. I ran no experiment for it. The odds are my judgment from the repository record and from the literature as I remember it; every literature pointer should be checked before it is quoted.

## 1. Short answers

- **The headline goal as worded is not reachable.** I do not think "baby-level intelligence on a MacBook Pro, without full-circuit propagation" can be reached by this project on any time frame worth planning for.
- **The project is not hopeless.** A narrower digital baby is reachable: a learner that picks up a new word from one or two exposures, keeps it without rehearsal, and uses it in a sentence it never heard, at laptop cost.
- **Muddling through is the default, not a necessity.** The record has at least four attempted resets since late September (the objective contract of 26 September, the research-loop kit of 24 October, the research reset of 2 November, the turn of 6 November). None held for more than days. Section 5 says what would make the next one hold.

## 2. Why the headline goal is out of reach

**2.1 The energy argument does not transfer.** The brain's 20 watts shows what neurons can do. It says nothing about what a laptop can do with the same power.

- A toddler's brain has on the order of 10^14 to 10^15 synapses. A laptop with 64–128 GB of memory holds about 10^10 to 10^11 adjustable numbers. That is a gap of three to four orders of magnitude.
- Nobody knows whether a baby's abilities need all of those synapses. Nobody has shown that better algorithms close the gap either.
- So "nothing forbids it from first principles" is true, and so is "nothing shows it can be done".

**2.2 The methods that get closest break the propagation rule.**

- Small language models trained on child-scale text learn simple grammar, and the small ones train on a laptop (TinyStories, the BabyLM challenge). They are trained by full backprop.
- Inside the rule, the best result in this repository is direct feedback on one arithmetic rule: 75–83% at its peak, against 95–98% for the layer-by-layer sign rule.
- In the wider field, rules without end-to-end error propagation work on small networks and fall behind as depth grows. Training in small blocks with local objectives comes closest on image tasks. As far as I know, nobody has matched backprop this way at language scale.

**2.3 Learning without rehearsal is unsolved for shared weights.**

- A network whose weights are shared across inputs generalizes, and for the same reason new learning overwrites old. Replay is the standard cure and it is what Phase A used.
- A sparse one-shot memory does not have the problem, because each item gets its own nearly private code. That is the mushroom body. The project's first nine weeks showed that such a memory alone stores and does not generalize.
- The two halves are complementary. Neither one alone is a baby.

**2.4 The swatter example is about innate wiring.** A fly's escape from a looming hand is a hard-wired reflex, not something it learned from examples. That is an argument for separating what evolution built from what is learned in life. It is why the ruling in section 6 matters more than any experiment.

## 3. What is reachable

A learner in two parts:

- **A shared model, built before life.** A small language model trained on child-scale text on the laptop. It supplies the representation: similar situations get similar codes.
- **A fast memory, used during life.** A sparse associative memory of the mushroom-body kind. A new experience is one local write. Recall is one lookup. There is no rehearsal and no forgetting until the memory fills.

During life the learner changes only the fast memory, plus slow local adjustments if any are allowed. This is the design the 6 November review sketched in its section 8 before the work moved to arithmetic. It also answers the cost concern directly: learning an event costs one write.

Its limit is plain. The shared model does not gain new abilities during life. Stage 3 below is the attempt to lift that limit, and it is research with long odds.

## 4. Roadmap

| Stage | What | Time | Stop rule | My odds |
|---|---|---|---|---|
| 0 | Write the target and the rules | this week, no compute | nothing starts without it | not applicable |
| 1 | Find where allowed learning rules fail | 1–2 weeks | see 4.2 | about even |
| 2 | Build and test the two-part learner | 4–8 weeks | see 4.3 | about 60% or 25% |
| 3 | Learn new rules during life | 2–3 months, fixed end date | stop at the date | about 1 in 5 |

### 4.1 Stage 0: the target and the rules

Three documents, each one page.

- **The battery.** What "baby-level" means, as tests with pass marks, taken from infant research and not invented here. Candidates:
  - learn a new word from one to three exposures and still know it after other learning;
  - use a new word in a sentence frame it was never heard in;
  - apply a regular ending to a made-up word;
  - prefer a grammatical simple sentence to an ungrammatical one.
  
  Each test needs a shortcut baseline and a history-wide check that the held items were never taught in another form.
- **The rules.** The ruling on the six propagation cases in the memo, in Mr. Zhou's words, with one worked example per case.
- **The budget.** Hours on the reference laptop per run, memory, and the replay share allowed.

### 4.2 Stage 1: where allowed rules fail

Rerun the existing ladder (learn a rule, hold information across a gap, compose, retain) with rules inside the line only. The harness exists and the runs are cheap.

- **Output:** one table, each rung against backprop on the same data.
- **What I expect:** rule learning partly works already. Holding and composition are about even.
- **Decision:** if allowed rules cannot train shared computation beyond the first rung, shared computation has to come from training before life. If that is also ruled out, the learning line stops here, or the rule is relaxed to allow propagation inside small blocks.

### 4.3 Stage 2: the two-part learner

Build the learner of section 3 and score it on the battery, with cost per event reported.

Compare against three controls:

- the same shared model fine-tuned by backprop with replay, as the ceiling;
- the same system with a plain lookup memory in place of the mushroom-body one, which is the first fair test of whether the fly design buys anything;
- no memory at all.

- **Stop rule:** if it fails even with a shared model trained by full backprop before life, the idea has failed in its easiest form and the project should stop.
- **Odds:** about 60% if training before life may use backprop, about 25% if it may not. In the second case the quality of the shared model is the bottleneck.
- **What success is worth:** a working artefact and a fair test of the fly memory. It is mostly assembly of known pieces and should not be described as a breakthrough.

### 4.4 Stage 3: new rules during life

Move what the fast memory holds into the shared weights, using allowed rules and a small replay budget, so that the learner acquires a new rule during life and keeps the old ones.

- This is an open problem in the field. Expect slow progress.
- Fix the end date before starting and stop on it whatever the state.
- A partial result at toy scale is the realistic best case.

### 4.5 What I would stop

- Extending the arithmetic experiments. They have done their job and should stay as a regression suite.
- Adding seeds to settle the 0.80 gate.
- New lives under the layer-by-layer sign rule until the propagation ruling is in.

## 5. What stops the muddle

The agents are fast at running experiments. The record shows what is missing is a fixed point outside their own gates. Three things only the human side can supply:

- **The battery.** What counts as done, written before any run.
- **The rules, with worked cases.** The propagation mismatch was found after four days and hundreds of runs, because the rule lived in conversation.
- **A rhythm.**
  - One decision point a week, with a one-page state.
  - A cap on experiments between decision points.
  - An independent check of claims against the rules before each decision point, not after.

A heavy process will not be used. The research-loop kit of 24 October proved that. Three pages and one weekly habit are enough.

## 6. The ruling the plan forks on

May training before life use full backprop?

- **If yes:** Stage 2 is mostly engineering, and a narrow digital baby is likely within two months.
- **If no:** the project is Stage 3 from the start. I would put the odds of a convincing result within six months at under one in five.

## 7. What would change my mind

- A rule inside the line that trains holding and composition to within a few points of backprop on the existing ladder. That would raise the Stage 3 odds a great deal.
- A published result matching backprop with local rules at language scale.
- A battery pass by the two-part learner with a shared model trained by allowed rules only.
