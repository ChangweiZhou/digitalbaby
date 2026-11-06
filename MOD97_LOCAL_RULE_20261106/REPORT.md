# Mod-97 addition from raw bytes: backprop versus V181's local rule

**6 November 2026 · Development result · three seeds · executed by Claude at Mr. Xu's request**

**Question.** Codex's mod-97 run of 6 November showed that a small network trained by backpropagation answers about 90% of never-seen modular sums from raw bytes. Does the project's local credit-assignment rule (V181's direct-feedback rule) learn the same rule on the same task, or does it only memorize?

**Answer.** The local rule generalizes, partially and unstably. At its peak it answers 70–84% of sealed held sums, far above every shortcut (at most 1.1%) and above any native MiniFly result in the record. It is 15–25 points below backpropagation (97–99% peak) and it does not hold: with continued training, held accuracy falls 10–20 points while taught accuracy stays above 98%. Under V181's own optimizer settings (learning rate 0.001, weight decay 1e-4, dropout 0.1) the local rule never reaches taught acquisition and collapses, while backpropagation under those same settings is the most stable arm of all (95.5–99.7%).

## Setup

Task, split, rendering, model shape, loss and scoring are copied from Codex's `minifly_three_remaining_experiments_20261106` mod-97 campaign (commit 501ca0c: `fixtures.py`, `models.py`, `official_mod97.py`). Records are raw bytes `24+81=08` followed by EOS. Loss is next-byte cross-entropy over every position of the record with a BOS token, as in Codex's `loss`. The model sees an 8-slot ordered window of previous bytes through a learned 8-dimensional byte embedding, then 64 → 128 tanh → 128 tanh → 257 (about 60,000 parameters). The KC association residual is omitted; Codex measured it at 0.9–1.6% of held sums. The split is Codex's `mod_split` with the same seed: 4,704 taught ordered prompts, 2,351 monitoring, 2,354 sealed, both orders of a pair always on the same side. An answer counts only when both answer bytes and the model's own EOS are exact.

Five arms, three seeds each (73129, 93683, 116089), 12,288 updates at batch 64, same data order per seed, scored every 512 updates:

| Arm | Credit assignment | Optimizer |
|---|---|---|
| bp | autograd backpropagation | Adam, lr 0.003, no weight decay (Codex's setting) |
| dfa | V181's direct-feedback rule: the output error is projected to each hidden layer by a fixed random matrix (257×128, scale 1/√257 as in V181's `learner.py`); each weight update is presynaptic activity × projected error. The embedding gets its own fixed random projection | Adam, lr 0.003 |
| dfa_fix | as dfa, embedding frozen at its random birth (a fixed input code) | Adam, lr 0.003 |
| bp_v181 | backpropagation | V181's settings: Adam lr 0.001, coupled weight decay 1e-4, dropout 0.1 on the second hidden layer |
| dfa_v181 | direct-feedback rule | V181's settings |

Qualification (`runs/QUALIFY.json`): the split sizes match Codex's; every rendered answer passes the modular oracle; the swapped form of every sealed pair is sealed; the output-layer and second-hidden-layer local gradients equal autograd's when the feedback matrix is set to the true output weights (with and without dropout); all arms train finitely.

## Results at 12,288 updates (sealed held panel)

| Rule | Settings | Seed | Taught exact | Held exact, final | Held exact, peak (step) | Both orders, final | Carry subset | Own EOS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Backprop | lr .003 | 73129 | 100.0% | 95.8% | 97.3% (5,120) | 93.1% | 94.5% | 100% |
| Backprop | lr .003 | 93683 | 100.0% | 98.5% | 99.3% (3,584) | 97.3% | 98.5% | 100% |
| Backprop | lr .003 | 116089 | 99.6% | 95.4% | 97.7% (6,656) | 91.7% | 95.3% | 100% |
| Local rule | lr .003 | 73129 | 98.5% | 57.4% | 74.9% (4,096) | 38.0% | 60.4% | 100% |
| Local rule | lr .003 | 93683 | 99.5% | 74.1% | 83.3% (3,584) | 58.6% | 76.6% | 100% |
| Local rule | lr .003 | 116089 | 98.9% | 67.5% | 77.7% (5,120) | 49.4% | 69.5% | 100% |
| Local rule, frozen embedding | lr .003 | 73129 | 99.5% | 62.1% | 70.1% (3,584) | 44.1% | 61.6% | 100% |
| Local rule, frozen embedding | lr .003 | 93683 | 99.9% | 78.5% | 84.2% (6,656) | 64.8% | 79.5% | 100% |
| Local rule, frozen embedding | lr .003 | 116089 | 99.6% | 65.5% | 69.3% (5,120) | 49.4% | 68.1% | 100% |
| Backprop | V181 settings | 73129 | 99.7% | 95.5% | 95.8% (10,752) | 93.3% | 94.2% | 100% |
| Backprop | V181 settings | 93683 | 99.9% | 99.7% | 99.7% (9,728) | 99.7% | 99.5% | 100% |
| Backprop | V181 settings | 116089 | 99.7% | 96.1% | 96.2% (11,776) | 93.8% | 97.6% | 100% |
| Local rule | V181 settings | 73129 | 39.3% | 33.7% | 59.0% (6,656) | 23.8% | 30.1% | 100% |
| Local rule | V181 settings | 93683 | 13.9% | 10.5% | 38.4% (5,632) | 4.8% | 6.8% | 100% |
| Local rule | V181 settings | 116089 | 36.0% | 28.6% | 45.4% (5,632) | 16.8% | 31.0% | 100% |

Shortcut ceilings on the sealed panel, from Codex's audit of the same split: exact and canonical lookup 0, frequency 1.0%, nearest taught pair 0.4–1.1%. The sealed panel was scored at every checkpoint in all arms; no arm was selected or stopped by it. Every run took about one CPU minute. Full curves are in `runs/*.json` and `runs/SUMMARY.json`.

## Reading the curves

- **Backprop** reaches 90% held within 1,024 updates and stays within a few points of its peak for the rest of training. Codex's 87.6–93.5% is lower than the 95–99% here because Codex's arm carried the KC residual and stopped at 5,632 updates.
- **The local rule at lr 0.003** reaches 50% held by 1,536 updates and peaks at 70–84% between 3,500 and 6,700 updates. After the peak, taught accuracy keeps rising to 98–99% while held accuracy falls by 10–20 points. The network continues to fit the taught set after it has stopped improving the rule. That is memorization overtaking a partial rule, the opposite of grokking.
- **Learned versus frozen embedding** makes no consistent difference. The input code does not need to be learned for this task.
- **V181's settings** stabilize backprop (the best and flattest curves in the table) and break the local rule. With lr 0.001 and weight decay, the local arm climbs slowly to a taught peak of 50–66% around 6,000 updates, then both taught and held fall for the rest of training. Weight decay removes weights faster than a random-feedback gradient can rebuild them.

## What this settles

1. **The local rule is not why the project saw zero generalization.** Given a task that is learnable at this scale and a shared computation path, V181's own credit-assignment rule produces 70–84% reuse on never-seen inputs. Nothing in the project's native record exceeds a 17-point lift. The difference is the sensor (random expansion, measured at about 1% here) and the task posing, as the review argued, not the absence of backpropagation.
2. **The local rule is weaker and unstable.** It gives up 15–25 points at peak and loses more with continued training. Any lifetime learner built on it needs a stabilizer that backprop does not need. V181's chosen stabilizers (weight decay, dropout, low learning rate) make it worse.
3. **The project's shipped configuration would have failed this task.** V181's rule with V181's hyperparameters peaks below 60% held and ends below 35%. Every earlier negative result on the native learner was obtained with that configuration or a weaker one.

## Evidence level

Development result in a fresh harness, three seeds, no independent confirmation. It reproduces Codex's task and split from the same source code and seed, and the backprop arm exceeds Codex's published score, so the harness is at least as capable as theirs. Within the task this is E3 reuse on never-reinforced inputs with Codex's shortcut audit as the ceiling. It says nothing about MiniFly's native sensor or about tasks other than fixed-width modular addition.

## Next cheap steps (each about one CPU minute per run)

- Stabilize the local rule: early stopping on the monitoring split (that is what the split is for), weight decay on the output layer only, or a lower learning rate without decay. Report peak and final.
- Sign-symmetric feedback (feedback matrix = sign of the forward weights). Under the clarified constraint, a neuron may use information from nearby neurons, and this is the cheapest such rule.
- Taught-fraction sweep (50%, 35%, 25%, 15%) for both rules, with weight decay for backprop, to find where generalization fails for each.
- Decimal addition with carries and variable length on the same network and both rules.

## Reproduction

From this directory, with a Python that has PyTorch (the runs used Python 3.11.5 and torch 2.9.1 on macOS, one thread per process):

```sh
python3 mod97_local.py qualify
for arm in bp dfa dfa_fix bp_v181 dfa_v181 dfa_lowlr dfa_outwd sign; do for seed in 73129 93683 116089; do python3 mod97_local.py $arm $seed 12288; done; done
for f in 0.35 0.25 0.15; do for arm in bp bp_v181 dfa sign; do for seed in 73129 93683 116089; do python3 mod97_local.py $arm $seed 16384 $f; done; done; done
for f in 0.15 0.25; do for arm in bp_v181 sign bpw01 bpw03 bpw10 signw10; do for seed in 73129 93683 116089; do python3 mod97_local.py $arm $seed 98304 $f; done; done; done
python3 summarize.py
```

`mod97_local.py` SHA-256 after the third round: c994c6276c91996f5addf25f92af544c3e37b5e2fc9f8c0a13b6e2c2df73939e (second round: c5baa68dc24367d18a5eddf17252d3c5cccafcfc72d988562166eec73f699438 (first-round file: ad0486f5a71210de14f67e3a5509a7729c6c8ec41ad3424260affffac88b884b; the first-round arms are unchanged in behaviour). Source of the task code: GitHub `ChangweiZhou/digitalbaby`, branch `codex/minifly-three-experiments-20261106`, commit 501ca0c, folder `minifly_three_remaining_experiments_20261106/output/minifly-remaining-experiments/`.

# Second round: stabilizers, sign-symmetric feedback, taught-fraction sweep

**Same day · 45 further runs · about one CPU minute each**

## Stabilizers for the local rule (50% taught, 12,288 updates)

Early stopping needs no new run: every checkpoint already carries monitoring and sealed scores, so the sealed score at the monitoring-best checkpoint is the early-stopped readout. The other arms are new.

| Arm | Sealed held, peak (3 seeds) | Sealed held at monitoring-best step | Sealed held, final |
|---|---|---|---|
| Local rule (DFA), lr .003 (first round) | 74.9% · 83.3% · 77.7% | 74.8% · 83.1% · 77.0% | 57.4% · 74.1% · 67.5% |
| Local rule, lr .001, no decay | 75.4% · 73.9% · 76.5% | 75.4% · 73.9% · 76.3% | 68.7% · 73.9% · 76.5% |
| Local rule, lr .003, decay 1e-4 on the output layer only | 76.7% · 85.4% · 79.3% | 76.5% · 84.5% · 78.8% | 59.3% · 79.8% · 72.2% |
| **Sign-symmetric feedback, lr .003** | **95.3% · 98.2% · 95.5%** | 94.8% · 97.6% · 95.5% | 92.7% · 96.8% · 93.6% |
| Backprop, lr .003 (first round) | 97.3% · 99.3% · 97.7% | 97.3% · 99.2% · 97.6% | 95.8% · 98.5% · 95.4% |

- **Early stopping on the monitoring split recovers the local rule's peak to within one point**, at no cost. The decay after the peak is not damage, it is drift that a held-out monitor catches.
- **A lower learning rate removes the decay but not the ceiling.** At lr 0.001 the local rule is still climbing at 12,288 updates and ends at 69–77%, the same band as the peak at lr 0.003. Taught accuracy is only 90–97%, so it is simply slower.
- **Output-layer-only decay adds one to two points** at peak and does not stop the drift.
- **Sign-symmetric feedback closes the gap.** Each hidden layer receives the error through the sign of its own forward weights, scaled to their mean magnitude, with no transport of the weight values. At birth its hidden-layer updates have cosine 0.75–0.87 with the true gradient, where random feedback has 0.01–0.07 (`logs/QUALIFY2.log`). It reaches 95–98% at peak and 93–97% final, two to three points under backprop, and it crosses 80% held by 1,024 updates like backprop. Under Mr. Zhou's clarified constraint, a synapse reading the sign of the weight it sits on, and a neuron reading its neighbours' error signals, are local operations.

## Taught-fraction sweep (16,384 updates, same sealed panel in every arm)

The taught set is a seeded subsample of the taught groups, both orders kept together, as a fraction of all 4,753 unordered pairs. The monitoring and sealed panels are unchanged, so every cell below is scored on the same 2,354–2,356 sums.

| Rule | 50% taught (4,704 prompts) | 35% (3,294) | 25% (2,350) | 15% (1,415) |
|---|---|---|---|---|
| Backprop, lr .003, peak | 97.3 · 99.3 · 97.7 | 93.1 · 97.8 · 92.8 | 90.1 · 95.6 · 88.3 | 83.5 · 87.5 · 79.0 |
| Backprop, lr .003, final | 95.8 · 98.5 · 95.4 | 87.2 · 92.0 · 86.9 | 79.5 · 86.6 · 76.0 | 64.1 · 71.5 · 58.6 |
| Backprop, wd 1e-4 + dropout, peak | 95.8 · 99.7 · 96.2 | 94.1 · 99.2 · 94.0 | 93.3 · 96.4 · 91.2 | 85.1 · 91.1 · 81.5 |
| Backprop, wd 1e-4 + dropout, final | 95.5 · 99.7 · 96.1 | 93.0 · 98.9 · 93.5 | 91.1 · 94.4 · 89.3 | 73.5 · 78.4 · 67.5 |
| Sign-symmetric feedback, peak | 95.3 · 98.2 · 95.5 | 92.8 · 94.9 · 91.3 | 91.1 · 93.3 · 87.9 | 74.2 · 81.0 · 60.5 |
| Sign-symmetric feedback, final | 92.7 · 96.8 · 93.6 | 81.7 · 87.5 · 81.1 | 71.5 · 77.7 · 72.3 | 46.8 · 57.1 · 41.5 |
| Local rule (DFA), peak | 74.9 · 83.3 · 77.7 | 79.4 · 65.2 · 65.3 | 57.4 · 66.0 · 51.7 | 38.9 · 35.9 · 37.1 |
| Local rule (DFA), final | 57.4 · 74.1 · 67.5 | 55.0 · 56.6 · 46.6 | 34.8 · 32.7 · 31.1 | 21.3 · 13.7 · 17.7 |

Percent exact on sealed held sums, three seeds. Taught accuracy reached 99–100% in every cell. Lookup and nearest-pair ceilings stay at or below 1.1% at every fraction because the sealed panel is unchanged.

- **Every rule degrades smoothly with less data; none falls off a cliff down to 15%.** Backprop at 15% of all pairs (1,415 taught prompts) still answers 79–88% of unseen sums at peak.
- **Weight decay helps most where data are scarce**, as the grokking literature predicts: at 25% it lifts backprop's final score from 76–87% to 89–94%, and at 15% from 59–72% to 68–78%.
- **The local rule needs roughly twice the data of backprop for the same accuracy.** Its 25% peak (52–66%) is below backprop's 15% peak (79–88%). Sign-symmetric feedback tracks backprop within a few points down to 25% and then opens a wider gap at 15%.
- **No delayed rise appeared in any arm within 16,384 updates.** Every curve peaks between 1,500 and 7,000 updates and then declines or holds. Taught accuracy reaches 100% long before held accuracy stops improving, so this is the regime where the grokking literature sees the late rise only after tens of thousands of further updates with weight decay. A 98,304-update probe at 15% with weight decay is reported below.

## What the second round settles

1. **The local rule's instability is manageable and its ceiling is not.** Early stopping on a monitoring split or a lower learning rate removes the post-peak decay. Nothing in this round raised the random-feedback ceiling above about 85% at half the pairs, and it collapses to 36–39% at 15%.
2. **The cheapest rule that the clarified constraint allows reaches backprop's level.** Sign-symmetric feedback is the arm to carry forward if the project wants local credit assignment: 95–98% at half the pairs, 88–93% at a quarter, two to three points under backprop, and learned at backprop's speed.
3. **Data fraction, not compute, is the lever for generalization here.** The whole 48-run sweep cost under an hour of CPU. The project's 80-fact addition tables were two orders of magnitude below the smallest fraction tested here.

## Long probe for a delayed rise (15% taught, 98,304 updates)

| Rule | Seed | Peak held (step) | Held at 16,384 | Held at 49,152 | Held at 98,304 | Taught at end |
|---|---|---|---|---|---|---|
| Backprop, wd 1e-4 + dropout | 73129 | 85.1% (3,584) | 73.5% | 69.8% | 70.8% | 100% |
| Backprop, wd 1e-4 + dropout | 93683 | 91.1% (4,096) | 78.4% | 71.2% | 70.1% | 100% |
| Backprop, wd 1e-4 + dropout | 116089 | 81.5% (3,584) | 67.5% | 64.8% | 65.3% | 100% |
| Sign-symmetric feedback | 73129 | 74.2% (1,536) | 46.8% | 42.7% | 39.1% | 100% |
| Sign-symmetric feedback | 93683 | 81.0% (1,536) | 57.1% | 49.8% | 44.6% | 100% |
| Sign-symmetric feedback | 116089 | 60.5% (1,536) | 41.5% | 35.1% | 35.2% | 100% |

Six runs, about six CPU minutes each. **No delayed rise.** Backprop with the project's weight decay (coupled L2 at 1e-4) falls from its early peak to a flat 65–72% plateau by 16,000 updates and stays there for the remaining 80,000, with taught accuracy at 100% throughout. Sign feedback keeps drifting down. This is the regime the grokking papers describe, except for the strength of the regularizer: Power et al. and Nanda et al. use decoupled weight decay around 1.0, four orders of magnitude stronger than V181's 1e-4. The absence of a late rise here says the project's weight decay is far too weak to produce one, not that the phenomenon is absent from the task. The next knob, if a grokking demonstration is wanted, is decoupled weight decay in the 0.1–1.0 range at 15–25% of pairs, with the same harness. That is about six CPU minutes per run.

# Third round: strong decoupled weight decay, looking for a delayed rise

**24 runs · AdamW, lr 0.001, betas (0.9, 0.98), decoupled decay 0.1, 0.3 or 1.0 · 98,304 updates · 15% and 25% of pairs taught · three seeds**

| Rule | Decay | Taught % | Held peak (step), 3 seeds | Held final | Taught final |
|---|---|---|---|---|---|
| Backprop | 0.1 | 15 | 84.7 (3,584) · 90.9 (3,584) · 85.1 (5,120) | 67.1 · 65.8 · 61.7 | 100 · 100 · 100 |
| Backprop | 0.3 | 15 | 88.8 (11,264) · 93.7 (17,408) · 89.0 (9,216) | 84.8 · 88.5 · 82.4 | 99.9 · 100 · 100 |
| Backprop | 1.0 | 15 | 86.9 (28,672) · 91.1 (81,408) · 85.9 (90,112) | 85.0 · 90.2 · 83.4 | 90.8 · 93.4 · 88.8 |
| Sign feedback | 1.0 | 15 | 87.9 (58,880) · 84.8 (32,768) · 65.9 (48,128) | 86.9 · 84.3 · 62.0 | 90.4 · 89.6 · 74.2 |
| Backprop | 0.1 | 25 | 93.0 (8,192) · 98.2 (11,264) · 92.7 (10,752) | 89.2 · 94.6 · 87.2 | 100 · 100 · 100 |
| Backprop | 0.3 | 25 | 96.6 (36,864) · **99.4** (31,744) · 94.4 (14,336) | 95.8 · **99.2** · 91.1 | 99.9 · 99.9 · 99.7 |
| Backprop | 1.0 | 25 | 89.6 (57,344) · 91.9 (85,504) · 86.5 (97,792) | 87.0 · 90.4 · 83.7 | 88.7 · 92.4 · 86.4 |
| Sign feedback | 1.0 | 25 | 89.2 (91,648) · 89.0 (25,600) · 89.3 (11,776) | 88.2 · 87.1 · 88.0 | 90.6 · 90.9 · 90.5 |

Percent exact on the same sealed panel as every earlier round. Earlier rows for comparison at 15%: coupled decay 1e-4 ended at 65–72%, and no regularizer at 59–72%.

**Decay 0.3 is the best regularizer in the study.** At 25% of pairs it reaches 94–99% held with taught at 99.9%, the highest cells anywhere in the three rounds, and holds them for the rest of training. At 15% it ends at 82–89% where the project's decay 1e-4 ended at 65–72%. Decay 0.1 still drifts down after an early peak. Decay 1.0 is too strong at this learning rate: the network cannot memorize (taught caps at 86–93%) and whatever it learns is the rule, so held tracks taught.

**Strong decay also repairs the local rule.** Sign-symmetric feedback with decay 1.0 ends at 84–88% held at both 15% and 25% (one 15% seed at 62%), where without decay it drifted to 35–45% at 15%. The drift that early stopping had to catch is gone.

**There is still no grokking, and the sweep shows why.** Grokking is a specific signature: taught accuracy saturates, held accuracy stays low for a long stretch, then rises late. One seed (93683) at decay 1.0 does sit near 30% held for 40,000 updates at 25% and then jumps to 90% within 3,000 updates. But its taught accuracy sat at 32% through the same stall and jumped with it (taught 31.9% → 66.3% → 85.5% → 91.3% across the four checkpoints of the jump; see `runs/bpw10_93683_f25.json`). That is a late escape from a partial solution, with taught and held moving together, not memorization followed by generalization. At decay 0.3 the held curve tracks the taught curve from the first checkpoint (seed 93683 at 25%: taught 94.6% and held 89.4% at 2,048 updates; 99.9% and 99.2% from 16,384 on). At decay 0.1 and below, held peaks early and decays while taught stays at 100%. Across 30 long runs there is no regime in which this network memorizes the taught set first and finds the rule later. On this task with a shared MLP, the rule is found at least as fast as the table, so there is nothing to wait for.

**Reading for the project.** Mr. Zhou's question 2 asked why 99% taught accuracy never led to grokking. The answer from these rounds is that grokking is not a milestone that generalization has to pass through. It appears only when memorization outruns rule-finding, which needs a model and task where the table is the cheaper fit for a long time. Here the table was never cheaper at 25–50% of pairs, and at 15% the right regularizer gives 82–89% directly. The project's 80-fact tables with a random-expansion memory are the opposite regime: the table is always cheaper, and nothing in the loss ever pushes toward the rule. What to carry forward is decoupled decay near 0.3 for the shared path, and decay near 1.0 if the sign-feedback local rule is used.
