# Plan 2: connections that turn over, with a trial period (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

## Authorisation and source

- **Owner's instruction (21 November 2026, this session):** change the making rule, after thinking about how biology does it; the plan below was put to him and he answered "可以，跑起来吧".
- **Program:** `TISSUE_EVENT/` of this folder at the commit of this plan. With the new switch off, the program gives the same numbers as at 6cbcd1fab, first form with withdrawal by use, 8,192 bytes: connections, emissions, cell quantities and generator state compared with `==`.
- **Learner:** as plan 1 (E63's tissue, 2,048 cells, neighbourhood radius 1/8, 8 starting connections, the four switches on, 32 steps a byte, no feedback), with withdrawal by use on in every arm.

## Why

Plan 1: the number of connections is set by the making rule. A cell makes a connection every 128 steps while its average activity is below a target that almost no cell reaches, so it connects to almost everything within reach, and what withdrawal by use removes is made again. In the nervous system, as far as the session recalls it (not checked against sources): a neuron is connected to a minority of the cells it could reach; contacts are made and lost all the time and most new ones are transient; a new contact is kept when it becomes useful, that is when its input goes with the receiving cell's activity, and weak, little-used ones are removed. The number is then the balance of a rate of making and a chance of being kept, not a quantity that is filled up.

## The rule (switch `--turnover`)

1. **Making.** Every `--make-period` steps a cell makes one connection to a cell or line within reach that it is not yet connected to, whatever its activity and threshold (the target a* no longer decides making). Feedback cells make none, as before.
2. **Trial.** A connection made from a cell during the life is on trial for `--trial` steps. Its use during the trial is the mean, over the steps at which the receiving cell was active, of the sender's trace times the receiving cell's activity (the quantity withdrawal by use already measures). At the end of the trial it is withdrawn if that use is 0 (including when the receiving cell was never active) or below `--trial-theta` times the mean running use of the receiving cell's other connections from cells that are not on trial; otherwise it stays, and from then on withdrawal by use and withdrawal by strength apply to it as to any connection.
3. Connections from lines are not put on trial (their use is not measured by the program).

## Question and hypothesis

**Question.** With turnover and a trial, does the number of connections a cell stop growing by itself below the size of the neighbourhood, and do the cells near the input lines still carry the byte?

**H2.** In each form, for at least one setting below, the mean number of connections a cell changes by less than 5% over the last 4,096 bytes and ends below 47 (half the neighbourhood), and S(1,0) in window A2 is at least the level 0.1103.

**Refuted if** no setting of a form meets all three.

| No. | Prediction | Chance |
|---|---|---|
| Q1 | At equal making period and trial, the threshold 1 ends with fewer connections than 1/4 (all eight pairs) | 0.8 |
| Q2 | At equal trial and threshold, the making period 2,048 ends with fewer connections than 512 (all eight pairs) | 0.7 |
| Q3 | H2 holds in the first form | 0.4 |
| Q4 | H2 holds in the coincidence form | 0.4 |
| Q5 | In each form, the setting with the fewest connections costs less a byte (core) than plan 1's reference R of that form | 0.3 |

## Arms

Seed 73129; E63's stream of stage A; 16,384 bytes; counts every 1,024 bytes; E63's readings.

| Making period (steps) | Trial (steps) | Threshold |
|---|---|---|
| 512, 2,048 | 512, 2,048 | 1/4, 1 |

Eight settings in each of the two plastic forms: 16 tissues. The references are plan 1's R and W of each form.

## Cost

Forecast from a trial of one setting over 4,096 bytes (first form, making period 512, trial 512, threshold 1/4: 6.5 ms a byte core): at most about 8 ms a byte with records, about 4 minutes a tissue with the run without records, about 15 to 20 minutes on 4 cores for all. Limit 1.5 CPU-hours.

## Not run in this plan

A making period of 128 steps (plan 1's rate; costly with use measured); other seeds; other targets a*; normalisation of a cell's total incoming strength (plan 3); a set point of the cell's own (plan 4); trials for connections from lines.

## Outcome

(to be entered after the runs)
