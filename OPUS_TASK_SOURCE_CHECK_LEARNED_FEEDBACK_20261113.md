# Read-only job: source check on learned feedback weights (13 November 2026, 11:55 EST by the clock)

> **Status.** Read-only: no code, no runs, nothing in the repository changes except your memo. Under the rule of review 9.82 a read-only job may start without an audit. Its memo is audited before it steers any design.

## Why

On 11 November Mr. Xu ruled that learned feedback weights are inside the line: a backward pathway separate from the forward weights that learns by a local rule and never copies a weight (review 9.101, addendum). Nobody in this project has opened a source on them. The audit session cited two mechanisms from memory and from a search summary (its note of 11 November, 4.4), and said itself that this needs a source check before it steers a design.

## Questions

1. **The mechanisms.** For each of: Kolen and Pollack (1994); Akrout, Wilson, Humphreys, Lillicrap and Tweed, "Deep Learning without Weight Transport" (2019), its weight mirror and its modified Kolen–Pollack rule; and any later variant you find: what exactly is the update of the feedback weights, what does each synapse need to know, and is any forward weight read by the backward pathway at any time?
2. **Evidence.** What was it tested on, against what, with how many seeds, and how close to backprop? Has anyone replicated it?
3. **Through time.** Has any such mechanism been applied to recurrent weights or through time? If not, say that no such study was found and how you searched.
4. **Against sign feedback.** Does any source compare learned feedback with sign symmetry or sign feedback on the same task? What was the gap?
5. **Unpredictable targets.** Does any source report a feedback-approximation method failing when most targets are unpredictable, or a next-token objective under such a method?
6. **Fit to this project's line.** The line is the four working rulings (review 9.82 addendum and 9.101 addendum). For each mechanism, state plainly whether it copies a weight at any point, including at initialisation, and what a hostile reader would say about it.
7. **Cost.** Extra parameters and extra work per update for a GRU of 256 units with a 512-wide input.

## Rules

Open every source you cite, in this session; give the section, equation or table for each claim; mark each claim confirmed, corrected or not found. Quote at most one short passage per source. Do not design an experiment. End with "The memo decides nothing."

## Deliverable

`PROJECT_DOCUMENTS/LIT_CHECK_20261113/SOURCE_CHECK_LEARNED_FEEDBACK.md`, at most four pages, and one paragraph as `REVIEW_SECTION.md` beside it. Own branch.
