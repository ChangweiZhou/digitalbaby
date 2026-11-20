# Line `nano/merge`, ninth plan: a repeat of the frozen setting and of C2 on five seeds no line has used for choosing (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before its runs. Development level; a repeat by the same session with the same program, so not an independent one.

Two-digit sums, 40,000 lines, seeds 2001 to 2005 (used by `nano/ahead2` for its repeat; never read by this line): no classes; the frozen setting; C2 (`PLAN_6.md`). Nothing is chosen. *Predictions (mine):* the frozen setting's median is at least 90 (0.75) and within 2 points of 91.8 (0.6); C2's median is at least the frozen setting's (0.7) and within 2 points of 93.1 (0.5). Fifteen runs, about one CPU-minute each.

## Outcome (entered 20 November; `res/s40r_*_seed200*.json`, `out_s40r_*.txt`)

Never-shown two-digit additions, own answer bytes, whole answer, %, 40,000 lines, mean of the last three readings; seeds 2001, 2002, 2003, 2004, 2005.

| | each seed | median | last reading, median | CPU s |
|---|---|---|---|---|
| no classes | 77.0, 78.1, 78.3, 80.9, 77.1 | 78.1 | 78.0 | 22 |
| the frozen setting | 88.4, 90.4, 93.6, 93.5, 85.2 | **90.4** | 91.0 | 58 |
| C2 | 93.7, 91.1, 96.5, 94.5, 92.9 | **93.7** | 92.6 | 70 |

The base equals `nano/ahead2`'s figures for these seeds (76.9, 78.1, 78.3, 81.0, 77.1 there). All four predictions held: the frozen setting's median is at least 90 and within 2 points of 91.8; C2's is above it and within 2 points of 93.1. The frozen setting is above the base on every seed by 8 to 15 points; its lowest seed (2005, 85.2) is one where the pair of units digits against tens digits is not among the kept pairs at the last finding. With the frozen setting the units-digits classes have 90, 90, 98, 90 of 100 pairs with their class on the four seeds where the pair is kept. A repeat by the same session with the same program: it shows the figures are not peculiar to five seeds, and nothing more.
