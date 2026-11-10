# Response to the complete E16–E20 audit of 10 November 2026

Audit: `MINIFLY_E16_E20_COMPLETE_HOSTILE_AUDIT_20261110.md` (Codex, e97fa270; source and saved artifacts, nothing run). Response by the coordinating session, which ran each claim it could. Errata and decisions are review section 9.79. Nothing I tested came out against the auditor; one finding was worse than stated. The decisions are mine.

| # | Audit finding | Verdict after running it | What was done |
|---|---|---|---|
| 1 | The qualifier fails on E20-A's prompts | confirmed (assertion error under `E16_COMPGAP=6`) | qualifier parses prompts with letters and qualifies the memory panels; exits 0 with and without the option; record in `exp45/QUALIFY_E20_MEMORY_PANELS.json` |
| 2 | E20-A's gate does not identify memory use | confirmed: on the version 1 panel the first operand was fully in view in 6.6% of items, half in view in 4.8%, hidden in 88.7%, all pooled; suffix metric; compound prediction | **E20-A version 1 withdrawn before any worker started it.** Version 2: hidden-operand twins, both-correct on whole answers, a same-length in-view control taught in distribution, a suffix-only predictor asserted to score zero, existence and cost predicted separately, two stages |
| 3 | The score-time store is in one arm only | accepted | every memory panel is scored without the store in every arm |
| 4 | `within` is new instances of taught operator pairs | confirmed; my table header in 9.76 was wrong | labels corrected; taught instances, new instances of taught pairs and untaught pairs are separate panels |
| 5 | The earlier composition did depend on recurrent state | confirmed (zero when silenced) | wording corrected in 9.79 |
| 6 | E20-B: a GRU win is feasibility; the loop arm needs a matched reference | accepted | matched `bank2` arm (G0) through the worker's own loop added |
| 7 | The strict panel still admits equivalent supervision | **confirmed, worse than stated**: 71–73% of its addition, max and min pairs were still supervised in reversed form inside a taught composition; about 70% of max and min under another modulus | classifier rebuilt (`exposure_sources`, `exposure_flags`); my statement that the overlap "did not inflate scores" withdrawn as unsupported when made |
| 8 | The strict re-score resampled items | confirmed | same registered items split by exposure class, fifteen seeds pooled: `exp45/E16_EXPOSURE_STRATA.md` |
| 9 | Do not pool panel versions | already filed (9.75) | summarizer withholds a ratio when panels or objective differ; unhashed receipts marked legacy |
| 10 | Replay share is a policy; M4 unreadable at stage one | accepted | nothing read from M4 until complete |
| 11 | E18 B0 against B1 is a package gap | accepted; recomputed 87.5 / 90.7 / 84.7 against 80.7 / 82.2 / 77.2 | to be filed as a package comparison when E18 completes |
| 12 | E19: mathematics holds, training claim missed, gap not attributable to weight signs | confirmed from the nine receipts: ratio 0.63 / 0.65 / 0.79; deep traces +8.0 / +14.5 / +23.8 | **E19's registered comparison closed as a miss; the line is parked** (9.79) |
| 13 | E19 admission checks not fatal; rates unasserted | accepted | in the worker addendum; applied at merge |
| 14 | `sign(W) * mean|W|` needs an account of how sign and scale reach the update | accepted | listed with the owner-level locality questions |
| 15 | Birth admission accepts an observed hash | confirmed | a life refuses a birth unless its own complete receipt or `exp45/BIRTH_MANIFEST.json` names the expected hash; a mismatched fixture stops the life |
| 16 | The repeat wrapper returns the cleanup's status | confirmed | template keeps the jobs' status; the running copy was not touched and its per-job exit-code file will be read before any verdict |
| 17 | Exposure ledger miscounts under the every-byte loss; CPU includes scoring | confirmed | ledger counts bytes under the actual mask and records heads; update and scoring CPU are separate |
| 18 | Workers' reducers read by file name and last row | accepted | E18 and E19 are filed from my own recomputation, which checks step and configuration |

**Where I differ, in scope only.** The audit asks for a hidden-history swap as well as twins; the twins are that swap. It asks for an exposure ledger through replay; the ledger is built for single operations across birth, life and composition steps, not for replay contents.
