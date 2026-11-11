# Target test, step 0: generator qualification (report)

Brief `OPUS_TASK_TARGET_TEST_STEP0_20261111.md`, version 5, at `3c928dfa`. No learner was trained. Nothing is concluded about any learner. Standard library only. Python 3.13.

**Files.**
- `target_stream.py`: world, split, episodes, streams, panels, the twin-scoring protocol, hashes.
- `target_solver.py`: the prefix-only solver. Given bytes and the change point. Never sees the hidden values, and imports nothing from the generator.
- `target_oracle.py`: the oracle. Given the hidden values, the change point, and each probe's family, context, cue and position. Reads no stream, and imports nothing from the generator.
- `target_step0_check.py`: the checks.
- `TARGET_STEP0_CHECK.json`: results for world seeds 1, 2 and 3. The memorisation and recency tables are inside the checker and read bytes only.

**Result: every check passes for all three seeds, and every fixture fails as required** (`python3 target_step0_check.py`, exit 0, about 5 s). A check counts as passed only when both hold.

## Counts (identical in shape for the three seeds; sizes in `SIZES`)

| | |
|---|---|
| Per family | 216 taught; 16 + 16 + 16 first-encounter pairs (acquire, retain, revise); 12 sealed twin pairs (one per cue) |
| Revision set, family 1 | Holds 3 pairs of each of the 2 changed cues (seed 1: c, g; seed 2: d, l; seed 3: a, h) |
| Episodes | 3,552: acquire 1,760 (each taught pair 4 times, plus 32 first encounters); family 1 absent 432; return 464; revise 896 |
| Change point | At episode 2,742, 10% into revise |
| Panel probes | A at episode 1,760, I at 2,192 (family 1's return), R at the end; 150 renderings per family each |
| Bytes, seed 1 | Ordinary 46,564; with lines 82,084 (one 10-byte line per revealed episode, 3,552 lines) |
| Redraws | Twins: 0, 0, and 3 in seed 3 family 2. Splits: 0. Revision placements: 1 attempt each |
| **Distinct twin pairs scored** | **12 per family in every world**, beside the 900 renderings per world |
| **Disclosure lines that could not be moved** | **0 in every seed.** The 20 classes (segment × delay × era) hold 16 to 479 lines each |

## Checks, each with its fixture (figures for seeds 1 / 2 / 3)

| # | On the real streams | Fixture, which fails as required |
|---|---|---|
| 1 | Ordinary-stream prefix: 96 first-encounter probes and 144 sealed members, all determined and equal to the oracle. No revision probe lacked the 3-episode, 3-context evidence. Same result on the wider, moved and extra streams | A context taught only with a cue no other context has: its held pair is reported undetermined |
| 2 | Oracle: 900 twins (both members) and 96 first-encounter pairs, none lost, including after the revision | One context value off by one: loses exactly the predicted set (113 / 42 / 77 items) |
| 3 | Prefix solver, complete episode: 900 of 900 twins. Ten-byte suffix (one guess per suffix): 0 twins, 13 / 13 / 26 of 1,800 single members, 0 abstentions | Suffix plus the context byte: 900 of 900 |
| 3b | 10-byte, 40-byte and exact-key tables: 0 twins both right. Every member and every first-encounter pair is an abstention, so collisions are 0, against 0 expected by chance | Twins inserted byte for byte after their probe's preceding bytes: the exact-key and 40-byte tables score 900 of 900 (the 10-byte table 0) |
| 4 | Recency: 0 twins, 13 / 13 / 26 single members. Changed-cue revision pairs: current 0, obsolete 0, neither 6; unchanged-cue pairs: current 0, neither 26 | One context only: 288 of 288 |
| 5 | Ledger over all six streams: 0 held or sealed pairs before their scoring point; 0 sealed occurrences | One held pair inserted at episode 5: reported |
| 6 | 900 renderings: shared 10-byte suffix, different contexts and outcomes, same length, no context byte in the window | A twin whose contexts share a value: rejected (same outcome) |
| 7 | Line symbol frequencies equal. Every line stays in its segment, delay and era, and is true where it stands. Aligned lines 27 / 14 / 19 against 20 expected by chance (p 0.08 / 0.93 / 0.62). The extra stream matches line count and byte placement exactly | Lines left in place: 3,552 aligned, p = 0 |
| 8 | Same seed, same bytes, same hashes | Another seed: every hash changes |
| 9 | Control stream, prefix: every family-1 answer of the 4 control contexts undetermined at every probe, with and without lines. 257 / 250 / 251 episodes replaced, 5 / 5 / 6 of them first-encounter reveals. Same position, cue, filler and era. Replacements are taught pairs of non-control contexts. No family-1 byte names a control context | Families sharing context values: 4,752 determined answers |
| 10 | Stateful responder through `score_twin`: 900 twins, state unchanged, both members answered from one state | A scorer that does not restore: 900 twins with a trace |

## Hashes (sha256, first 12 hex characters; full values in the JSON)

| Seed | ordinary | wider | moved | extra | control | control+lines | sealed f1 / f2 |
|---|---|---|---|---|---|---|---|
| 1 | 55d90a342dfa | 11df65893bf8 | e279398226e6 | cf806b5e0c98 | 40b3f4eee7bc | 91875e51f11f | b9db43c90ca3 / 07cd5da3973e |
| 2 | 958e10b89e3c | a5b1cc8a1787 | 39931eaf3c4d | 4f5bed03f2a6 | bc5644521f58 | b27c309d1dcf | 6b8f124afb6a / 455e533059e2 |
| 3 | b51ea7a1566a | d54cfdac013e | 57cb15eeabec | 80586aafb400 | 8d64f5e3a7e4 | 7f31f47cbf66 | 3d4d1f1a8d91 / 81a24b62842b |

The six panels and six held sets per seed are hashed in the JSON as well.

## What I found unclear, and what I chose (for the audit)

1. **Moved lines can state a held pair outright before its probe.** The slot rule permits it. In the moved stream, a line naming exactly the probed pair precedes 51 / 47 / 55 of the 96 first-encounter probes; in the aligned stream it precedes none. Check 5 counts episodes, not lines, so it passes. But first-encounter scores of the aligned and moved arms would then not be compared on equal terms. A rule such as "a line naming a held pair is not moved before that pair's probe" would remove this. That would be a rule change, so I did not make it.
2. **The disclosure ledger never differs between the aligned and moved arms.** With a line after every revealed episode, both arms have disclosed every context and cue value long before the first probe. The two arms differ only in when a value is told relative to its episode, never in what has been told by a probe. A sparser line schedule (a proposal to revisit) would make the ledger informative.
3. **The interfere phase is two segments for moving lines.** Moving "within the phase" would let a family-1 line land in the stretch where family 1 is absent. Lines move only within the absence or within the return. This is narrower than the brief's wording, not looser.
4. **"Three streams."** The aligned wider-view arm is generated too, since the other two are defined from it. Control versions of the ordinary and aligned streams are generated as well.
5. **Smaller choices:**
   - The solver is told only the change point, so it keeps old and current values apart for every cue, not just the two that changed.
   - First-encounter episodes use 10 to 14 filler bytes, like the panels; the brief does not say.
   - In the control stream a control context's first-encounter pair is still scored, from its own prompt. Its reveal is replaced, which adds taught exposure at that position. Per-position timing is matched; per-pair counts are not.
   - The 40-byte fixture trains on one branch per member, since a single linear stream cannot put both members after the same bytes.
   - Check 7's "more often than chance" uses a Poisson bound (p > 0.001).
   - The memorisation tables never find a key: random filler makes every key new. So check 3b's collision counts are zero and say little.
