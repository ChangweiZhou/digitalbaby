# Memo: round 23 — Codex V216 reviewed; count-based state-law learner moves object answers below chance (2026-11-04)

## 1. Codex V216 (`74d1f154`): experience pretraining, 8 families, 1.6M events

**Setup:**
- The V181 compact learner (feedback-alignment-style local errors, AdamW; no backprop, as far as I can see in its design).
- Arms: FEW vs MANY environments, versus COLD start.
- Scoring: exact first autonomous prediction of a consequence record.

**Paired contrasts** (`SUMMARY.json`):

| contrast | effect (95% CI) |
|---|---|
| dose | +8.6 pts (3.1–14.1) |
| diversity | +7.4 pts (2.2–12.6) |
| adaptation | +12.9 pts (8.8–17.0) |
| **write effect** | **+1.6 pts (−0.2–3.4), i.e. about none** |

**Assessment:**
- **A proper scale experiment, and the right question.** It is far better than earlier multi-cycle qualification work.
- **The report shows only differences.** Its engineering cycles had 0–1 of 8 absolute, so absolute accuracy is probably low.
- **"Screens passed" is a low bar.** It means about +8 points on a probably small base.
- **The write effect ≈ 0 matters.** The adaptation gain comes from reading the new consequences in context, not from what the weights retain. This is the same lesson as V214B: the headline gain was not the thing we care about.

**Adjustment:** report absolute levels next to every contrast, and treat "persistent write effect" as the primary persistent-learning number.

## 2. This iteration: discover state-update laws by counting (`statelaw.py`)

**Stream:** the round-19/20 MiniWorld log (25 worlds, 1,500 questions, new names per world). It is processed online.

**What is learned:**
- The only labels are the answer tokens that appear in the stream itself.
- No positions, no QA pipeline, no verb list.

| learned item | how (all counting) |
|---|---|
| function vs content words | **dispersion**: a word recurring across ≥3 stretches of life is a function word; world-specific names are not |
| event type | the line's **signature**, its function words with content slots, e.g. `_ put down the _` |
| place-typed slots | per (signature, slot): how often the slot's token later shows up as an answer |
| state | **non-destructive per entity**: an entity's place changes only through place-typed slots. This fixes the V211C overwrite failure by design |
| update law | per signature, the reliability of 4 generic operations, counted from observed answers |

The 4 operations:
- **CO:** co-mentioned place;
- **OWN:** own current place;
- **SNAP:** a co-participant's place *at event time*;
- **PTR:** a co-participant's place *now* (pointer).

**What it learned without being told:**

| signature | learned rule (counted) |
|---|---|
| `_ put down the _` | SNAP 55/56 |
| `_ discarded the _` | SNAP 52/54 |
| `_ grabbed the _` | PTR 92/116 |
| `_ arrived at the _` | OWN/CO 64/64 |

So the learner treats drop verbs as "freeze where it was dropped", get verbs as "follow the holder", and arrivals as "set place".

**Results** (top-1 / answer bits; chance = 2.58 bits; round-20 GLN + recall ≈ 3.0–3.3 bits):

| arm | person top-1 | object top-1 | person bits (1st / 2nd half) | object bits (1st / 2nd half) |
|---|---|---|---|---|
| **full** | **0.89** | **0.76** | **1.21 / 1.21** | **1.60 / 2.41** |
| PTR only | 0.89 | 0.68 | 1.13 / 1.22 | 1.94 / 2.68 |
| SNAP only | 0.89 | 0.48 | 1.14 / 1.21 | 3.08 / 3.65 |
| shuffled signatures | 0.11 | 0.06 | 5.2 / 6.7 | 5.6 / 6.7 |
| no state updates | 0.62 | 0.00 | 2.2 / 2.9 | 5.9 / 6.1 |

**Reading:**
- **First learner below chance on object answers** (1.6–2.4 bits against 2.58). Choosing operations per verb beats either fixed operation (0.76 vs 0.68 / 0.48).
- **Ablations behave as expected:** shuffling the event types and removing the state updates both break it.
- **Same disease check:**
  - This is **not** 99%: object top-1 is 0.76.
  - The probability calibration is crude; second-half object bits are worse than first-half, which I have not explained.
  - Only one stream, one seed.

**Scaffold ledger, honestly:**
- **Still built in:** whitespace words; line = event; a *repertoire* of 4 operations (the same kind of structure as round 12, now with no positions or QA channel); an entity → place store.
- **Learned:** function words, event types, place slots, and the verb → operation mapping.
- The repertoire is the next scaffold to remove. Its operations are all "copy a place from somewhere, now or at event time"; a small grammar of such pointer paths could be enumerated and selected by counts (MDL) instead of listed by me.

## 3. Plan adjustment

1. **Transfer.** Run this learner on the fresh-world assay (new names *and* held-out templates). Report the absolute answer bits for fresh vs experienced learners, and the answer-byte share of the transfer.
2. **Remove the repertoire.** Enumerate pointer paths up to length 2 over (entity, slot, time-of-event / now), and select by code length.
3. **Then plug it into the GLN compressor as an expert,** so total bpb, answer bits and prose bpb are all reported together.
