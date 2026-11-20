# The system with nothing given by hand and no convention about time: where does it stand? (roadmap second revision, track L) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "继续大胆尝试，小心求证。" (my translation: "Go on trying boldly, and verify with care."). **Source.** `claude/friendly-darwin-8mmhfm` at 0b1c9d9ac; `gate_all.py` with three options added (no cells that step once a word; the second cell's parts without the level of counts; the score on every eighth byte beside all bytes). With none of them used the program gives `PLAN_16.md`'s figures for T2 again (1.4973, 1.4820; 1,826,213 cells and 79,776 parts at the frozen scoring), and its check passes. Written before any run with the new options. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

The executed check of `PLAN_16.md` (its section 7) left the figure 1.4421 standing as a measurement and named what it leans on: the ends of words are given by hand to three kinds of cell, which hold three quarters of T2's cells; the companions whose counts fade are scored with the time standing still, and the same system has 1.4652 if the time goes on; and the reference network was scored exactly on every eighth byte only. The line `nano/layers` found, on three systems, that the second cell's parts are better without the level of counts in their index, with a twentieth of the parts. Mr. Xu's steer of the afternoon asks for means that are not laid down for one text.

## 2. The question, and what is run

**What does the system reach when nothing about English is given to it and no convention is used?** The bare system: the table's cells (runs of 0 to 6 bytes), the cells that time how long ago each byte value last came, and the pool of cells keyed on two or three of the last eight bytes, both born with no trust (`PLAN_8.md`, `PLAN_9.md`); no cells that step once a word, no companions, no relay system. With the first answering cell, and with the second in two forms: as in `PLAN_15.md` (parts indexed by kind, level and byte) and without the level (by kind and byte; `nano/layers`).

Beside it, each as one more thing: the relay system as a sender; the word cells; both. Measured on the last tenth of the test text with nothing changing (a tenth scored for about seventy-five variants of this line; nothing is chosen on it here), on all 111,540 bytes and on every eighth byte, where the reference line's network has 1.4562 (windows of 129 to 256 bytes) and 1.4541; and with everything going on.

## 3. Predictions (mine, before any run)

There is no mechanism to hold or fail here; these are what I expect of a measurement, written so that I can be wrong.

| | Statement | Chance |
|---|---|---|
| N1 | the bare system with the second cell of `PLAN_15.md`: at or under 1.475 on all bytes | 0.50 |
| N2 | parts without the level are at least 0.002 lower than parts with it, on the bare system | 0.60 |
| N3 | the bare system with the relay system as a sender, in the better form of the second cell: at or under 1.465 | 0.40 |
| N4 | the word cells are worth under 0.010 nats beside the second cell (bare against bare with word cells, better form) | 0.45 |
| N5 | on every eighth byte the bare system's best form is within 0.010 of the reference network's 1.4541 | 0.30 |

## 4. Cost, what is not run

Each run two to four CPU-minutes and 2 to 5 GB, one at a time; the relay system's opinions ten minutes more, once. Not run: the lives with other books (memory, as in `PLAN_16.md`); kinds made while reading in place of the word cells (`nano/merge` has them on its own system); the wider cells of `nano/layers`, which lose on the system with all kinds.

## 5. Outcome (entered 20 November after all runs; `res/gateall_text_*nolevel*.json`, the printed lines in `out20/`)

The last tenth of the test text, nats a byte; first cell, then the second cell with parts by kind, level and byte, then the second cell with parts by kind and byte. "Eighth": on every eighth byte, where the reference line's network has 1.4541 (its figure on all bytes, from shorter windows: 1.4562).

| System | Nothing changing, all bytes | Nothing changing, every eighth byte | Everything going on | Cells; counts above zero; parts used (kind and byte) |
|---|---|---|---|---|
| **bare** (runs, timing cells, pool; nothing given by hand, no convention) | 1.5010, 1.4712, **1.4671** | 1.4994, 1.4721, 1.4714 | 1.3906, 1.3636, 1.3565 | 2.41 million; 13.3 million; 59,795 |
| bare with the relay system as a sender | 1.4920, 1.4643, **1.4583** | 1.4912, 1.4663, 1.4638 | 1.3895, 1.3627, 1.3539 | the same and the relay system's 0.73 million numbers; 60,185 |
| bare with the cells that step once a word (ends given by hand) | 1.4802, 1.4546, 1.4518 | 1.4809, 1.4567, 1.4556 | 1.3683, 1.3465, 1.3414 | 3.77 million; 14.9 million; 60,943 |
| bare with both | 1.4715, 1.4479 (the check's run), 1.4438 | 1.4730, 1.4512, 1.4488 | 1.3673, -, 1.3392 | 3.77 million; 14.9 million; 61,333 |
| all kinds of `PLAN_10.md` (with the companions, scored with the time standing still) | 1.4707, 1.4421, **1.4395** | 1.4733, 1.4463, 1.4451 | 1.3470, 1.3204, 1.3138 | 4.23 million; 15.8 million; 64,012 |

About 90 to 105 CPU-seconds a run; 430 with the relay system's reading.

| | Statement | Outcome |
|---|---|---|
| N1 | the bare system with the second cell of `PLAN_15.md` at or under 1.475 | held (1.4712) |
| N2 | parts without the level at least 0.002 lower on the bare system | held (0.0041 lower; lower on every system here, by 0.003 to 0.006) |
| N3 | bare with the relay system, better form, at or under 1.465 | held (1.4583) |
| N4 | the word cells worth under 0.010 beside the second cell | **failed** (0.0153; with the relay system 0.0145) |
| N5 | the bare system's best form within 0.010 of the network's 1.4541 on every eighth byte | **failed** (1.4714, which is 0.017 above; with the relay system as a sender 1.4638, 0.010 above) |

Readings (mine).
1. **With nothing about English given to it and no convention, the system stands at 1.467 on this tenth (1.458 with the relay system as one more sender), a little above the reference line's network run (1.454 on every eighth byte, where the bare system has 1.471 and 1.464; 1.456 on all bytes from shorter windows).** That is the figure to hold against Mr. Xu's steer. The level of that run is reached, and passed by 0.01, only with the word ends given by hand (worth 0.015) and the companions scored with the time standing still (worth 0.004).
2. **The second cell in the simpler form is the better one everywhere here**, as the line `nano/layers` reported: parts by kind and byte, some 60,000 of them in place of 1.4 million, 0.003 to 0.006 lower. From here on that is the form to use.
3. The second cell is worth 0.028 to 0.034 on every one of these systems; it is the largest single step of the day on prose and the only one of the large ones that is not about English.
4. What is given by hand buys 0.015 and should be found by the system instead; the line `nano/merge` has kinds made while reading that gain as much on books (not on this line's system), and the line `nano/suite` now tests what is general on streams that are not English.
