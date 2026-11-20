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
