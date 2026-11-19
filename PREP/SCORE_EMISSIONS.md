# The scorer for what a tissue gives out (`PREP/score_emissions.py`)

This is the reviewer's task of 19 November 2026, 05:07 UTC. The program runs on the stream of one life (`PREP/life_stream.py`, commit f4d8f847, which this branch starts from). It uses Python and numpy only and imports only `life_stream.py`. `python3 -I score_emissions.py` runs the test in about 6 seconds. With `--comparator DIR` it also scores the comparator's records; its output is in `score_emissions_test_output.txt`.

## Input

`score(given, L)` takes two arguments:

- **`given`**: one entry for each position of the life. The entry is the byte the tissue gave out for that position, before the byte arrived (0 to 255), or −1 for none.
- **`L = life(seed)`**: the life's bytes and labels.

## Output

For each window and each kind of due position, the scorer returns the following.

**The tissue's figures:**
- `n`: the number of positions of that kind;
- `right`: the share at which the given byte equals the stream's byte. One byte is given per position, so this is right at the first attempt;
- `given`: the share at which a byte was given.

**Four reference guessers**, computed by the scorer from the stream alone:
- `ref_mode`: the most frequent byte;
- `ref_1`, `ref_2`, `ref_3`: tables of the last one, two and three bytes.

Each guesser is fitted only on the positions before the one it guesses (positions 0 to t − 1), over the whole life. It gives the byte seen most often so far, after the same context for the tables. Ties go to the smaller byte value. A context never seen before gives no byte, which counts as not right.

**Ceiling:** `ceiling` is 1 for RECALL and the table answers, where the stream fixes the right byte.

## Kinds of due position

These are the labels of `life_stream.py`, and they match `PREP/comparator_read.py`:
- R1 and R2: the first family's answer digits in phase FIRST, where nothing is due;
- SIGN, EQ and END, in phase KNOWABLE;
- RECALL by gap 1, 2, 4 and 8;
- ANSWER1 and ANSWER2, and the same on the first asking of a key.

## Windows

- **`mode="life"`** (the default): consecutive windows of 16,384 positions from position 0 through the life.
- **`mode="phase"`**: each phase, with QUIET left out, is cut into windows of 16,384 positions. A last window of each phase's final 16,384 positions is added. This is the scheme of `comparator_read.py`.

## The test (seed 73129; all 11 assertions pass)

1. **Always the end mark.** The given share is 1 everywhere. END is right at share 1, and every other kind is right at share 0.
2. **The stream delayed by one.**
   - Pooled over the windows, every kind's share right equals the share of its positions whose byte repeats the byte before. The scorer and a direct computation agree exactly.
   - RECALL and ANSWER1 are never right. ANSWER2 is right at 0.050, where the two answer digits are equal. R2 is right at 0.102.
3. **The 3-byte table itself.** An independent implementation (`collections.Counter`) gives the same guess as the scorer's `ref_3` at all 704,541 positions. Its share right equals `ref_3` in every window and kind.
4. **The comparator.** In `mode="phase"`, the comparator's records for seeds 73129, 93683 and 116089 (`PREP/comparator_runs/*.npz` at a77a3a0c on `claude/e55-comparator-017dSEpW`) give the same counts and shares as `COMPARATOR_READ.json`. All 136, 134 and 136 (window, kind) entries match exactly.

## What the references show (last window of each phase, seed 73129)

- **The 3-byte table:**
  - It answers SIGN, EQ and END at 1.000.
  - It answers RECALL at gaps 1 and 2 at 1.000, but at gaps 4 and 8 at only 0.083 and 0.107.
  - It answers ANSWER1 at 1.000 and ANSWER2 at 0.75 to 0.94.
- **The most frequent byte** is right at 0 on every kind.

So a tissue's score on SIGN, EQ, END, RECALL gaps 1 and 2, and ANSWER1 says nothing beyond a three-byte memory. Only RECALL gaps 4 and 8, and ANSWER2, ask for more.

## Choices of this program, which no brief fixes

- The guessers are fitted on all earlier positions of the life.
- Ties go to the smaller byte value.
- An unseen context gives no byte; there is no backing off to a shorter context.
- The standard error of a share is not computed.
