# Tenth plan: two small language tasks (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, during this work: "If possible, can you also try some small scale nlp tasks while waiting? thanks!" **Source.** This branch at b7975740. Written before any run of either task. Development level; small cases; one development seed first; no checker gate. Nothing here asks for a text to be reproduced exactly: the score is the next byte of text never read, beside tables and a comparator.

## A. The next byte of real prose (`prose.py`; the system of the second plan)

**Text.** `minifly_v148_pretrain_probe_dev_20261025/data/bryant-stories.txt` (Sara Cone Bryant, *Stories to Tell to Children*, from the NLTK Gutenberg selection already in this repository; 249,439 bytes), put in lower case, runs of white space made one space, 36 byte values kept (the letters, the space, nine marks of punctuation) and every other byte made one further value: 37 in all. The first nine tenths are the stream; the last tenth is never read.

**System.** `cells2.Sys` unchanged (one line a byte value at each of 10 delays, 370 lines; cells in between with logistic output; one answering cell with a branch a byte value; the mismatch relayed over learned return connections), run by the compiled kernel. The stream is read eight times, one byte at a time, the system naming the next byte before it arrives.

**Score.** With learning off, the next byte named right, %, at every position of the tenth never read; also the mean number of bits a byte, from the answering cell's branches cut off below at 1e-3 and made to sum to one.

**Arms.** `relay` (the candidate); `comparator` (return strengths that start at nothing and so equal the forward strengths throughout: back-propagation through this network); `fixed` (cells that never change); `none` (no cells in between: the answering cell reads the lines); tables of the last k bytes, k = 1 to 6, counted on the stream, the longest context seen deciding.

**Numbers.** 256 and 1,024 cells; eta1 in 1/16, 1/4, 1 on the candidate with 256 cells, the best used for every arm; eta2 = 1/16, lambda = 1e-5 as before; seed 1.

**Hypothesis.** On real prose the cells taught by the local relay name the next byte of text never read as well as the comparator does, and better than the table of the last three bytes. *Refuted if* the relay is more than 5 points below the comparator, or below the table of the last three bytes.

| | Prediction (mine, before any run) | Chance |
|---|---|---|
| L1 | the relay is within 3 points of the comparator (1,024 cells) | 0.70 |
| L2 | the relay with 1,024 cells is at or above the table of the last three bytes | 0.60 |
| L3 | the relay with 1,024 cells is at or above the best table | 0.30 |
| L4 | no cells in between: at least 5 points below the relay | 0.80 |
| L5 | cells that never change: at least 5 points below the relay | 0.75 |

## B. A verb agrees with its subject across words in between (`agree.py`; the cells of the seventh and ninth plans)

**Text.** Sentences in bytes: `the <noun>[s] <words in between> <verb>[s].` with an end mark. Ten nouns and ten verbs of three letters; the subject is singular or plural at random; a singular subject takes the verb with `s` (`the cat often runs.`), a plural subject the verb without (`the cats often run.`). In between, 0 to 4 words. Text `ag0`: adverbs only. Text `ag1`: each word in between is, one time in three, the phrase `by the <noun>[s]` with a noun and a number of its own. Of the 100 pairs of noun and verb, 20 are never shown together.

**Score.** With learning off, the byte after the verb's third letter (`s` or the full stop) named right, %, on the never-shown pairs, both numbers, by the number of words in between; for `ag1` also apart for sentences whose last phrase in between carries the other number. A system that names the more frequent of the two is right half the time.

**System.** `hold.Hold` with the present byte and the two before it on lines (`chain=2`, so that a three-letter word is on a cell's lines when it ends), 256 cells, K = 16, eta1 = 1; `cross=0`, `cross=1 ret=exact`, `cross=1 ret=grown`; 300,000 sentences; seed 1.

**Hypothesis.** With one crossing the cells hold the number of the subject across the words in between and apply the rule to pairs never shown; without the crossing they do not. *Refuted if* the crossing arm is at or below 65% with four words in between on `ag0`, or the arm without crossing is within 10 points of it.

| | Prediction (mine, before any run) | Chance |
|---|---|---|
| G1 | `ag0`, crossing, no word in between: at least 90% on never-shown pairs | 0.60 |
| G2 | `ag0`, crossing, four words in between: at least 80% | 0.45 |
| G3 | `ag0`, no crossing, four words in between: at most 70% | 0.60 |
| G4 | `ag1`, crossing: sentences whose last phrase in between carries the other number are answered at least 20 points worse than the others | 0.55 |

## Cost, and what is not run

A's runs take seconds to a minute each; B's about three to five minutes. Not run: other books; word-level scores; a second language task read after the first (keeping); the same tasks on reported seeds unless a result is worth it.

## Outcomes (entered 20 November; the account is in `REPORT.md`, section 14)

**A.** Next byte right on text never read, Bryant (Alice): most frequent byte 19.0 (18.2); no cells 35.5 (37.0); 1,024 cells that never change 32.7; relay 45.8 with 256 cells and 46.2 with 1,024 (51.5); comparator 48.9 and 50.2 (55.5); tables of the last 2, 3, 4, 5 bytes 41.3, 49.5, 53.5, 54.8 (44.0, 54.4, 58.3, 59.6). eta1 = 1 was the best of 1/16, 1/4, 1, 4. L1 failed (4.0 points), L2 and L3 failed, L4 and L5 held. Added after: the size of the return strengths at birth (at 0.03 of the earlier size the relay is level with the comparator, 49.2); Alice; two more seeds with 256 cells.

**B.** `ag0`: 100% on never-shown pairs at 0, 2 and 4 words in between with and without the crossing; cells that never change 48 to 50. `ag1`: with a phrase of the other number in between, own path 48, 46 and 51 on seeds 1, 2, 3; one crossing with the reader's own strengths 100 on the seeds run. G1 and G2 held; G3 and G4 failed. Added after: seeds 2 and 3; the arm with cells that never change; `ret=local2` (100).
