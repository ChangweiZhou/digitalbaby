# The stream of one life (`PREP/life_stream.py`)

Task 26 of the reviewer's `MESSAGES_TO_WORKERS_20261118.md` (19 November 2026). The program uses Python and numpy only, imports nothing from the repository, and runs in about four seconds for the three seeds. `python3 -I life_stream.py` runs the test; its output, for seeds 73129, 93683 and 116089, is in `life_stream_test_output.txt` beside this file.

**What it gives.** `life(seed)` returns one unbroken stream of bytes, about 704,500 for the defaults, and four labels for every position:
- its *phase*;
- its *class* in a first-family item (TASK, A1, A2, SIGN, LET, B1, B2, EQ, R1, R2, END; none elsewhere);
- what is *due* there (nothing, SIGN, EQ, END, RECALL, ANSWER1 or ANSWER2);
- for the later phases, the *gap* of a delay item, or the *table* and *key* of a table item, with a flag on the first asking of each key in the life.

`iter_life(seed)` yields the same data one position at a time. All lengths are fields of `LifeConfig` and can be changed.

**The phases** (default lengths, seed 73129):

| Phase | Positions | What it is | Due |
|---|---|---|---|
| FIRST | 0 to 262,143 | The first family's items as in the step-1 brief, section 4: the record's split (taught half), order, letters and `render`, with the end mark byte 10 and no start symbol | nothing |
| KNOWABLE | 262,144 to 393,225 | The same item stream, continued without a break, as in the step-2 brief. The item in progress at 393,215 is completed | SIGN, EQ and END: the bytes that the last three bytes fix |
| QUIET | 4,096 bytes | The byte 32 (space), repeated. Between every two of the later phases | nothing |
| DELAY | about 131,072 | Items `# X F… = X \n`: a digit X is shown, then a gap of L bytes (L − 1 capital letters, then `=`), then X is due again; L is 1, 2, 4 or 8, uniform | RECALL, the second X |
| TABLE1 | about 65,536 | Items `@ k1 k2 = a1 a2 \n`: a key of two lower-case letters and its fixed two-digit answer, from a table of 16 keys drawn once for the seed; shuffled passes through the table | ANSWER1 and ANSWER2 |
| TABLE2 | about 65,536 | The same, from a second table of 16 other keys | ANSWER1 and ANSWER2 |
| TABLE1_RETURNS | about 32,768 | Table 1 again | ANSWER1 and ANSWER2 |

**The counts for seed 73129** (all three seeds are in the test's output):
- **Phases:** FIRST 262,144; KNOWABLE 131,082; QUIET 16,384 (four stretches); DELAY 131,075; TABLE1 65,541; TABLE2 65,541; TABLE1_RETURNS 32,774.
- **What is due:** SIGN 8,162, EQ 8,162 and END 8,162; RECALL 16,961, of which L = 1: 4,147, L = 2: 4,374, L = 4: 4,296 and L = 8: 4,144; ANSWER1 and ANSWER2 23,408 each.
- **Tables:** each phase asks all 16 keys of its table. Each key's first asking falls in TABLE1 or TABLE2; TABLE1_RETURNS has none.

**What the test asserts, on each seed:**
- **The streams of steps 1 and 2:** positions 0 to 262,143 and 262,144 to 393,215 have the SHA-256 digests printed in the briefs of steps 1 and 2. The counts of SIGN, EQ and END in KNOWABLE are one higher than the step-2 brief's for that range, because the item in progress is completed.
- **The reserve:** a scan of the whole life for the pattern of a first-family item finds every item of phases 1 and 2, none outside them, and no pair of the reserve. The reserve is the second half of each task's split, as in `check_step1_brief.py`. The later phases use no task byte (p, q, v and w are excluded from the keys) and no sign byte.
- **The delay items:** every RECALL byte equals the byte shown L + 1 positions earlier.
- **The tables:** every answer is the one its table gives.

**What a reader should know.**
- **Delays within three bytes:** for L = 1 and L = 2, the byte due is still among the last three bytes before it. A table of the last three bytes could therefore answer those delays; only L = 4 and L = 8 lie beyond it. The test prints this for every L.
- **New byte values:** phases 3 and 4 bring in byte values the first 393,216 positions never had (space, `#`, `@` and the lower-case letters). A tissue meets them for the first time there.
- **What the program does not do:** it computes no countermodel (a table of taught pairs, or the frequencies of answer parts), divides no reserve into early, delayed and after-other-learning parts, and runs no tissue. Those belong to the briefs that will use this stream (task 27 prepares the reserve).

**Choices of this program, which no brief fixes:**
- every length;
- no quiet stretch between phases 1 and 2, so that the first 393,216 positions are the streams of steps 1 and 2 exactly;
- completing the last item of phase 2;
- the quiet byte;
- the forms and marks of the delay and table items, and 16 keys a table;
- the generators `Random(seed * 1000 + 3)` for phase 3 and `Random(seed * 1000 + 4)` for the tables.
