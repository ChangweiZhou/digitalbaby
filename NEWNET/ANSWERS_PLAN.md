# Whole answers on never-shown combinations: plan (22 November 2026, written before any run)

Helper session for the NEWNET line. Development level; no gate, seal or outside review is claimed. Source: branch
`claude/magical-albattani-sei232` @ `15899397` (latest NEWNET commit read), work on `claude/newnet-general-sei232`. The learner is
`NEWNET/net10.py` unchanged; the two best structures of `SEARCH_SUMMARY.md`:

| name | options | context cells | learned output strengths (approx.) |
|---|---|---|---|
| K8 | `L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75` | 16,384 | 2.1 M (context) + 16 x 262 k (holders) = 6.3 M |
| N1 | `L=8 cC=64 M=32 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4` | 16,384 | 2.1 M + 8 x 262 k = 4.2 M |

(about 16.5 M / 10.9 M stored numbers in all, counted from the arrays at birth; neither is a "mini" model in parameter count.)

## What is measured (`NEWNET/t21_answers.py`)

Three tasks, each its own stream, n_values=128, lines ending in "\n": **add2** `"07+85=092\n"` (2,000 of 10,000 pairs never shown),
**copy** `"abc=abc\n"` and **rev** `"abc=cba\n"` (3 letters of `abcdefghij`; 200 of 1,000 strings never shown). Never-shown items: a fixed
seed (2021), the same for all runs. Life: random shown lines, learning on (teach), 400,000 bytes; a test every 100,000 bytes, learning
off, on 500 shown and 500 never-shown items (copy/rev: only 200 never-shown exist, all 200 are used). **Free** test: feed up to "=",
the structure answers with its own bytes fed back until "\n" or 6 bytes; the whole answer with its newline must be right.
**Forced** test: the right bytes are fed, the share of answer bytes right, by place. Seed 1 for both structures on all three tasks;
seed 2 if time.

Exposure: add2 lines are 10 bytes, so 400,000 bytes are about 40,000 lines, each shown pair read about 5 times; copy/rev lines are 8
bytes, 50,000 lines, each shown string about 62 times.

Reference (not a learner): a table that answers every shown item from memory scores 1.0 on shown and **0 on never-shown** items. A
smoke run of 3,000 bytes (N1, 20 items) showed the script works; its numbers are not results.

## Hypotheses (one per task) and what refutes them

The structures learn by conjunctive context codes (a region answers to a run of recent bytes) read out by delta rules; nothing
in them is shaped for a rule over a variable. Hence the shared hypothesis: **they come to answer shown items far better than
never-shown ones; never-shown whole answers stay near zero**, while single places that depend on few, often-seen bytes are right on
never-shown items too.

| task | predicted at 400,000 bytes, K8 and N1 alike: shown whole / never-shown whole | forced, by place (never-shown) | refuted if (at the last test, either structure) |
|---|---|---|---|
| add2 | 0.10-0.50 / <= 0.05 (each pair read only ~5 times) | hundreds >= 0.85, tens 0.10-0.40, units 0.15-0.50, newline >= 0.95 | never-shown whole >= 0.20, or >= half the shown whole with shown >= 0.10 |
| copy | 0.60-1.0 / <= 0.15 | first letter (4 bytes back) <= 0.5, newline >= 0.9 | never-shown whole >= 0.40, or >= half the shown whole |
| rev | 0.60-1.0 / <= 0.15 | first letter (the byte just before "=") >= 0.95; second and third lower | never-shown whole >= 0.40, or >= half the shown whole |

Also refuted (other direction) if a structure fails to learn the shown items of copy/rev (shown whole < 0.30 at 400,000 bytes): then
the test says nothing about combination, only that the life is too short. The two structures are predicted not to differ by more than
0.10 on any whole-answer share.

## Cost

Timed at birth (one process): K8 3.9 ms a byte learning, 3.1 ms not; N1 2.8 / 2.5 ms. A run: 400,000 bytes of life plus 4 tests of about
20,000 bytes: K8 about 30 min, N1 about 21 min of one CPU. Seed 1: 6 runs, about 2.5 CPU-hours, 4 at once (OMP_NUM_THREADS=1). Seed 2:
the same again if time. A run stops at 1,800 s of life (`tmax=1800`) and records the bytes reached and a last test.

Results go to `NEWNET/answers_runs/*.jsonl` and `NEWNET/ANSWERS_REPORT.md`. No existing file is edited.
