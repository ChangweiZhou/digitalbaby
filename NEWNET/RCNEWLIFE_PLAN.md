# Rule cells on a new life (plan, written before any run; 23 November 2026)

Helper branch `claude/newnet-rcnewlife-sei232` from `claude/magical-albattani-sei232` @ 32db8c13. Development level, one machine.
No gate, seal or outside review is claimed.

## Question
Are the persistent rule cells (`net12_rc.py`, rc=64 rc_gate=1; RULECELLS_REPORT.md) tuned to the one test they were found on (t23:
copy, rev, text, succ, copy, rev with delimiters "=", "~", "+")? Test them, with the settings of RULECELLS_REPORT.md and **no
retuning**, on a life that was not used in any design.

## The life (`t27_newlife.py`; the t23 measurement design, new content)
- Lines `w D W\n`; w = 3 symbols of a 40-symbol alphabet: printable bytes 33-126 except `: | # % @`, a different draw than t23's
  (permutation seed 27; 15 of the 40 symbols are also in t23's alphabet; `~` is a symbol here). Starts with 8 symbols, gains one every 150 lines;
  30% of life items contain the newest symbol (as t23).
- Rules (answer W):
  - copy `w0 w1 w2`
  - swap `w1 w0 w2`
  - rot `w1 w2 w0`
  - skip `w0 w2 w0`
  - map: each symbol to its partner in a fixed random pairing of the 40 (20 pairs, permutation seed 127; an involution; not succ).
- Delimiters: cue=1 copy `:`, swap `|`, rot `#`, skip `%`, map `@`. cue=0: all `:`.
- Order (one life, nothing reset), 3,000 lines a rule block: P0 rot, P1 map, P2 text (40,000 bytes of tinyshakespeare from byte
  300,000), P3 swap, P4 skip, P5 rot, P6 map, P7 copy, P8 swap. Returns: rot (P0 -> P5), map (P1 -> P6), swap (P3 -> P8). Copy is met
  only near the end. Total 29,000 line-equivalents (text = bytes / 8), as t23.
- Scores (as t23): first-met answer bytes by layer a / b / n / c per phase (prequential, learning on); right share in windows of 250
  lines from each phase's start; probes with learning off every 1,000 lines (60 fresh items of each of the 5 rules, teacher-forced;
  1,000 never-read text bytes from byte 600,000); total bits of the life and its parts (input / answer / text).
- Layer a uses each rule's source distance (answer byte i comes from w[SRC[i]], 4 + i - SRC[i] bytes back); none for map.

## Runs (seed 1 first; then seeds 2-3 for the two nets if time)
| name | command (t27_newlife.py ...) |
|---|---|
| base | net=12 lagro=1 relay=1 |
| rc | net=12 lagro=1 relay=1 rc=64 rc_gate=1 (net12_rc defaults rc_k=3 rc_tau=8 rc_vig=0.4 rc_eta=0.02) |
| K8 | net=10 (struct K8: L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75) |
| store | net=store |

Each for cue=1 and cue=0. OMP_NUM_THREADS=1, up to 4 processes (nproc = 4). Logs: `rcnewlife_runs/`.

## Predictions (cue=1; the criteria)
1. **rc keeps the learned mapping:** map probe at its last probe (line 29,000, after copy and swap followed P6 map) >= 0.6.
2. **rc keeps every structural rule:** copy, swap, rot, skip probes >= 0.9 at line 29,000.
3. **rc uses fewer total bits than base** (net12 lagro=1 relay=1) on the same life.
All three met at seed 1 = "not tuned to t23" for this life; any one missed = prediction failed (reported as such). Seeds 2-3, if
run, are reported separately; the verdict says how many seeds meet each.

## cue=0
No prediction is asserted. Expectation from RULECELLS_REPORT.md: with one delimiter the rule cells can separate rules only by the
answers already given; the first answer byte of a probe cannot tell the rules apart, so probes should be low for all nets.
Reported as measured.

## Also reported (no criterion)
First-met layers per phase, savings on the returns (P5 vs P0, P6 vs P1, P8 vs P3), text probe after P2 and at the end, number of
rule-cell recruitments, run times.

## Not done
No retuning of any option; no change to existing .py files; no other battery tests.
