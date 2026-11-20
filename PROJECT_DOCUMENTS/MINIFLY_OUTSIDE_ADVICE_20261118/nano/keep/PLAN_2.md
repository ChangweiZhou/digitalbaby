# Keeping, second plan: age (candidate a). 20 November 2026

Line K1 (`nano/keep`); the same authorisation and level as `PLAN_1.md`. Written before any run of this plan; the first plan's runs were in progress and unread.

**Question.** If cells in between join over the whole life at a steady rate, and a cell's steps shrink with its age, do the cells themselves settle where subtraction is written (in the young cells), with the program knowing nothing of when the task changes, and is addition kept as the program's choice keeps it (`reserve=128`: 67.0% with one old line in 32, 4.4% with none)?

**Hypothesis (one).** Steps that shrink with age leave what the old cells wrote in place while the young cells take the new task, so that with one old line in 32 much more of addition is kept than in the plain system (15.6%). **Refuted if**, at the setting chosen on seed 1, the median of five seeds with one old line in 32 has addition below 30% at the end, or subtraction below 50%.

**The learner** (`kage.py`, numpy, the reference; `kage.c` through `fastage.py`, compiled; `check_kage.py` compares every strength of the two on ten cases, agreement to 1e-15, and compares `kage.py` with all cells present and no law against `cells2.py`: 3e-15).
The plain system with 384 cells in between in all (as many as the program's choice has: 256 + 128). `H0` cells are present from the first byte; the others join one by one, cell H0+i at learned byte (i+1) * 5,400,000 / (385 - H0): a steady rate over the 600,000 lines, fixed before the life begins. Nothing in the program refers to the change of task. Before it joins a cell is silent and unchanged; it joins with its birth strengths (answering strength nothing).
A joined cell of age a (learned bytes since it joined) has a rate r(a): `exp`: exp(-a / tau); `hyp`: 1 / (1 + a / tau). The rate multiplies (i) the cell's own steps on its strengths from lines and its threshold, (ii) the answering cell's step on the strength from that cell, (iii) the step of the return strength to that cell, and (iv) the fading of (ii) and (iii). (iv) is needed: a strength that goes on fading at 1e-5 a byte while its steps have stopped is gone after 1.8 million bytes whatever was learned (the error recorded in `loop_small_case/PLAN_4.md`). The answering cell's constants change as in the plain system. Everything else (eta1 1/4, eta2 1/16, lambda 1e-5, the mismatch, the relay over learned return strengths) is the plain system's.
Numbers that learn: 64,910 (53,760 strengths from lines, 384 thresholds, 5,376 answering strengths, 14 constants, 5,376 return strengths), as in the program's-choice arm; 43,278 in the plain system. Each cell has in addition its age, which is not learned.

**One change.** The candidate differs from the plain system in the age mechanism, which has two parts that cannot be separated in one direction (steps that shrink with age without joining stop all learning). The other direction is run as a control: joining with no law (`law=0`).

**Development (seed 1), one old line in 32:** H0 in {1, 128}; law in {exp, hyp}; tau in {30,000, 100,000, 300,000, 1,000,000} learned bytes; and `law=0` for both H0: 18 runs. **Chosen**: the setting with the largest value of the smaller of the two scores at the end. That setting is then run on seed 1 with no old line, and both on the five reported seeds.

**Arms on the five reported seeds** (`plan=add:400000,sub:200000`): plain and the program's choice, with no old line and with one in 32 (run under `PLAN_1.md`); the candidate at the chosen setting, no old line and one in 32; joining with no law at the chosen H0, no old line and one in 32. **Measures:** never-shown addition and subtraction at the end; addition after its own phase.

**Predictions.**
- A1 (0.60): at the chosen setting addition after its own phase is at least 5 points below the plain system's 84.2 (median).
- A2 (0.85): with no old line the candidate ends with addition at 10% or less (median): the young cells speak on addition lines, as the program's new cells do.
- A3 (0.45): with one old line in 32 the candidate ends with addition at 40% or more (median).
- A4 (0.20): with one old line in 32, addition at 60% or more and subtraction at 65% or more (medians): the level of the program's choice within a few points.
- A5 (0.50): joining with no law, one old line in 32, ends with addition below 25% (median).

**Cost.** A run is 5.4 million bytes with up to 384 cells: 80 to 120 CPU seconds expected. Development 19 runs, reported 20 runs: about an hour of CPU, a quarter of that on the clock.

**Not run.** Other laws; a rate that differs between a cell's own steps and the answering cell's; more or fewer cells; a second change of task; silent cells with age; the relay over two steps.
