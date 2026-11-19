# Third plan: keeping. Addition, then subtraction, in the cells that learn (19 November 2026)

Authorisation: Mr. Xu, 19 November: "Can you proceed further and do better? ... Feel free to experiment." Same system as PLAN_2.md (256 cells in between, relay with learned return connections, eta1 1/4, eta2 1/16, lambda 1e-5), one more byte value in use (`-`). Written before any run.

**Question.** After 400,000 lines of addition (`aa+bb=ccc`), the same system reads 200,000 lines of subtraction (`aa-bb=ccc`, larger number first). What of addition is still answered, on sums never shown, and does a change that each connection can make by itself help?

**Arms.** `plain`: nothing added. `stiffen`: each strength keeps the running total of the size of its own changes and divides its step by 1 + total/k (a connection much changed becomes hard to change; k chosen on development seed 1 from 0.1, 1, 10 by the smaller of the two scores at the end). `rehearse`: one line in eight of the second phase is an addition line. `mixed`: both from the start, 600,000 lines (reference).

**Score.** Whole answer right on 500 never-shown pairs of each task, at the end of each phase. Five reported seeds as before.

**Predictions.** K1 (0.85): `plain` keeps at most 10% of addition. K2 (0.30): `stiffen` ends with at least 50% on both tasks at the median. K3 (0.70): `rehearse` ends with at least 70% on both. K4 (0.75): `mixed` reaches at least 75% on both.
