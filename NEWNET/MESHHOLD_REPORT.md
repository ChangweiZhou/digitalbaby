# Holders that take part in the network: report (22 November 2026)

Plan: MESHHOLD_PLAN.md (committed at `93383251`, before any run). Branch `claude/newnet-meshhold-sei232`, from
`claude/magical-albattani-sei232@cf5ff1d1`. Development level, not a MiniFly claim. Logs: `meshhold_runs/` (command, exit code,
wall time in each).

**Change (`net12.py`).** This is net10 with `joint=1 mesh=2 mesh_base=0 mesh_dag=1`. Within each byte the context cells now step
first. Each holder of module m then also reads module m's own context cells for that byte, through fixed random strengths N(0, 1) to
its candidate and its gate, scaled by 1/sqrt(number of active cells). `hin=add` keeps the byte's line as well; `hin=ctx` uses the
context cells instead of it. net8c's C step is used unchanged, because the added input is passed in through the per-holder constants
(bc, bg), which never learn with fb=none.

**Correctness check.** One net per process, 1,500 bytes of Moby Dick: net12 `hin=none` and `hin=add wx=0` gave answers identical to
net10's at every byte, and the final outputs differed by 0.0. On t17 at T=2000 / NT=1000, all three score 0.336. (With several nets
in one process the answers differ. net7's C random generator is global to the process, so this check must use one net per process.)

## Every run (L=8, seeds 1 and 2; t17: T=16,000 read, NT=10,000 held-out; t18: context stream G=16, T=30,000, then 200 frozen trials)

| Test | net10 (s1 / s2, mean) | net12 hin=add (s1 / s2, mean) | net12 hin=ctx (s1 / s2, mean) |
|---|---|---|---|
| Moby Dick held-out (tables 0.292 / 0.368 / 0.379) | 0.394 / 0.391, **0.3925** | 0.392 / 0.391, **0.3915** | 0.390 / 0.390, **0.390** |
| Code held-out (tables 0.255 / 0.349 / 0.386) | 0.392 / 0.400, **0.396** | 0.396 / 0.400, **0.398** | 0.392 / 0.399, **0.3955** |
| Memory: share right online, last quarter, at the byte after the gap | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Memory, frozen, branch right: context alone | 0.46 / 0.51 | 0.46 / 0.51 | 0.46 / 0.51 |
| Memory, frozen, branch right: holders alone | 1.00 / 1.00 | 1.00 / 0.99 | 0.99 / 1.00 |
| Memory, frozen, branch right: sum | 1.00 / 1.00 | 0.99 / 0.99 | 0.99 / 0.99 |
| Memory, frozen, full answer over all byte values | 1.00 / 1.00 | 0.99 / 0.99 | 0.99 / 0.99 |
| Wall time each | 100-107 s | 106-115 s | 108-117 s |

## Plain reading

- **The hypothesis is refuted on both streams: it did not help.** The plan predicted at least +0.005 over net10 on the two-seed mean.
  Measured: Moby Dick add −0.001 and ctx −0.0025; code add +0.002 and ctx −0.0005. Every difference is at or below the seed-to-seed
  spread (Moby Dick 0.003, code 0.008). No variant is worse than net10 by more than 0.005, so by the plan's rule nothing hurts
  either. The predicted Moby Dick range of 0.400-0.410 was not reached.
- **Holding is kept.** Context cells alone are at chance (0.46 / 0.51), as predicted: the mesh's paths reach about 10 bytes, fewer
  than the 17 the test needs. The holders alone still carry the branch across the 16-byte gap (0.99-1.00), even when they no longer
  read the byte's line at all (`hin=ctx`). One frozen trial in 200 went wrong in each net12 run (it gave the other end byte). This is
  within the plan's 0.1 bound.
- What these runs show: with fixed random strengths, routing the context cells' codes into the holders gives the output end nothing
  the context compartments and the byte-line holders did not already give it at 16,000 bytes read. These runs do not explain why. A
  possible cause, not tested here: the context compartments already read those same codes directly, so the holders add only a
  slow-filtered copy of the same information.
- NOT_RUN: holders reading their source regions' cells (named in the plan); other wx values; longer reading (T=64,000); L other
  than 8. Total cost: 18 runs of about 2 minutes each at 4 at once (about 33 CPU-minutes), plus the equivalence checks.
