**9.73 E19 (cloud session, 10 November 2026): credit through the second bank's memory can be carried forward exactly, and it helps; under weight signs, births reach about two-thirds of backprop.**

`E19_REPORT.md`, `exp45/SUMMARY_E19.md`, `exp45/E19_CHECK*.json`.

**The network.** `kbank2` is a two-bank learner:
- the first bank's write is a table over the current byte;
- the second bank integrates a *linear* map of tanh of the first bank's state;
- five fixed retention rates, 1.18M parameters.

**The rule.** It carries every gradient forward in time, holding only the two states and traces. For the first bank's table it adds five "deep traces" per entry, one per second-bank rate (K-prop's construction).

**Acceptance (P19.1 holds).** With exact feedback the rule equals autograd's backprop through time to 5e-16 in float64, on random initialisations and on the trained births. A byte-at-a-time streaming version equals the batched one to 7e-16.

**Births.**
- **Backprop (Stage A):** hard mean (add89/sub89 at k = 12) 69.0 / 70.5 / 69.3. That is below the gated two-bank net's 91–95, but it passes the gate of 60 (P19.2 holds).
- **The rule with weight signs (R1):** 43.2 / 45.8 / 54.5, which is 0.63–0.79 of backprop (P19.3 misses in all three seeds).
- **Without deep traces (R2):** 35.2 / 31.3 / 30.7. R1 is ahead by +8.0 / +14.5 / +23.8 (P19.4 holds in two of three).
- **Alignment at the births:** the deep traces raise the first bank's cosine with the true gradient from 0.41 to 0.62–0.69 under weight signs.

**Reading, from the brief's own reading.** P19.3 missing with P19.2 holding puts the limit on the weight-sign feedback, not on the credit path. Every layer above the first bank aligns at only 0.71–0.82 under signs, while the credit path is exact by construction. Two qualifications:
- R1 was still rising in two seeds;
- the one-change arm that would measure this directly — the same forward-only rule with exact feedback — was not in the brief and was not run.

**Cost.** R1 costs 2.1× backprop's CPU per update (220 vs 103 ms) at the same memory (0.5 GB). R2 costs 1.3×.

**Evidence level and caveats.** E2, development panels; one birth per arm and seed. No life is tested. Weight-sign locality remains disputed.
