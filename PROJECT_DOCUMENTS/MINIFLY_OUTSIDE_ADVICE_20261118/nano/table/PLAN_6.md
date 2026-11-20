# Cells of two memories: each cell of the table gets a companion whose counts fade (roadmap second revision, tracks K, L and C6) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Good. I see you are making progress. Keep experimenting with the agents." **Source.** `claude/friendly-darwin-8mmhfm` at c33512e80; the life, texts and scoring of `PLAN_5.md`; `trust.py` as it stands. Written before any run. Development level; no checker gate.

## 1. The evidence this follows

`PLAN_5.md`: the counting cells with trust keep the first text (a rise of 0.009 nats after three other books), but against a system that read only the new books, having read the first text is worth 0.059 on the first new book and costs 0.042 and 0.027 on the second and third. `PLAN_4.md`, reading 4: a system that goes on counting through the last tenth of the first text is 0.11 nats better there than the same system stopped. Both say that what was read lately should weigh more than the counts let it. Mr. Xu, 20 November: "Also the network could have many different types of cells that adjusts around different time clocks, having different memory ability, ability to output signals (chemicals) etc."

## 2. The one change

Each cell of the table (runs of the last 0 to 6 bytes) gets a **companion** for the same context. The companion counts as the cell does, but its counts fade with the time since it was last active: before it speaks or counts, its counts are multiplied by one half to the power (bytes read since it was last active, over H). It keeps the time of its last activity itself. It says s(v) = log((c(v) + e) / (n + 65 e)) from its faded counts, and is silent when its faded total is under one. The answering cell holds a separate trust for the companions of each length of run, by the same levels (the companion's level of u is the number of its counts of one half or more). Nothing else changes: e = 0.02, the trust's constant step 0.00005, one reading. During a scoring nothing changes and the time stands at the end of the reading.

H is one number. Two values are stated here and both are reported, none chosen: **H = 10,000 and H = 100,000 bytes.**

*Systems.* T0 and T2 of `PLAN_5.md` (their figures stand as the reference), and each with companions at each H: T0+10k, T0+100k, T2+10k, T2+100k.

*Measured,* as `PLAN_5.md` (A after A; A after A and B; B's three held-out parts after A and B, and for a system of the same kind that read B only), and one more: the last tenth of A read with everything going on changing (each byte named before it is counted), after A.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* With companions the system still keeps A and no longer pays for it on the later books: for T2 with companions (at the better H by this measure, both reported), the rise on A is under 0.02, and on each of the second and third books the system that read A first is within 0.005 of the system that read B only, or better. *Refuted if* the rise is 0.02 or more, or the system that read A first is more than 0.005 worse on either book at both values of H.

| | Statement | Chance |
|---|---|---|
| U1 | T2 with companions: the rise on A is under 0.02, at both H | 0.65 |
| U2 | on the second and third books, having read A first costs at most 0.005, at one H at least | 0.40 |
| U3 | with nothing changing, companions change the loss on the last tenth of A after A by less than 0.005 | 0.60 |
| U4 | with everything going on changing, companions lower the loss on the last tenth of A by at least 0.01, at one H at least | 0.60 |
| U5 | B after A and B is at least 0.01 lower with companions than T2's 1.4152, at one H at least | 0.60 |

## 4. Size, cost, what is not run

The table's cells are doubled (a companion for each: 579,255 more cells after A and B) and the answering cell holds 280 more trusts. A life takes under a minute of CPU. Not run here: companions for the cells that step once a word; fading by the number of the cell's own activations in place of time; more than two values of H; the same life for a reference taught by back-propagation and for the relay system (asked of the lines `nano/ref` and `nano/loop`).

## 5. Addition 1 (20 November; written after the four lives were read and before the runs it describes): the return to A with everything going on

The four lives are in (section 6 will give them): with companions the cost on the later books is gone, but the loss on A after B, scored with nothing changing, rises by 0.10 to 0.13 where it rose by 0.009 without them. Reading (mine, to be tested by this addition): at the end of B the companions hold what B said lately and the answering cell has learned to trust them; scored on A with nothing allowed to change, they speak for the wrong text. If that is the reason, a system that is allowed to go on changing when it returns to A should recover within the last tenth of A, and be better there with companions than without. `PLAN_5.md` listed this measure as not run.

*Measured:* after A and B, the last tenth of A read with everything going on (each byte named before it is counted), for T2 and T0 without companions and with companions at both H; beside the same measure after A alone (section 2).

*Prediction (mine):* on the return to A with everything going on, T2 with companions is better than T2 without, at one H at least (chance 0.7); and T2 with companions is within 0.02 of its own figure before B (chance 0.5). *If the first fails*, the companions' cost on A is not a matter of being frozen and the reading above is withdrawn.

## 6. Addition 2 (20 November; written after addition 1 was read and before the runs it describes): companions of both half-lives together

Addition 1 is in: with H = 10,000 the system returns to A better than the system without companions; with H = 100,000 it returns worse. The four lives gave the opposite order for the later books of B (no cost at H = 100,000; a cost on the second book at H = 10,000). So each half-life is good at one thing. The one change here: each cell of the table gets both companions (three memories for each context: one that never fades, one of 100,000 bytes, one of 10,000), each kind with its own trust. Systems: T2 and T0 with both; the same measures, the return to A included.

*Prediction (mine):* on each of four measures (A after A with nothing changing; B after A and B; the cost on the second and third books; the return to A with everything going on) T2 with both companions is within 0.005 of the better of the two single half-lives, or better (chance 0.45 for all four together; 0.8 for at least three).

## 7. Addition 3 (20 November; written after addition 2 was read and before the runs it describes): two faults of my own measure, and a look at where A's loss comes from

Addition 2 is in (both companions together are not the better of the two on three of four measures; its prediction failed). Reading the eight systems side by side shows two faults in how `PLAN_5.md` measured, which did not matter there and do here:

1. Every held-out part was scored at the end of the life. The first and second books of B are then scored after one or two later books have been read, when a companion holds the later book. What a recent memory is for cannot be seen that way. *Change of measure:* each text's held-out tenth is scored **right after its own stream**, with nothing changing, in the life A, B and in the life of B only.
2. A's loss after B was scored with every trust as B left it. *A look inside, not a new system:* the same scoring with the companions silenced (their trusts set to nothing for the scoring), to see whether what the cells that never fade hold of A is intact.

Systems: T2 and T0, without companions, with each half-life, with both. *Predictions (mine):* (a) scored right after its own stream, each of the three books of B is better for the system with companions of H = 10,000 than for the system without, in the life A, B (chance 0.7); (b) with the companions silenced, A's loss after B is within 0.03 of the system that never had companions (1.5067 for T2), at both half-lives (chance 0.6). If (b) fails, the companions have damaged what is kept and not only what is listened to.
