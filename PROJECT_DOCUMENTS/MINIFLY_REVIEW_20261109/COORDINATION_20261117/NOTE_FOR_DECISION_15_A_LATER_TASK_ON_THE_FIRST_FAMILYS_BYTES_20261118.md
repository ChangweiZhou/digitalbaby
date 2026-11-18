# Note for the coordinator's decision 15: how a later task that shares the first family's bytes could be learned when the cells in reserve receive nothing from those bytes, and what would be measured (18 November 2026)

By the outside reviewer, after the verification of E55's arm `blind` (memo, section 33.26). One page; it proposes no brief.

**The design in question.** A code in which each byte has values of its own, a new birth, and cells in reserve that read only values of bytes the first family does not use. On an item of the first family such a cell receives exactly zero and not a rounding remainder, so that the growth over positions found in `blind` cannot occur, as long as the cell's own biases stay zero.

**The answer.** In that design a task on the first family's bytes is not learned by the cells that receive nothing from those bytes; `blind` shows the same limit by measurement (the task `conflict`: 69.4; 63.2 after 16,384 updates, the cells in reserve adding about 34 points from the one byte they can read). Three conditions are necessary for such a task to be learned while the first family's answers stay, and they differ in what the record says about them.

1. *The later task must present something that the first family does not.* If no byte of an item differs, the same input asks for two answers and no design keeps both. In this world every family has an operation or task byte of its own, so the condition is met.
2. *The cells that learn the task must read the shared bytes; they are then silent on the first family only through a gate of their own, which the byte of condition 1 opens and which stays open over the item.* In E51's `keyed` the program held this gate. In E55's arm `shut`, which is running, a bias holds it and learning may open it. Whether learning opens it on the new task and leaves it shut on the first family is not known; `shut` is the test, and nothing should be designed before it is verified.
3. *The old cells read the shared bytes too, and receive the new task's error.* Unprotected, the first family is lost (E51's `open`: 1.3; 0.0 on `conflict`). The record has two protections: the projection of the old rows, which is an act of the program (E51, E55), and rehearsal with a quarter to a half of the stream (E51's Part B, three seeds), which Mr. Xu accepts in measure. No quantity that a cell holds itself has been shown to protect its rows.

**Consequence.** The second part of item 11 would show condition 2 with exact zeros and would leave conditions 1 and 3 as they are; by decision 14 its acts of the program remain, and it cannot meet requirement A. The reviewer advises against a brief for it as formulated.

**What would be measured, in order.**

- *From `shut`, when verified:* the first family's answers at 22 and at 50 positions; the never-taught score of `conflict` as run and with the reserve held shut; the share of positions at which a gate of the reserve is open by more than a half, on the new task's items and on the first family's.
- *If the gate opens on the new task only:* one experiment on the network that E61 and E62 leave standing, whose input is already a code with values of its own for each byte. The code is born with values for bytes not yet used; the gated cells exist from birth and are not added at the change of task; nothing is projected; the old rows are protected by rehearsal at shares of 0, 1/8, 1/4 and 1/2, with and without the gated cells. Measured: the first family's A and R and the number of its answers that change, at 22 and 50 positions; the new task's never-taught score; the least share of rehearsal that keeps the family within 5 points. The question is whether such cells reduce the rehearsal needed, which is requirement A in the form Mr. Xu gave it.
- *If the gate is not specific:* nothing follows for a design; the next evidence is step 7 of the chain of steps (where an experience is written in the sparse network).

The outside memo's proposal 5.2 (a new family learned in the rows of its own code bytes; decision 35) concerns conditions 1 and 2 and is step 9 of the chain.
