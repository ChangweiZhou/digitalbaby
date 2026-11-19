# Independent inventory of what the program fixes or supplies (E50 and E62 network, harness, and sw_run.py)

Written 2026-11-18, 21:02 UTC (clock read with `date -u`). Made from the source code in `e61src/` and, for the small-world runner's companions, from `sw_net.py`, `lw_world.py` and `sw_world.py` on the branch `origin/codex/minifly-consolidated-20261023`. No report, memo or other document was read. Line numbers refer to the copies in `e61src/` unless another file is named.

## Terms used

- **Cell**: one of the 256 recurrent units (U = 256). Each cell has three gates (r, z, n), so it owns three rows of each map that feeds it.
- **Value**: one of the 512 entries of the fixed input code (D = 512).
- **Readout unit**: one of the 512 units of the hidden readout layer z2 (H = 512). **Output unit**: one of the 257 units of l3.
- **Crossing**: one application of the recurrent map, which moves the state one connection further.
- **Exact rule**: `local=False`, gradient by automatic differentiation. **Sign rule**: `local=True`, where the error crosses each map through sign(W) times one scale per map.
- **Classification**: **N** = something a neuron has (its own state, or what arrives over its own connections, used by itself); **X** = supplied or fixed by the experimenters; **U** = unclear. Each entry gives the reason in one sentence.
- **Setting `C46-one`**: in `exp50_code.py`, SETTINGS["C46-one"] = (46, 256). Each cell reads 46 input values; each readout unit reads all 256 cells.

---

## 1. The inventory

### (A) Time and order of computation within an item

**A1. The state is set to zero at the start of each item.** `e50_sparse_code.py` L55 (`h = T.zeros(B, U)`); also `arch26.py` L57 and `arch20.py` L76.
What it does: every row of a batch (one item) starts from h = 0, and nothing is carried from one item to the next.
Class: X. The program erases the state at a boundary it defines, so nothing held in the state can last beyond one item.
Numbers: initial state 0.

**A2. A start symbol is placed before each item.** `exp45.py` L160-164 (`data` filled with BOS = 257; the row starts at index 1); `exp11_cell.py` L196 (scoring).
What it does: position 0 is always byte 257, which has its own row in the input code (A15, F7).
Class: X. The program marks where an item begins.
Numbers: BOS = 257.

**A3. One synchronous crossing per byte, and every cell updates at every byte.** `e50_sparse_code.py` L48 (`assert layer.K == 1`), L56-61; `e42_cells.py` L22 (PERIOD_SETS["1"] = {1: 256}), L43 (phase = randrange(1) = 0).
What it does: at each byte, every cell reads the states of its sources from the previous byte, once, and all 256 cells update together.
Class: X. The program fixes that cells exchange what they hold exactly once per input byte, and gives no step without input.
Numbers: period 1 for all 256 cells; phase 0; one crossing per byte.

**A4. The input of the current byte and the recurrent input arrive at the same step.** `e50_sparse_code.py` L52 (input map for all positions), L57-60.
What it does: the gate pre-activations at byte t are the sum of the input map applied to the code of byte t and the recurrent map applied to h(t-1).
Class: N for the sum inside the cell; X for the timing. The cell combines its own inputs, but the program decides that input and relay share one step.
Numbers: none.

**A5. The cell's update equations are fixed GRU equations.** `e50_sparse_code.py` L58-60.
What it does: r = s(i_r + h_r), z = s(i_z + h_z), n = tanh(i_n + r * h_n), h' = (1 - z) n + z h.
Class: N, with its form fixed by X. The computation uses only the cell's own pre-activations and its own previous state, but the experimenters chose its form.
Numbers: three gates; sigmoid and tanh; n uses r * h_n (the PyTorch form).

**A6. The readout at a byte reads the cell states of the same byte and the code of the same byte, with no delay.** `e50_sparse_code.py` L62-67.
What it does: z2(t) = tanh(l2 code(x_t) + lm h(t) + biases); logits(t) = l3 z2(t). The output at byte t is formed in the same step as the state.
Class: X. The program chooses that output is due at once, with no step of its own.
Numbers: no delay.

**A7. The first answer byte is due at the step that reads `=`.** `exp45.py` L166-168 (mask positions len(p) .. len(p)+len(t)-1 of y); `exp11_cell.py` L231 (prompt ends with `=`).
What it does: the target at the position where `=` is the input is the first answer digit; the second digit and the end symbol are due at the next two positions.
Class: X. The item format leaves no step between the end of the question and the answer.
Numbers: three target positions per item (two digits and EOS).

**A8. Between the last digit of the second operand and the first answer byte there is exactly one crossing.** Follows from A3 and A7 (`exp11_cell.py` L231: `... b b =`).
What it does: the second digit of b is read at step len(p) - 1; `=` is read at step len(p), where the first answer is due; so one crossing separates the last operand byte from the first answer byte, and one crossing separates each answer byte from the next.
Class: X. The program fixes how far information can travel between cells before an answer is scored (see section 2, item 3).
Numbers: 1 crossing before the first answer byte; 1 crossing between answer bytes.

**A9. During learning the answer bytes are given as inputs (teacher forcing).** `exp45.py` L160-165 (`x = data[:, :-1]` contains the target bytes).
What it does: when the second answer digit is due, the input is the true first digit, not the one the network produced.
Class: X. The program supplies the correct answer as input during learning.
Numbers: none.

**A10. During scoring, each generated byte is fed back and the whole item is recomputed from zero.** `exp11_cell.py` L188-214 (respond), with `E.forward` replaced by E50's forward (`exp50_code.py` L137-145).
What it does: greedy argmax over the 257 outputs at the last position; the byte is appended and the forward is run again from h = 0 over the whole prompt; at most 4 bytes; stops at EOS.
Class: X. The program generates and judges; the recomputation gives the same values as continuing, so it changes no result.
Numbers: cap = 4 generated bytes; groups of at most 256 prompts.

**A11. Credit is carried back through every earlier step of the item.** `e50_sparse_code.py` L56-61 (h is never detached); `exp50_code.py` L151-155.
What it does: under both rules, the error at an answer position reaches the weights used at every earlier step of the same item, through stored states, in reverse order of time.
Class: X. This is a backward pass over the whole item that the principle prohibits (section C).
Numbers: none.

**A12. Weights are constant during an item and during a batch; one change follows the whole batch.** `exp50_code.py` L148-156.
What it does: forward on 64 items in parallel, one backward pass, one optimizer step.
Class: X. The program decides when weights may change: never within an item.
Numbers: 64 items per change.

**A13. After the end symbol, positions are filled with BOS and carry no error.** `exp45.py` L162-168.
What it does: rows of different length are padded to the longest row plus one; the padding comes after the item and does not affect earlier states.
Class: X (implementation of batching). It has no effect on learning.
Numbers: none.

**A14. The end symbol is read as an input after the second answer digit, with a zero code.** `exp45.py` L161-165; `e50_sparse_code.py` L24 (256 is not in USED, so its row of the table is zero).
What it does: at the position after the last target, the network reads EOS, but no target is set there, and EOS gives no input to any cell.
Class: X. The end of an item is defined by the program, and the network receives no input when it occurs.
Numbers: EOS = 256.

**A15. Letters between the operator and the second operand delay the second operand.** `exp11_cell.py` L229-231.
What it does: k uppercase letters are inserted after the operator; the first operand must be held across k + 3 crossings before b arrives.
Class: X. The program fixes the gap and draws it.
Numbers: k in 0..12 during learning; 12 and 40 at scoring (E7).

### (B) What a cell reads and what reads a cell

Weight counts for `C46-one` (my own count with a numpy-free script; the total equals the program's own constant `WEIGHTS_TAKING_PART["C46-one"] = 568,576`, `exp50_code.py` L59):

| Map | Shape in use | Weights in use | Share of total in use |
|---|---|---|---|
| Input map (code value to cell gate), `rnn.weight_ih_l0` masked | 768 rows x 46 values | 35,328 | 6.21 % |
| Recurrent map (cell to cell gate), `rnn.weight_hh_l0` masked | 768 rows x 11 sources | 8,448 | 1.49 % |
| lm (cell to readout unit), dense | 512 x 256 | 131,072 | 23.05 % |
| l2 (code value to readout unit), dense | 512 x 512 | 262,144 | 46.11 % |
| l3 (readout unit to output unit), dense | 257 x 512 | 131,584 | 23.14 % |
| **Total weights in use** | | **568,576** | 100 % |

Biases trained in addition: bias_ih 768, bias_hh 768, lm 512, l2 512, l3 257 = 2,817. Parameters present but without use or gradient: book 2,064, l1 4,608. The maps that belong to cells (input map and recurrent map) together hold 7.70 % of the weights in use; the readout maps (lm, l2, l3) hold 92.30 %.

**B1. A fixed input code: each used byte turns on 11 values of its own, out of 512.** `e50_sparse_code.py` L22-35.
What it does: a table of 258 x 512; the 46 bytes in USED each own 11 disjoint values drawn from a shuffle; every turned-on entry has the same amplitude; all other entries are zero; the table never changes.
Class: X. The code is made by the program, not formed by the learner; by the principle it is the experimenters' information.
Numbers: D = 512, ON = 11, 46 used bytes, 506 values owned, 6 values never on, amplitude 3.2387 (F6).

**B2. Input map: each cell reads exactly one value of each of the 46 used bytes, the same 46 values for its three gates.** `exp50_code.py` L81-91, L113-122.
What it does: for each cell and each used byte, the program picks one of the byte's 11 values; the cell's three gate rows read only these 46 values.
Class: X. The program uses its knowledge of which value belongs to which byte to ensure that every cell receives input for every used byte.
Numbers: 46 values per cell; 35,328 weights; each value is read by 9 to 38 cells (my count, seed 73129); no owned value is unread.

**B3. Input biases.** `e50_sparse_code.py` L51-52 (`bias_ih_l0`).
What it does: one bias per gate row, added with the input.
Class: N. A bias is the cell's own number.
Numbers: 768.

**B4. Recurrent map: each cell reads 10 other cells drawn from a fixed neighbourhood on a 16 x 16 torus, and itself.** `e42_cells.py` L21, L28-46, L77-79; `exp50_code.py` L112 (`N=10`).
What it does: the neighbourhood of a cell is the 48 cells within Chebyshev distance 3 on a 16 x 16 sheet with wrapped edges; 10 are drawn once by `Random("e42|connections|seed")`; the three gates read the same 11 sources.
Class: X. The geometry, the number and the draw are the program's; the cell does not choose or change its sources in E50.
Numbers: SIDE = 16, RADIUS = 3, neighbourhood 48, N = 10, 11 sources with itself, 8,448 weights. My count of the directed graph for seed 73129: diameter 5 crossings, mean distance 2.84, every cell reaches 11 cells in 1 crossing, 74 in 2, 213 in 3, all 256 in 4 to 5; a cell is read by 3 to 17 cells; 520 of 2,560 connections are reciprocal.

**B5. Each cell always reads itself.** `e42_cells.py` L77 (`self.mask = T.eye(U)`), L194.
What it does: the self-connection is forced in addition to the GRU's own z * h path.
Class: X. Fixed by the program.
Numbers: 1 per cell (3 weights).

**B6. Recurrent biases.** `e50_sparse_code.py` L53 (`bias_hh_l0`), used inside r * h_n for the n gate.
Class: N.
Numbers: 768.

**B7. The connections do not change in E50.** `exp50_code.py` L319-427 never calls `e42_cells.replace`; the masks are checked unchanged every 256 updates (L378-381).
What it does: the cell's set of sources is fixed for the whole run.
Class: X. A neuron cannot change which connections it has, only their weights.
Numbers: none (E42's replacement constants EVENT = 256, MIN_AGE = 512 and the threshold exist but are unused here).

**B8. lm is dense: each of 512 readout units reads all 256 cells.** `e50_sparse_code.py` L65 (`iface.k_out == U` gives `lin(Hs, m.lm, local)`); `exp50_code.py` L55.
What it does: every readout unit sees the whole state of the network at each byte.
Class: X. This is a quantity formed from the whole network and delivered to the output, contrary to "learning local also at the output".
Numbers: 131,072 weights, 512 biases.

**B9. l2: the readout reads the fixed code of the current byte directly, without passing through any cell.** `e50_sparse_code.py` L66 (`lin(z1, m.l2, local)` with z1 = table[x]).
What it does: a dense 512 x 512 map from the code to the readout layer, added to lm h.
Class: X. A path from input to output that bypasses the cells and holds 46 % of the weights in use.
Numbers: 262,144 weights, 512 biases.

**B10. A hidden readout layer of 512 tanh units between the cells and the outputs.** `e50_sparse_code.py` L66-67.
What it does: z2 = tanh(l2 code + lm h); logits = l3 z2; a two-layer dense network outside the cells computes the answer from all cells and the current byte.
Class: X. The structure and its width are taken over from the dense network (`arch20.py` L44-50).
Numbers: H = 512.

**B11. l3: each of 257 output units reads all 512 readout units.** `e50_sparse_code.py` L67; `arch20.py` L50.
Class: X.
Numbers: 131,584 weights, 257 biases; 257 outputs (bytes 0-255 and EOS; BOS is not an output).

**B12. The byte table and l1 are present but take no part.** `e50_sparse_code.py` L47-49; `exp50_code.py` L230-234 (checked bit-equal to their start).
What it does: these parameters are in the module and in the optimizer, but receive no gradient and do not change; they are used only once, to set the amplitude (F6).
Class: X.
Numbers: 2,064 + 4,608 parameters.

### (C) How the error is formed and how it reaches each weight

**C1. The error is formed at the output as softmax cross-entropy over 257 outputs.** `exp50_code.py` L151-152 (`E.cross_entropy` is `torch.nn.functional.cross_entropy`, `exp11_cell.py` L39).
What it does: the error at output unit o is softmax(logits)_o minus 1 for the target byte; the softmax denominator is a sum over all 257 output units.
Class: X. Each output unit's error depends on all other output units' values through the normalisation, and the target is supplied by the program.
Numbers: 257 classes.

**C2. Only the three answer positions carry error, chosen by the program.** `exp45.py` L166-168.
What it does: the mask is set from the length of the prompt, which the program knows; no error is formed at any prompt byte.
Class: X. The program, not the learner, decides where a target exists.
Numbers: 3 positions per item.

**C3. The error is averaged over all answer positions in the batch.** `exp50_code.py` L152 (default reduction "mean").
What it does: every error is divided by 192 (64 items x 3 positions).
Class: X. A divisor computed over the whole batch.
Numbers: 192.

**C4. Exact rule (`local=False`): one backward pass by automatic differentiation through all maps and all steps.** `e50_sparse_code.py` L52, L57, L65-67 with `local=False`; `exp3_state.py` L107-108 (`linear`); `e42_cells.py` L135 (`linear(h, W * m3, bias)`).
What it does: the error at the outputs is carried back through l3 (transpose of W), the tanh of z2, lm and l2, then through the recurrent map at every earlier step, through the gates, and to the input map; each cell receives a sum over 512 readout units and over the cells that read it, over all later steps.
Class: X. This is the single global backward pass that the principle prohibits.
Numbers: none.

**C5. Sign rule (`local=True`) at l3, lm and l2: the error crosses sign(W) times the mean magnitude of the whole matrix.** `exp3_state.py` L91-104 (SignLinear), used through `lin` at `e50_sparse_code.py` L65-67.
What it does: the error sent back is g @ (sign(W) * mean|W|), with the mean over all entries of the matrix, recomputed at each backward pass; the weight gradient is the usual outer product.
Class: X. The scale is a quantity computed from a whole map and delivered to every unit; the pass is still one global backward pass.
Numbers: one scale per matrix.

**C6. Sign rule at the recurrent map: sign of the masked weights times the mean magnitude inside the mask.** `e42_cells.py` L50-65 (MaskedSign), L111-118 (sign_scale); `e50_sparse_code.py` L54, L57.
What it does: the error at cell i's gates is sent back to i's sources j through sign(W_ij) times one scale computed over all 8,448 kept weights; so a cell receives error from the cells that read it, over the reverse direction of their connections.
Class: X. The scale is a whole-map quantity; and since only 520 of 2,560 connections are reciprocal, most error arrives over a direction in which the receiving cell has no incoming connection in the forward computation.
Numbers: one scale for the map.

**C7. Sign rule at the input map: sign of the masked weights times the mean magnitude of the kept weights.** `e49_interface.py` L56-60; `e50_sparse_code.py` L52.
What it does: the same rule; the error sent to the code is discarded because the table is fixed and needs no gradient.
Class: X (whole-map scale). The weight gradient itself is local (C10).
Numbers: one scale for the map.

**C8. Under both rules the gates' nonlinearities and products are differentiated exactly.** `e50_sparse_code.py` L59-60 (autograd through sigmoid, tanh and products).
What it does: the derivative of h' with respect to the gate pre-activations and to h is exact.
Class: N for the derivative inside one cell (it uses only the cell's own values); X for the fact that it is part of the global pass.
Numbers: none.

**C9. Under both rules credit passes through time over the whole item.** See A11; `exp50_code.py` L151-155.
What it does: the sign rule changes only the matrices crossed backwards, not the backward order in time; it requires every state of the item to be kept until the end of the item.
Class: X.
Numbers: none.

**C10. The weight gradient is the product of the error delivered to a row and the activity of the source, summed over positions and items, and zero outside the masks.** `e42_cells.py` L63; `exp3_state.py` L102.
What it does: dW_ij = sum over items and positions of g_i x_j, multiplied by the mask.
Class: U. The product uses only quantities present at the two ends of the connection, but g_i is delivered by the global pass (C4-C7).
Numbers: none.

**C11. Bias gradients are the sum of the delivered error.** `e42_cells.py` L64; `exp3_state.py` L103.
Class: U (as C10).
Numbers: none.

### (D) How a weight is changed

**D1. The optimizer is AdamW with fixed rate and moment constants.** `exp16_integrated.py` L176-187; `exp26_candidate.py` L39-40, L72-76; `exp20_reference.py` L53; `exp42_cells.py` L161-168.
What it does: each weight keeps a running mean of its gradient and of its squared gradient, and moves by lr times their ratio after bias correction.
Class: U. The two running averages are kept per weight and use only that weight's gradient; the constants and the schedule are the experimenters'.
Numbers: lr = 0.001; betas = (0.9, 0.98); eps = 1e-8 (PyTorch default, not set in the code); amsgrad off.

**D2. Bias correction uses a step counter shared by all weights.** PyTorch AdamW, through D1.
What it does: the correction factors 1 - 0.9^t and 1 - 0.98^t use the number t of updates since the optimizer was made.
Class: X. A clock supplied by the program and the same for every weight.
Numbers: t = update count.

**D3. Decoupled weight decay on every parameter except the byte table.** `exp26_candidate.py` L40, L67-76; `exp16_integrated.py` L181-187.
What it does: before each step, every weight and bias (input map, recurrent map, lm, l2, l3 and their biases) is multiplied by 1 - lr x wd.
Class: X. A constant pull toward zero, fixed by the program and the same for every weight.
Numbers: wd = 0.25; factor 0.99975 per update; 0.938 per 256 updates; 0.0021 over 24,576 updates; the byte table has wd = 0.

**D4. One change per batch of 64 items.** `exp45.py` L30 (BATCH = 64); `exp11_cell.py` L256-259; `exp50_code.py` L148-156.
What it does: the gradients of 64 items, processed in parallel copies of the network with separate states, are averaged into one change.
Class: X. The weights change from 64 items that no single copy of the network has seen in sequence.
Numbers: 64.

**D5. Weights outside the masks are held at zero.** `e42_cells.py` L63, L87; `exp50_code.py` L121-122, L378-383 (checked every 256 updates, with the optimizer's averages there also zero).
What it does: the gradient outside a mask is zero, the initial value is zero, and decay keeps zero at zero.
Class: X. The structure is imposed, not formed.
Numbers: none.

**D6. The amount of learning is fixed.** `exp50_code.py` L56-57, L395-404; `exp42_cells.py` L48-49.
What it does: 49,152 updates for `C46-one` (3,145,728 items); a state is saved at 24,576; scoring every 256 updates, without learning.
Class: X.
Numbers: 49,152; 24,576; 256.

**D7. The same rate and decay for all maps; no schedule; no clipping.** `exp26_candidate.py` L84-91; `exp50_code.py` L148-156.
Class: X.
Numbers: as D1 and D3.

### (E) The items

**E1. Item format.** `exp11_cell.py` L228-232.
What it does: prompt = task code byte, operand a as two decimal digits, operator symbol, k letters, operand b as two digits, `=`; target = result as two digits, then EOS.
Class: X. The format, including zero-padded decimal digits and the place of the operator, is the experimenters'.
Numbers: 2 digits per number; codes `p`, `q`, `v`, `w`; symbols `+ - > <`.

**E2. Four tasks modulo 89.** `exp11_cell.py` L49-50, L224-225.
What it does: add and sub modulo 89; max and min.
Class: X.
Numbers: 89; four tasks.

**E3. Half of the ordered pairs of each task are taught.** `exp11_cell.py` L235-241; `exp50_code.py` L346.
What it does: the 7,921 pairs (a, b) are shuffled with `Random(seed * 1000 + 50 + 89 * 7 + code[0])`; the first half is taught, the rest is held for scoring.
Class: X.
Numbers: frac = 0.5; 3,960 taught pairs per task; 15,840 in all.

**E4. Items are drawn in shuffled passes over all taught pairs of all four tasks, with a random number of letters.** `exp11_cell.py` L244-253.
What it does: one generator `Random(seed + 77)` shuffles the 15,840 task-pairs for each pass, draws k uniformly in 0..12 and each letter uniformly among 26; the four tasks are mixed in every batch.
Class: X.
Numbers: kmax = 12 (`exp11_cell.py` L54, `KMAX["M"]`); 26 letters.

**E5. During learning the target is given at the answer positions and the answer is given as input.** See A9 and C2.
Class: X.

**E6. Scoring panels.** `exp42_cells.py` L212-231; `exp50_code.py` L270-290, L345.
What it does: A: 300 held pairs per task with 12 letters; taught panel: 300 taught pairs per task with 12 letters; R: 1,000 held pairs each of add and sub that need the reduction (a + b >= 89, a - b < 0), with no letters; length test: 300 held add pairs at 12 and at 40 letters. The reported A, R and taught figures count add and sub only.
Class: X.
Numbers: 300, 1,000, 12, 40; generators `Random(seed + 3)` for A and the length test, `Random(seed + 4)` for the taught panel (`panel` called with seed + 1), `Random("e41|seed|50|op")` for R.

**E7. An answer is right only if all three generated bytes equal the target.** `exp11_cell.py` L217-220; `exp50_code.py` L274-290.
Class: X.
Numbers: 3 bytes; at most 4 generated.

**E8. What begins and ends an item for the network.** A1, A2, A13, A14.
What it does: an item begins with BOS and a zero state, and ends after EOS; the next item is in another row of the batch or in another batch, with its own zero state.
Class: X.

**E9. Letters carry no error and are not predictable.** `exp45.py` L166-168 (no target at prompt positions); `exp11_cell.py` L230.
Class: X.
Numbers: none.

### (F) The start of the network, and knowledge of the task used to build it

**F1. Initial values from one seed in a fixed order of construction.** `arch20.py` L36-50, L62-69; `arch26.py` L45-47; `exp50_code.py` L104.
What it does: `T.manual_seed(73129)`, then book, l1, GRU, lm, l2, l3 are created with PyTorch defaults: Linear weights and biases uniform in plus or minus 1/sqrt(fan_in); GRU weights and biases uniform in plus or minus 1/sqrt(256) = 0.0625; the byte table N(0, 1) with the BOS row zero.
Class: X.
Numbers: seed 73129; bounds 0.0625 (GRU and lm), 0.0442 (l2 and l3).

**F2. Recurrent weights inside the mask are scaled by sqrt(256 / 11); those outside are set to zero.** `e42_cells.py` L84-87.
Class: X.
Numbers: 4.824.

**F3. Input weights inside the mask are scaled by sqrt(512 / 46); those outside are set to zero.** `exp50_code.py` L113-122.
Class: X.
Numbers: 3.336.

**F4. lm and the biases are not scaled.** `e49_interface.py` L43-47 with both masks full (`exp50_code.py` L114).
Class: X.
Numbers: none.

**F5. The code assignment is drawn by a seeded shuffle.** `e50_sparse_code.py` L28-35; `exp50_code.py` L74-78.
What it does: 512 values shuffled by `Random("e50|code|seed")`; byte number k in the order of USED takes values 11k to 11k + 10 of the shuffle; codes of different bytes share no value, so all used bytes are equally dissimilar (for example, digits carry no order).
Class: X.
Numbers: 11; 512.

**F6. The amplitude of a turned-on value is computed from the initial byte table and l1 of the dense network.** `e50_sparse_code.py` L38-43; `exp50_code.py` L105-109, L62.
What it does: amplitude = sqrt(mean over used bytes of |tanh(l1 book(b))|^2 / 11).
Class: X. A number taken from an unused part of a different network.
Numbers: 3.2387 for seed 73129 (the program's own constant).

**F7. The set of 46 used bytes is taken from the alphabet of the first task family.** `e50_sparse_code.py` L24.
What it does: digits, `+ - > < =`, 26 uppercase letters, the four codes `p q v w`, and BOS; every other byte, including EOS and every byte from 128 to 255, has a zero code and gives no input to any cell.
Class: X. Knowledge of the task's alphabet is built into the input.
Numbers: 46.

**F8. The set of 20 bytes "the task must tell apart" is used to stop a run.** `e50_sparse_code.py` L25; `exp50_code.py` L185-193, L334-337.
What it does: the run is stopped before learning if a needed byte reaches no cell, a cell has no input, or a cell is not read.
Class: X. Knowledge of which bytes matter for the task is used to accept a network.
Numbers: 20.

**F9. The C46-one draw ensures that every cell has input for every used byte.** `exp50_code.py` L81-85.
What it does: for cells 0..255 and the 46 bytes in order, one of the byte's 11 values is drawn by `Random("e50|one|seed")`.
Class: X. The connection pattern is designed from the program's knowledge of the code.
Numbers: 46 per cell, 1 per byte.

**F10. Connection draws and geometry.** See B4; `e42_cells.py` L35-46.
Class: X.
Numbers: as B4; generators `Random("e42|periods|seed")`, `Random("e42|phases|seed")`, `Random("e42|connections|seed")`.

**F11. Diagnostics that use task knowledge but do not affect learning.** `exp50_code.py` L196-227 (digit groups, input size), L160-177 (counts).
Class: X (no effect on learning).
Numbers: groups at 0-2, 3-4, 5 or more of ten digits.

### (G) A learner's life in sw_run.py

**G1. The learner in sw_run.py is the dense GRU, not E50's network.** `sw_run.py` L433-436; `sw_net.py` (SWNet, `step` uses `arch26.forward`).
What it does: a learned byte table (258 x 8), l1, a dense GRU (every unit reads every unit and the whole input), dense lm, l2, l3; widths H = U = 64 in the small world, H = 256 and U = 128 in the long-life world.
Class: X. None of the limits of section B apply in the lives.
Numbers: H, U as stated; 52,625 parameters at 64 (sw_net head).

**G2. The life starts from a saved birth, with a new optimizer.** `sw_run.py` L241-258, L433-437.
What it does: in the long-life world the weights of a birth made elsewhere (E41) are loaded; Adam's running averages and step counter start from zero at the start of the life.
Class: X.
Numbers: none.

**G3. One optimizer for the whole life; nothing is reset between visits.** `sw_run.py` L437, L718-734.
What it does: weights, Adam's averages and counter, and the store of old items are kept from visit to visit; only the network state is reset, at every item (A1).
Class: X (the reset of state); N for weights kept.
Numbers: none.

**G4. Each update takes 64 items from the stream: 48 taught, then 16 asked.** `sw_run.py` L6-12, L659-673; `lw_world.py` L18, L84-96.
What it does: the 16 asked items are answered first, under no gradient, greedily, 3 bytes; the answer is right if all three bytes equal the target; nothing is learned from them.
Class: X.
Numbers: 64, 48, 16, 3.

**G5. The store of old items is held by the program.** `sw_run.py` L549-552, L684-696.
What it does: each new taught item (prompt and target, as presented) is offered to a store; while it is not full it is appended; after that it replaces slot j = randrange(n) if j < capacity (reservoir sampling); capacity = (4 x P) // 23 items in the long-life world, (4 x P) // 15 in the small world, P the parameter count including the byte table.
Class: X. Memory formed by an act of the program, outside the network, holding exact copies.
Numbers: 4, 15, 23; P.

**G6. Replay replaces part of the new items.** `sw_run.py` L460, L676-683.
What it does: once the store holds at least k = round(rho x 48) items, the batch is the first 48 - k new taught items plus k items drawn uniformly with replacement from the store by `Random("world|seed|replay")`; the last k new items are not learned at that update (they are still offered to the store).
Class: X. The program chooses what is rehearsed, how much, and when.
Numbers: rho (for example 0.5, `sw_run.py` L55-56); 48.

**G7. Visits and the schedule of families.** `lw_world.py` L19-22, L53-74, L84-96; `sw_run.py` L465-466, L840-841.
What it does: a visit is 256 updates on one family (the four tasks of one modulus, drawn uniformly per item); a new family comes with probability 0.05 (0.2 in the small world); a return draws an absence class (1-4, 5-32, 33 or more visits) uniformly; the last 4 families are never scheduled.
Class: X. The program decides when a family is presented and for how long, and presents one family at a time.
Numbers: 256, 0.05, 0.2, 4, 32, 4 probe families (20 probe groups in the small world).

**G8. The long-life world's items.** `lw_world.py` L9-21, L33-51, L76-79.
What it does: 56 families, moduli 41..100 except 73, 83, 89, 97; each task has a two-byte code drawn from bytes 128..255; 10 % of the p x p pairs are taught; 0..12 letters.
Class: X.
Numbers: 41, 100, 0.10, 12; code bytes 128-255.

**G9. Constant rate and decay over the whole life.** `sw_run.py` L97-98, L981.
What it does: lr = 0.001, wd = 0.25 (0 in some L2 runs) for every update of the life; decay acts on the weights of every earlier family as on the current one.
Class: X.
Numbers: 0.001, 0.25; factor 0.938 per visit; 0.129 after 32 visits without relearning.

**G10. The program checks that every learned item is a taught pair with its true answer.** `sw_run.py` L699-703.
Class: X (a guard; no effect on learning).

**G11. The list and "decay alone" are measurements by the program.** `sw_run.py` L277-287, L305-310, L503-543.
What it does: at the end of each visit, the first 100 reduction pairs of add and sub of each family taken, and 2,000 items of the birth family, are answered without learning; copies of the weights multiplied by (1 - lr x wd)^u answer the same items.
Class: X (no effect on learning).
Numbers: 100, 1,000 per task.

**G12. A birth inside the world ends when the program judges that enough tasks are learned.** `sw_run.py` L428-430, L615-650.
What it does: joint learning on the birth groups; every 64 updates, tasks whose share right over the asked items is at least 0.8 are counted; the birth ends when at least `need` tasks qualify, or after 32,768 updates.
Class: X. A change of phase decided by the program from scores.
Numbers: 64, 0.8, 32,768, need = 2 x groups - 1.

**G13. The learning rule may be switched by the program between birth and life.** `sw_run.py` L94-95, L603-613.
What it does: `sweep-after-naive` continues a life learned by the exact rule under the sign rule.
Class: X.

**G14. Branches replace the schedule.** `sw_run.py` L867-901.
Class: X (an experimental act; it continues learning on groups named by the experimenter).

**G15. The E50 input code would give no input for the long-life world's task codes.** Combination of F7 and G8.
What it does: bytes 128..255 and the small world's lowercase codes other than `p q v w` have zero rows in E50's table; a network with that code placed in these worlds could not distinguish the tasks.
Class: X.

### Count of entries by group

A 15; B 12; C 11; D 7; E 9; F 11; G 15; total 80.

---

## 2. The entries that depart furthest from the principle or most constrain the network

1. **One global backward pass under both rules (C4, C5, C6, C9, A11).** The sign rule replaces the transposed weights with signs, but keeps the backward pass over all layers and over all steps of an item, and multiplies the signs by a scale computed over a whole matrix. Expected consequence: learning of the cells depends on errors that no cell could receive by relay within the time available; the results under the sign rule do not show what a learner restricted to local information can do. Test: rerun with the state detached at every step (the dense network's `through_time=False` already exists in `arch26.py` L60) and with the error sent back only one crossing per step forward in time; compare accuracy on R and on the length test.

2. **The dense readout and the direct path from input to output (B8, B9, B10, B11).** 92.3 % of the weights in use lie outside the cells; each readout unit reads all 256 cells and the code of the current byte. Expected consequence: the answer can be computed largely by a dense two-layer network that reads the whole state at once, so the limits on cell-to-cell connections constrain little; local learning at the output is not satisfied. Test: run with `silence=True` (already in the forward, `e50_sparse_code.py` L63-64) to measure what l2 alone gives; then mask lm to a few cells per readout unit (E49's K_OUT) and remove l2, and measure accuracy.

3. **The item format leaves one crossing before the answer (A3, A6, A7, A8).** The relay graph has diameter 5 and mean distance 2.84 (B4), but only one crossing separates the last digit of b from the first answer byte, and one crossing separates the answer bytes. This constrains entry 2: within cells, b can be combined with a only in the cells that read b directly (in C46-one every cell does, by F9) plus one neighbour, and everything else must be done by the dense readout. It also constrains entry 1, since the exact backward pass carries credit along paths that the forward pass never uses before the answer is due. Test: add m steps with a neutral byte (or several crossings per byte) between `=` and the answer, for m = 0, 1, 2, 4, 8, with the readout restricted as in entry 2; measure whether accuracy rises up to m near the diameter.

4. **A fixed input code made with knowledge of the task alphabet (B1, B2, F5-F9, G15).** The program chooses which 46 bytes have a code, gives them disjoint codes of equal amplitude, ensures that each cell sees every used byte, and stops a run if a byte it considers needed reaches no cell. Expected consequence: the network cannot form its own code, cannot represent similarity (for example, the order of digits), and receives no input for any byte outside the first family, including the task codes of the long-life world. Test: present the long-life world's items to E50's forward and count rows with zero input; compare with a code drawn without reference to USED (all 258 bytes), and with a code the learner adapts by a local rule.

5. **The program forms and places the error (C1, C2, C3, A9).** The target is given only at positions the program knows to be answers, through a softmax over all 257 outputs, averaged over the batch, with the true answer fed as input. Expected consequence: the learner never has to find when an answer is due or predict anything else; the output units are coupled through the normalisation. Test: train with loss at every position (prediction of every next byte) and with an independent error per output unit (no softmax), and compare learning speed and accuracy.

6. **The state is reset at each item and items are learned in parallel batches (A1, A12, D4, E8).** Nothing can be held in the state across items, weights never change within an item, and each change averages 64 items processed by separate copies of the network. Expected consequence: memory can only be in weights changed by the global pass, never in activity, and the learner never experiences a continuous stream. Test: run one continuous stream with batch size 1 and no reset, with a change after each item, and compare.

7. **Uniform weight decay (D3, G9).** Every weight is multiplied by 0.99975 at every update: 0.938 per visit of 256 updates, 0.129 after 32 visits. Expected consequence: knowledge of a family not revisited within a few tens of visits fades regardless of its use, so long-term memory depends on rehearsal or return visits that the program schedules. Test: the list's "decay alone" rows already estimate this; compare lives at wd = 0.25 and wd = 0 on families with long absences.

8. **The store of old items held and replayed by the program (G5, G6).** The program keeps exact copies of items, chooses them uniformly, and sets the replay share and the capacity. Expected consequence: retention in replay arms measures the program's store, not memory formed by the learner; the principle allows some rehearsal, but not memory formed by an act of the program. Test: replace stored items with items generated by the network itself (its own answers to prompts it composes) and compare retention.

9. **The schedule of experience (G4, G7, G12, G13).** The program presents one family at a time for 256 updates, answers asked items without learning, ends a birth by a score criterion, and may switch the rule. Expected consequence: interference between families is largely decided by the schedule; results cannot separate what the learner does from what the schedule does. Test: interleave families within visits at the same total dose and compare.

10. **The optimizer's fixed constants and shared counter (D1, D2).** Each weight keeps two running averages of its own gradient, which is local, but the rate, the betas and the bias correction counter are global and fixed, and Adam's state is carried through the whole life. Expected consequence: the size of each change is normalised by a history that spans families; this may protect or erase old knowledge in ways not attributable to the network. Test: use plain gradient steps with a per-weight rate, or reset the averages at each visit, and compare retention.

---

## 3. What could not be determined from the code

- **E62.** No runner of E62 was among the files given; I cannot say how E62 differs from E50 in its settings, dose or data.
- **E41's birth** (the network that sw_run.py loads in the long-life world): `exp41_scale.py` was not among the files given, so its data, dose and rule are known only from the fields that `sw_run.py` reads from its receipt.
- **The small-world generator in full** (`sw_world.py`) was read only for its head (constants, task construction and schedule docstring); the scoring module `sw_score.py` was not read.
- **Whether E50's network was ever run inside sw_run.py.** The code given runs only the dense SWNet there (G1).
- **The PyTorch version used.** I assume that AdamW skips a parameter whose gradient is None (so the byte table and l1 are not decayed); the program's own check that they stay bit-equal (`exp50_code.py` L230-234) supports this. I also assume eps = 1e-8, which the code does not set.
- **The amplitude 3.2387** was not recomputed; it is the program's own constant, checked by the program against its computation.
- **The effective values of the runs** (losses, accuracies, cosines between the two rules) are not determined by the code and were not examined.
- **The meaning of "a few passes cover the network"** in terms of crossings per byte is not defined by the code; my counts of the graph (diameter 5, all cells reached in 4 to 5 crossings) are for seed 73129 only.
