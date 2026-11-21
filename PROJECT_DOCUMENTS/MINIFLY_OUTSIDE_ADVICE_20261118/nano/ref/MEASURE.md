# The measure of the test at the level of nanoGPT

Defined once, for every line under `nano/`. The program is `score.py` (plain numpy). Development level; no checker gate, seal or review is claimed.

## The text

`nano/data/tinyshakespeare.txt`, 1,115,394 bytes, 65 byte values. A byte value is given its index, 0 to 64, in the ascending order of the 65 values (index 0 is the newline; this is the table nanoGPT's own preparation makes). **The stream** is bytes 0 to 1,003,853 (the first nine tenths, `int(0.9 n)` as in nanoGPT). **The last tenth** is bytes 1,003,854 to 1,115,393: 111,540 bytes, never learned from and used for no choice.

## What a model must give

For every byte of the last tenth, before that byte arrives, a probability for each of the 65 byte values: non-negative, summing to one. The model may use any of the bytes before it in the text and nothing after. **The first bytes of the last tenth are predicted from the end of the stream**: the context of a byte is the bytes before it in the whole text, so all 111,540 bytes are scored and none is skipped for lack of a context. A model with a window of W bytes uses the W bytes before the byte; a model that carries a state reads the text in order and carries it.

## The two figures

Let y_1 ... y_n be the bytes of the last tenth (n = 111,540) and p_i the probabilities given for byte i.

1. **The stream loss**: L = -(1/n) Σ_i ln p_i(y_i), in nats a byte; in bits a byte it is L / ln 2. Nothing is cut off below and nothing is floored: a probability of zero for the byte that arrives makes the loss infinite. A model whose branches do not sum to one has to say how it makes them do so; that is part of the model, not of the measure.
2. **The share named right**: the share of i for which the largest of p_i is at y_i, in percent. A tie goes to the lowest index.

Two fixed points: equal probabilities give ln 65 = 4.1744 nats (6.0224 bits) ; the byte frequencies of the stream, used at every position, give 3.3473 nats (4.8292 bits) and name 14.90% right (the space).

## The program

```python
import score
x, cut, S = score.load()                 # the whole text as indices; cut = 1003854; S = 65
score.score(P)                           # P of shape (111540, 65), row i for byte cut + i  -> {n, nats, bits, right}
score.score_fn(f, window=W)              # f(ctx) -> (m, 65); ctx of shape (m, W), oldest byte first: exactly the W bytes before each byte
score.contexts(W, lo, hi)                # the windows and the bytes themselves, for any stretch of the text
```

`score.score` refuses rows that are negative or do not sum to one within 1e-4.

## How this differs from nanoGPT's own figure

nanoGPT's "val loss" is an estimate: the mean loss over 200 batches of blocks drawn at random from the last tenth, every position of each block counted, so that the first byte of a block is predicted from one byte, the second from two, and so on up to the block's length. It is therefore taken on a random sample, and on contexts that are on average half a window long. The stream loss scores every byte once, each from a full window. For a GPT both are reported (`gpt.py`): the stream loss is the measure of this test; nanoGPT's estimate is given only so that a figure here can be set beside nanoGPT's published ones.

For a GPT with a long window the exact stream loss costs one pass of the network for each byte. `gpt.py` therefore also gives a cheaper reading, windows moved on by half a block, in which each byte is predicted from between half a window plus one and a whole window of bytes; where both were computed, both are reported and the difference is stated.

## The tables of the last k bytes (`tables.py`)

Counted on the stream only. c(u, w): the number of times byte w follows the run u of k bytes in the stream.

* **Unsmoothed, for the share named right**: the longest run of at most k bytes before the byte that was seen in the stream decides, by its most frequent next byte. It has no loss (it gives zero to bytes it has not seen).
* **Interpolated Kneser-Ney, for the loss**, in Chen and Goodman's form with three discounts for each order. For the table of the last k bytes, with u the last j bytes and u' the last j - 1:

  P_j(w | u) = (C_j(u, w) - D_j(C_j(u, w))) / C_j(u) + γ_j(u) P_{j-1}(w | u'),  γ_j(u) = Σ_w D_j(C_j(u, w)) / C_j(u),

  where C_j(u) = Σ_w C_j(u, w); if C_j(u) = 0 then P_j(w | u) = P_{j-1}(w | u'); and P_{-1} = 1/65. At the top order j = k, C_k is the count c itself. At the orders below, C_j(u, w) is the number of different bytes seen in front of the run u w in the stream. D_j(0) = 0, D_j(1) = D1, D_j(2) = D2, D_j(c ≥ 3) = D3, with, from the numbers n_1 ... n_4 of runs counted once, twice, three and four times at that order, Y = n_1 / (n_1 + 2 n_2), D1 = 1 - 2Y n_2/n_1, D2 = 2 - 3Y n_3/n_2, D3 = 3 - 4Y n_4/n_3. No number is fitted to anything but the stream's counts.
* **Interpolated Witten-Bell**, as a second smoothing: P_j(w | u) = (c(u, w) + N(u) P_{j-1}(w | u')) / (c(u) + N(u)), N(u) the number of different bytes seen after u; P_j = P_{j-1} where c(u) = 0; P_{-1} = 1/65.
