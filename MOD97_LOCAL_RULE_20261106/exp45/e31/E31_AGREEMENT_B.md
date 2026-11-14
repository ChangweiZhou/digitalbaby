# E31 stage 1, agreement check (worker B, 14 November 2026)

**Hashes compared.** Worker B's dry receipts (`E31_DRY_B.json`, commit c8c7fb56 on `claude/charming-clarke-u9g850`) against worker A's (`claude/sleepy-knuth-q96dd1`, commit bc2ddfa9). Seed 73129, `E11_STAGE_UPDATES=256`, final `tensors_sha256`. I read worker A's branch only after both sets were pushed.

| Arm | Worker A | Worker B | Agree |
|---|---|---|---|
| no switch (A: N; B: none) | 60f36ab0e9711512 | 60f36ab0e9711512 | yes. Both equal `exp26_candidate.py life S 73129` at this dose |
| G | 12ef12e3bcebaf76 | 6ffc47938056b756 | **no** |
| H | f58873993a90cad9 | f58873993a90cad9 | yes |
| K | 7a0f74a1c47d2aae | 7a0f74a1c47d2aae | yes |
| P | 85174cc48d1e787c, λ = 10.91974 | 7a8ab648939787cc, λ = 10.88948 | **no** |
| P at λ = 0 (A: P0) | 60f36ab0e9711512 | 60f36ab0e9711512 | yes. Equals no switch |

**As the brief requires, I stopped and changed nothing.** The causes below are from reading worker A's code after both sets were pushed; neither worker's code is altered.

### G: the same rule, different rounding

Both write `gx = g[:, :512] @ W[:512] + g[:, 512:] @ S[512:]`, with S = sign(W) · mean|W| over the whole matrix.
- Worker A evaluates it as two matrix products, added (`arch31.SwitchLinear.backward`).
- Worker B evaluates it as one product with the stacked matrix `cat(W[:512], S[512:])` (`arch31.gate_exact_matrix`, `BackLinear`).

In float32 the two differ by rounding alone. On random errors of the life's shapes they differ by at most about 7–8e-7 relative per crossing; this was tested in scratch, not committed. Both pass the brief's 1e-6 check (B: 6.0e-7 in `E31_CHECK_B.json`), and the difference compounds over 1,536 updates into a different hash.

### P: different positions in the term

- **Worker A:** averages `(z − 0.5)²` over the positions after the first byte whose input byte is not BOS, which excludes padding (`x[:, 1:] != X.BOS`).
- **Worker B:** averages over every position 1 to L−1 of the padded batch tensor, padding included.

The brief says "over items, positions after the first byte, and units" and does not mention padding. The two readings give λ 10.91974 against 10.88948 on the same first batch, and different lives.

The coordinator must settle both before stage 2:
- **For G:** which expression to evaluate. Equivalently, whether a rounding-level difference between independent implementations counts as disagreement.
- **For P:** whether padding positions are in the term.

### Also to note: the source of `exp16_integrated.py`

The current `exp16_integrated.py` (c9fcedcf…, changed by deea398c on 12 November) fails the E26 birth's source admission (`exp26_candidate.admit_e26`).
- Worker B ran every check and dry life in a work folder with today's sources, except `exp16_integrated.py`, which was the 12 November version before deea398c (f4559971…). That is the version the births and the floor and ceiling lives were made with.
- Worker A's no-switch hash equals B's, so the two set-ups agree on the floor's update at this dose.
