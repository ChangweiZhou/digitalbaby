# lt3o check (24–25 Nov 2026)

> **NON-COMPLIANT REFERENCE ONLY (25 Nov 2026).** `lt`, `lt2`, `lt3` and `lt3o` break the owner's requirements: global readout (`rho_o` up to 0.5), fixed connection counts, a fixed firing share, and no feedback cells or error relay into the tissue. None of them is a base for new work.

`lt3o.c` is `lt3.c` with faster code and the same mechanism. Its results are not bit-identical to lt3, because random draws and visiting order differ. The file header lists every change and one small deviation: a source that gains room partway through a byte is offered from the next byte.

To run the scripts, build the two libraries in this folder first:
- `old.so` from `../lt3.c`
- `new.so` from `../lt3o.c`

Both are built with `cc -O3 -march=native -shared -fPIC`.

## Candidates

The candidate set (`cands.py`) is:
- the unfinished round-2 candidate p2_102 (15,360 cells), the costliest known corner;
- 12 candidates drawn at random (seed 0) from rounds 2 and 3.

## Speed (`bench.py`, `speed.txt`, `scale.txt`)

Measured in µs per byte on Moby after 2,000 warm-up bytes, one process at a time.

| Candidate | Old | New | Speed-up |
|---|---|---|---|
| p2_102 at 4× | 99,652 | 5,017 | 19.9× |
| The other 12 at 1× | — | — | 0.9–1.9× |

- The out-list consistency check passed on all 13.
- In lt3o, the exponent of per-byte time in network size is 0.64–1.17, so the total is linear in cells × bytes.

## Same scores (`eq.py`, `eq.jsonl`, `eqana.py`)

The full battery (life → cl, Moby ladder → slope) was run on 6 candidates, with old and new code and seeds 1 and 2: 24 runs in all.

- On every metric, new − old is within the old code's seed-to-seed gap.
- Two Moby 4× differences sit at the edge of that gap: +0.004 against 0.002, and −0.005 against 0.003. Their signs are mixed.
- m3_050 and m3_144 match exactly, because they grow no synapses.

## Profile (`prof.py`)

| Code | Where the time goes |
|---|---|
| Old, on p2_102 | about 94% in growth (scan of all active sources) |
| New, on p2_102 | mainly `spread` and `pop_step` (the dense per-byte leak and local competition), then growth at about 30% |
