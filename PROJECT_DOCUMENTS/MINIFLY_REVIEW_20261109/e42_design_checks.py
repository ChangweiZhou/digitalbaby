"""Arithmetic behind section 22 of MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md (the cell layer, proposed as E42).

Read-only and standard library only: nothing is trained and no file of the repository is opened.
Run from anywhere:  python3 -I e42_design_checks.py

  A  where the bytes of an item stand, as exp45.batch_tensor lays them out (start symbol at position 0)
  B  the neighbourhood of a cell on a 16 x 16 grid with wrap-around
  C  parameter counts of the present network and of each configuration of the cell layer
  D  for a cell of period p and phase f: in which training items it can carry both digits of the first operand
     from one period to the `=` sign without an update in between
  E  the factor by which the kept recurrent weights are scaled at the start
"""
from itertools import product

U, H = 256, 512                      # cells; feedforward width (arch20.GRU_UNITS, exp11_cell.H)
SIDE, RADIUS = 16, 3
KMAX = 12                            # letters in training: 0..12, uniform (exp11_cell.stream: rng.randint(0, kmax))


def layout(k):
    """Positions in x for an item with k letters: x = [BOS, code, a1, a2, op, k letters, b1, b2, '=', c1, c2]."""
    names = ["BOS", "code", "a1", "a2", "op"] + [f"d{i}" for i in range(k)] + ["b1", "b2", "=", "c1", "c2"]
    return {n: i for i, n in enumerate(names)}


def part_a():
    print("A. positions (start symbol at 0)")
    for k in (0, 12, 40):
        p = layout(k)
        print(f"   k={k:2d}: a1 {p['a1']}, a2 {p['a2']}, '=' {p['=']} (target c1), last target at {p['c2']} (EOS); "
              f"a1 is {p['='] - p['a1']} steps before '='; length of x {len(p)}")
    assert layout(0)["="] == 7 and layout(12)["="] == 19 and layout(40)["="] == 47


def neighbours(i, radius=RADIUS, side=SIDE):
    r, c = divmod(i, side)
    out = []
    for dr, dc in product(range(-radius, radius + 1), repeat=2):
        if dr or dc:
            out.append(((r + dr) % side) * side + (c + dc) % side)
    return sorted(set(out))


def part_b():
    sizes = {len(neighbours(i)) for i in range(U)}
    sym = all(i in neighbours(j) for i in range(U) for j in neighbours(i))
    print(f"B. grid {SIDE} x {SIDE} with wrap-around, radius {RADIUS} (largest of the two coordinate differences): "
          f"neighbourhood sizes {sorted(sizes)}, symmetric {sym}")
    for rad in (2, 3, 4):
        print(f"   radius {rad}: {len(neighbours(0, rad))} cells")
    assert sizes == {48} and sym


def params_present():
    book = 258 * 8
    l1 = 8 * H + H
    w_ih, w_hh, b = 3 * U * H, 3 * U * U, 2 * 3 * U
    lm = U * H + H
    l2 = H * H + H
    l3 = H * 257 + 257
    return dict(book=book, l1=l1, w_ih=w_ih, w_hh=w_hh, gru_bias=b, lm=lm, l2=l2, l3=l3)


PERIOD_SETS = {
    "{1}": {1: 256},
    "{1,2,4}": {1: 128, 2: 64, 4: 64},
    "{1,2,4,8}": {1: 128, 2: 64, 4: 32, 8: 32},
    "{1,2,4,8,16}": {1: 128, 2: 64, 4: 32, 8: 16, 16: 16},
}


def part_c():
    p = params_present()
    total = sum(p.values())
    print(f"C. present network: {total:,} parameters  {p}")
    assert total == 1124113
    print("   cell layer: kept recurrent weights = 3 * 256 * (N + 1), the cell's own previous value included")
    for n in (5, 10, 20, 255):
        kept = 3 * U * (n + 1)
        print(f"   N={n:3d}: kept recurrent {kept:7,} of {p['w_hh']:,}; whole network {total - p['w_hh'] + kept:,} "
              f"({(total - p['w_hh'] + kept) / total:.3f} of the present count)")
    for name, counts in PERIOD_SETS.items():
        assert sum(counts.values()) == U
        extra = sum(c * q for q, c in counts.items() if q > 1)
        print(f"   periods {name:13s} cells by period {counts}; weights over the steps of a period (learned variant): {extra}")


def carries(p, f, k):
    """True if a cell of period p and phase f has one update whose period contains positions 2 and 3 (a1, a2)
    and has no further update up to and including the '=' position 7 + k."""
    eq = 7 + k
    u = f
    while u <= eq:
        if u >= 3 and u - p < 2 and u + p > eq:      # period (u - p, u] holds a1 and a2; next update after '='
            return True
        u += p
    return False


def part_d():
    print("D. carrying both digits of the first operand to '=' within one period, with no update in between")
    print("   period  phases that can, by number of letters k = 0..12 (count of phases out of p)        share of (phase, k)   gaps with no phase")
    for p in (2, 4, 8, 16):
        row, tot = [], 0
        for k in range(KMAX + 1):
            n = sum(carries(p, f, k) for f in range(p))
            row.append(n)
            tot += n
        none = [k for k, n in enumerate(row) if n == 0]
        print(f"   {p:6d}  {row}   {tot / (p * (KMAX + 1)):.2f}   {none}")
    for p in (8, 16):
        n40 = sum(carries(p, f, 40) for f in range(p))
        print(f"   at 40 letters, period {p}: {n40} of {p} phases")
    # the statements used in the memo
    assert all(not carries(4, f, k) for f in range(4) for k in range(2, 13))
    assert all(any(carries(8, f, k) for f in range(8)) for k in range(0, 10)) and not any(carries(8, f, k) for f in range(8) for k in (10, 11, 12))
    every = [f for f in range(16) if all(carries(16, f, k) for k in range(13))]
    print(f"   period 16: phases that can for every k = 0..12: {every}")
    assert every == [4, 5, 6, 7]
    assert all(any(carries(16, f, k) for f in range(16)) for k in range(13))
    assert not any(carries(16, f, 40) for f in range(16))


def part_e():
    print("E. scale of the kept recurrent weights at the start, sqrt(256 / (N + 1)); nn.GRU draws every weight uniformly in +-1/16")
    for n in (5, 10, 20, 255):
        print(f"   N={n:3d}: factor {(U / (n + 1)) ** 0.5:.3f}")
    assert (U / (255 + 1)) ** 0.5 == 1.0


if __name__ == "__main__":
    part_a(); part_b(); part_c(); part_d(); part_e()
    print("all assertions hold")
