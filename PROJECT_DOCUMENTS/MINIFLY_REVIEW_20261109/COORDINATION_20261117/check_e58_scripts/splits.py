import sys
sys.path.insert(0, sys.argv[1])
import npnet as N
from random import Random
SEEDS = (73129, 93683, 116089)
print("round(f*100):", {f: round(f * 100) for f in (0.025, 0.03, 0.05, 0.10, 0.15, 0.20)}, " 0.05*100 =", repr(0.05 * 100), " round(0.05*6889) =", round(0.05 * 6889), round(0.10 * 6889), round(0.20 * 6889), round(0.025 * 6889))
for seed in SEEDS:
    T = {f: {t[0]: set(N.split(t, f, seed)[0]) for t in N.F4} for f in (0.05, 0.10, 0.20)}
    print(seed, "taught pairs per operation:", {f: sorted({len(v) for v in T[f].values()}) for f in T})
    print("   shuffle seeds:", {f: [seed * 1000 + round(f * 100) + 83 * 7 + t[2][0] for t in N.F4] for f in T})
    for op in ("add", "sub"):
        a, b, c = T[0.05][op], T[0.10][op], T[0.20][op]
        print(f"   {op}: taught at 0.05 also taught at 0.10: {len(a & b)} of {len(a)} (a nested split would give {len(a)}; chance gives {len(a)*len(b)/6889:.0f});",
              f"taught at 0.10 also taught at 0.20: {len(b & c)} of {len(b)} (chance {len(b)*len(c)/6889:.0f})")
    for t in N.F4[:2]:
        red = {f: {(a, b) for a, b in N.split(t, f, seed)[1] if N.needs_reduction(t[0], a, b)} for f in (0.05, 0.10, 0.20)}
        allr = sum(1 for a in range(83) for b in range(83) if N.needs_reduction(t[0], a, b))
        common = red[0.05] & red[0.10] & red[0.20]
        print(f"   {t[0]}: untaught pairs needing the reduction: {len(red[0.05])} at 0.05, {len(red[0.10])} at 0.10, {len(red[0.20])} at 0.20 (of {allr}); untaught at all three: {len(common)};",
              f"0.05 and 0.10: {len(red[0.05] & red[0.10])}; 0.10 and 0.20: {len(red[0.10] & red[0.20])}")
print("items presented in 8,192 updates:", 8192 * 64, "; presentations of each taught pair at 8,192 updates:", {f: round(8192 * 64 / (4 * round(f * 6889)), 1) for f in (0.05, 0.10, 0.20)},
      "; at 1,024:", {f: round(1024 * 64 / (4 * round(f * 6889)), 1) for f in (0.05, 0.10, 0.20)}, "; at 64:", {f: round(64 * 64 / (4 * round(f * 6889)), 2) for f in (0.05, 0.10, 0.20)})
print("taught entries of the stream (4 operations):", {f: 4 * round(f * 6889) for f in (0.05, 0.10, 0.20)}, "; items in 64 updates:", 64 * 64)
# the taught panel samples 300 of the taught pairs
for f in (0.025, 0.04, 0.05):
    try:
        Random(1).sample(list(range(round(f * 6889))), 300); print(f, "taught panel of 300: possible")
    except ValueError as e:
        print(f, "taught panel of 300:", e)
