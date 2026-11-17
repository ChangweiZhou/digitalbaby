import sys, runpy
sys.argv = ["x", "none"]
g = runpy.run_path(__file__.replace("e54_r2_figs.py", "check_e54_brief.py"))
split, digest, DEV, Random = g["split"], g["digest"], g["DEV"], g["Random"]
for f in (0.25, 0.15, 0.10):
    print("stream digest at row 4,096, f", f, digest(f, 73129, 4096)[:16])
for f in (0.50, 0.25, 0.15, 0.10):
    for t in DEV[:2]:
        taught, held = split(t, f, 73129)
        op = t[0]
        red = [(a, b) for a, b in held if (a + b >= 89 if op == "add" else a - b < 0)]
        print(f"f {f} {op}: taught {len(taught)}, held {len(held)}, held pairs that need the reduction {len(red)}")
# overlap of taught sets between fractions (addition)
T = {f: set(split(DEV[0], f, 73129)[0]) for f in (0.5, 0.25, 0.15, 0.10)}
for a in T:
    for b in T:
        if a > b: print(f"addition, seed 73129: of the {len(T[b])} pairs taught at {b}, {len(T[a] & T[b])} are taught at {a}")
