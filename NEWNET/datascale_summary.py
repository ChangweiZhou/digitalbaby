"""Table of the data-scaling batch (datascale_batch.sh): for each net and stream, by bytes read T: the right share (clip readout, as in
the battery), bits per byte and nats per byte of the learned-temperature readout, and the bits of the back-off count table fitted on
the same T bytes. Usage: python3 datascale_summary.py datascale_results_*.jsonl"""
import sys, json, collections
R = collections.defaultdict(dict)
for f in sys.argv[1:]:
    for l in open(f):
        j = json.loads(l); net, st, _ = j["name"].rsplit("_", 2); R[(st, net)][j["T"]] = j
Ts = sorted({T for d in R.values() for T in d})
for st in ("moby", "code"):
    print(f"\n### {st}\n\n| net | " + " | ".join(f"T={T // 1000}k right / bits / nats (table bits)" for T in Ts) + " |\n|---|" + "---|" * len(Ts))
    for (s, net), d in sorted(R.items()):
        if s != st: continue
        cells = [f"{d[T]['right']['clip']:.3f} / {d[T]['bits']['temp']:.3f} / {d[T]['nats']['temp']:.3f} ({d[T]['table_bits']:.3f})" if T in d else "-" for T in Ts]
        print(f"| {net} | " + " | ".join(cells) + " |")
lens = {(j["stream"], j["stream_len"]) for d in R.values() for j in d.values()}
print("\nstream lengths seen:", sorted(lens))
