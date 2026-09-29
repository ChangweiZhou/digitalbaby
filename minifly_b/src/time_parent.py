"""Resource probe: canonical-birth FE0 four-store parent life on TECHNICAL world 190000 only."""
import sys, time, json, resource, cProfile, pstats, io
sys.dont_write_bytecode = True
from pathlib import Path
PKG = Path(__file__).resolve().parents[1] / "package"
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "REFERENCE_SOURCE/MINIFLY_THREE_MECHANISM_ROUND_20260928"))
import portable_birth as pb
from common_platform import FourStore, clone_model, run_fourstore_life
t0 = time.monotonic()
base, raw = pb.canonical_fresh_native()
actor = FourStore([clone_model(base) for _ in range(4)])
pr = cProfile.Profile(); pr.enable()
t1 = time.monotonic()
doc = run_fourstore_life(190000, actor, technical=True)
t2 = time.monotonic(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
print(s.getvalue()[:6000])
print(json.dumps({"life_s": round(t2 - t1, 1), "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "receipt_json_bytes": len(json.dumps(doc, default=str)), "n_probes": len(doc["probes"])}))
