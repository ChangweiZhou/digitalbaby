"""R readout calibration on reserved worlds 190101-190108 (label-blind; W branch to the end of the old stage).

Scales: s_b = sqrt(mean over worlds, taught old cues and all four channels of (v_b,j - mean_k v_b,k)^2).
Novelty threshold theta: median R1 pre-answer novelty over all 336 old records of the 8 lives.
No correctness, held-out relation or science world enters.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import stores  # noqa: E402
import systems  # noqa: E402
from common_platform import output_before_teacher, branch_allows  # noqa: E402
from fixture import DT, RECORD_SECONDS, make_world  # noqa: E402

WORLDS = tuple(range(190101, 190109))
OUT = paths.ROOT / "results" / "calibration" / "R_CALIBRATION.json"


def raw_bank_values(system: systems.RSystem, cue: bytes, at: float):
    scratch = system.clone()
    before = system.digests()
    for i, b in enumerate(cue):
        scratch.feed(b, at + i * DT)
    when = at + len(cue) * DT
    s = [float(m.association_value(when)) for m in scratch.shared]
    p = [float(m.association_value(when)) for m in scratch.private]
    if system.digests() != before:
        raise AssertionError("calibration read mutated life")
    return s, p


def one_world(world: int) -> dict:
    doc = make_world(world)
    shared = [stores.birth("native")[0] for _ in range(4)]
    private = [stores.birth("content")[0] for _ in range(4)]
    system = systems.RSystem(shared, private, "R0", scales=(1.0, 1.0), theta=0.0)
    for index, row in enumerate(doc["records"][:336]):
        if row["stage"] != "old":
            raise AssertionError("calibration left the old stage")
        begin = index * RECORD_SECONDS
        cue, answer = bytes.fromhex(row["cue_hex"]), row["answer"]
        system.set_record_context(world, "W", index, row["stage"], row["domain"])
        _, _, teacher_at, domain = output_before_teacher(system, cue, begin)
        system.teach(answer, teacher_at, domain, write=branch_allows("W", row["stage"], domain))
        system.feed(10, begin + 13 * DT)
        system.flush(begin + RECORD_SECONDS)
    system.flush(doc["old_end_s"])
    novelty = [r[2] for r in system.mechanism_events if r[0] == "R"]
    cues = doc["old_fact"]["cues_hex"] + [r["cue_hex"] for r in doc["old_relation"]["taught"]]
    reads = [raw_bank_values(system, bytes.fromhex(c), doc["old_end_s"]) for c in cues]
    return {"world": world, "fixture_digest": doc["digest"], "novelty": novelty,
            "cues": cues, "shared": [r[0] for r in reads], "private": [r[1] for r in reads]}


def scale(rows: list) -> float:
    dev = [(v - sum(r) / 4.0) ** 2 for r in rows for v in r]
    return (sum(dev) / len(dev)) ** 0.5


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    t0 = time.monotonic()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        per = list(ex.map(one_world, WORLDS))
    s_s = scale([r for w in per for r in w["shared"]])
    s_p = scale([r for w in per for r in w["private"]])
    novelty = [v for w in per for v in w["novelty"]]
    theta = statistics.median(novelty)
    if not all(x == x and x >= 1e-8 for x in (s_s, s_p)):
        raise AssertionError("R readout calibration technical failure")
    doc = {"schema": "MINIFLY-A3-CLAUDE-R-CALIBRATION-v1", "worlds": list(WORLDS),
           "scales": {"shared": s_s, "private": s_p}, "theta": theta,
           "novelty_quantiles": statistics.quantiles(novelty, n=10),
           "novelty_count": len(novelty), "elapsed_s": time.monotonic() - t0, "per_world": per,
           "source_sha256": {f: hashlib.sha256((paths.ROOT / f).read_bytes()).hexdigest()
                             for f in ("src/stores.py", "src/systems.py", "src/calibrate_r.py")}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("scales", "theta", "novelty_quantiles", "elapsed_s")}))


if __name__ == "__main__":
    main()
