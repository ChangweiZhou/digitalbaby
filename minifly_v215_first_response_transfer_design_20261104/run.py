"""Complete engineering roster. The held science roster is deliberately absent."""
from collections import Counter
import json
import os
import resource
import time

from bridge import Learner
from controls import controlled, fork, qualified
from cycle1 import exact, fingerprint
from worlds import family

SEEDS = (2150301, 2150302)
ARMS = ("TRUE", "SHAM", "NO-RELATED", "BASE")


def assess(parent, probe):
    child = fork(parent)
    child.observe(probe.prompt)
    before = fingerprint(child)
    response = child.generate()
    assert before == fingerprint(child)
    assert len(response) <= 32
    return {"correct": exact(response, probe.answer), "hex": response.hex(),
            "terminated": response.endswith(b"\n")}


def diagnostic(parent, pairs):
    child = fork(parent)
    total, payload, eos, nonanswer, observed = 0., 0., 0., 0., 0
    # Independent online shadow stream.
    for twin in pairs:
        for probe in twin:
            scores = child.observe(probe.prompt + probe.answer + b"\n")
            boundary = len(probe.prompt)
            nonanswer += sum(scores[:boundary])
            payload += sum(scores[boundary:-1])
            eos += scores[-1]
            total += sum(scores)
            observed += len(scores)
    assert abs(total - (payload + eos + nonanswer)) < 1e-8
    return dict(bpb=total / observed, answer_bits=payload / 16, eos_bits=eos / 16,
                nonanswer_bits=nonanswer, observed=observed)


def run_family(seed):
    histories, interference, old, pairs = family(seed)
    output = {}
    audit = {}
    bytes_seen = 0
    for arm in ARMS:
        parent = Learner("0") if arm == "BASE" else controlled(arm == "SHAM", seed)
        history = histories[arm == "NO-RELATED"]
        parent.observe(history)
        old_before = assess(parent, old)
        parent.observe(interference)
        old_after = assess(parent, old)
        state_hash = fingerprint(parent)
        responses = [[assess(parent, probe) for probe in twin] for twin in pairs]
        diagnostics = diagnostic(parent, pairs)
        assert state_hash == fingerprint(parent)
        output[arm] = dict(short=sum(x["correct"] for twin in responses[:4] for x in twin),
                           long=sum(x["correct"] for twin in responses[4:] for x in twin),
                           both=sum(all(x["correct"] for x in twin) for twin in responses),
                           old_before=old_before, old_after=old_after, responses=responses,
                           compression=diagnostics)
        bytes_seen += (len(history) + len(interference) + 2 * len(old.prompt)
                       + sum(len(probe.prompt) for twin in pairs for probe in twin)
                       + diagnostics["observed"])
        if arm in ("TRUE", "SHAM"):
            # Compare on the same teacher-observed trajectory, never self-generated suffixes.
            common = fork(parent)
            for twin in pairs:
                for probe in twin:
                    common.observe(probe.prompt + probe.answer + b"\n")
            audit[arm] = dict(Counter(parent.control[3]) + Counter(common.control[3]))
            bytes_seen += diagnostics["observed"]
    checks, accepted = qualified(audit["TRUE"], audit["SHAM"])
    assert all(row["slots_equal"] for row in checks.values()), checks
    return dict(seed=seed, arms=output, control_counters=audit, matching=checks,
                sham_qualified=accepted, observed_bytes=bytes_seen, oracle_exact=16)


def main():
    assert os.environ.get("PYTHONHASHSEED") == "7"
    cpu = time.process_time()
    rows = [run_family(seed) for seed in SEEDS]
    report = dict(cycle=3, status="ENGINEERING_COMPLETE", science="NOT_RUN",
                  admission="ADMISSION_PENDING" if all(row["sham_qualified"] for row in rows)
                  else "CONTROL_NOT_QUALIFIED", families=rows,
                  cpu_seconds=time.process_time() - cpu,
                  process_peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
