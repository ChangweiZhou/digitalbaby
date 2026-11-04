"""Bounded engineering entry point. No held science roster is generated."""
import argparse
import json
import resource
import time

from engine import EOS, Life, T, new_sensor
from environment import exposure, oracle, seed, stream, twins


def score(life, panel):
    generated = [life.generate(probe.prefix) for probe in panel]
    hits = [value == [*probe.target, EOS] for probe, value in zip(panel, generated)]
    losses = [life.loss(probe.record) for probe in panel]
    return dict(exact=sum(hits), count=len(panel), eos=sum(x[-1] == EOS for x in generated),
                both=sum(a and b for a,b in zip(hits[::2], hits[1::2])),
                payload_bits=sum(x[-2] for x in losses)/len(losses),
                eos_bits=sum(x[-1] for x in losses)/len(losses),
                full_record_bits_per_target=sum(map(sum, losses))/sum(map(len, losses)))


def one(count, cycle):
    sensor = new_sensor()
    attach = ord("X")
    swapped = ord("Y")
    label = str(cycle)
    birth = seed(label + "|birth")
    panel = twins(seed(label + "|shared"), attach)
    changed = twins(seed(label + "|changed"), swapped)
    assert sum(p.prefix[11] == p.target[0] for p in panel) <= len(panel)//2
    warm = {}
    scores = {}
    elapsed = {}
    checkpoints = (count,) if cycle == 2 else (100, count)
    for arm, worlds in (("FEW", 4), ("MANY", 32)):
        life = Life(birth, sensor)
        records = list(stream(seed(label+"|train"), count, worlds, attach))
        exposure(records, panel)
        assert all(oracle(x.prefix, attach) == x.target for x in records)
        assert len({len(x.record) for x in records}) == 1
        scores[arm] = {}
        begin = time.process_time()
        costs = []
        for i, event in enumerate(records, 1):
            start = time.process_time()
            life.train(event.record)
            costs.append(time.process_time()-start)
            if i in checkpoints:
                scores[arm][i] = score(life, panel)
        elapsed[arm] = dict(training_cpu=sum(costs), tail_cpu=sum(costs[-100:]),
                            total_cpu=time.process_time()-begin, events=count,
                            bytes=sum(len(x.record) for x in records))
        warm[arm] = life
    assert elapsed["FEW"]["bytes"] == elapsed["MANY"]["bytes"]
    warm["COLD"] = Life(birth, sensor)
    adaptation = list(stream(seed(label+"|adapt"), 64 if cycle == 3 else 16, 4, swapped, 32))
    assert all(oracle(x.prefix, swapped) == x.target != oracle(x.prefix, attach) for x in adaptation)
    adapt = {}
    for name, parent in warm.items():
        adapt[name] = {}
        child = parent.fork(birth)
        adapt[name][0] = score(child, changed)
        for i,event in enumerate(adaptation, 1):
            child.train(event.record, gain=1)
            if i in (1,4,16,64):
                adapt[name][i] = score(child, changed)
        if name != "COLD":
            # Every engineering episode has an explicit sensory reset.
            # Consume the identical records with observation-only forward passes.
            frozen = parent.fork(birth)
            for event in adaptation:
                frozen.loss(event.record)
            assert all(T.equal(frozen.model.p[k], parent.model.p[k]) for k in parent.model.p)
            adapt[name]["NO_WRITE"] = score(frozen, changed)
            assert adapt[name]["NO_WRITE"] == adapt[name][0]
    estimated = 8 * sum(x["tail_cpu"]/min(count,100) for x in elapsed.values()) * 100_000 * 1.25
    return dict(cycle=cycle, science="NOT_RUN", learning=scores, adaptation=adapt,
                timings=elapsed, estimated_training_only_cpu_seconds=estimated,
                admission="RESOURCE_NOT_ADMITTED" if estimated > 28800 else "ADMISSION_PENDING",
                held_oracle=len(panel), changed_oracle=len(changed),
                peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, choices=(2,3), required=True)
    cycle = parser.parse_args().cycle
    print(json.dumps(one(64 if cycle==2 else 1000, cycle), indent=2))
