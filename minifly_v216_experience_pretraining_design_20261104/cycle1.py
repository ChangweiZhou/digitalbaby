from hashlib import sha256
import json

from engine import Life, PARENT, T, new_sensor, source


def main():
    sensor = new_sensor(fast=False)
    x = Life(21601, sensor, fast=False)
    y = source.Learner("21601")
    record = b"Oa1XabMa2Qb2"
    ids = ["0"] * (len(record)+1)
    y.step(x.codes(record), T.tensor([*record, 256]), ids, "NORMAL")
    x.train(record, gain=1)
    assert all(T.equal(x.model.p[k], y.p[k]) for k in y.p)
    assert all(T.equal(x.model.m[k], y.m[k]) and T.equal(x.model.v[k], y.v[k]) for k in y.p)
    assert T.equal(x.model.rng.get_state(), y.rng.get_state())
    assert T.equal(x.codes(record)[:8], x.codes(record[:7] + b"changed")[:8])
    clone = x.fork(21601)
    output = x.generate(record[:7])
    assert clone.generate(record[:7]) == output
    assert all(T.equal(x.model.p[k], clone.model.p[k]) for k in y.p)
    assert all(T.equal(x.model.m[k], clone.model.m[k]) and
               T.equal(x.model.v[k], clone.model.v[k]) for k in y.p)
    assert x.events == clone.events and x.model.clock == clone.model.clock
    assert T.equal(x.model.rng.get_state(), clone.model.rng.get_state())
    x.train(record)
    clone.train(record)
    assert all(T.equal(x.model.p[k], clone.model.p[k]) for k in y.p)
    print(json.dumps(dict(cycle=1, exact_normal_update=True, prefix_invariance=True,
                          fork_continuation=True, generated_symbols=output,
                          parameters=sum(v.numel() for v in y.p.values()),
                          sensor_digest=sensor.digest, source={name:sha256((PARENT / name).read_bytes()).hexdigest()
                                                             for name in ("learner.py","sensor.py")})))


if __name__ == "__main__":
    main()
