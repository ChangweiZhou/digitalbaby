"""Engineering wrapper around the V181 sensor and NORMAL update."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

PARENT = Path(__file__).resolve().parent.parent / "minifly_v181_mini_local_plasticity_dev_20261102"


def load(name):
    spec = spec_from_file_location("v181_" + name, PARENT / (name + ".py"))
    mod = module_from_spec(spec)
    paths = sys.path[:]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = paths
    return mod


source = load("learner")
T = source.T
EOS = 256


class Life:
    def __init__(self, seed, sensor, *, fast=True):
        self.model = source.Learner(str(seed))
        self.sensor = sensor
        self.events = 0
        self.fast = fast

    def codes(self, record):
        return T.from_numpy(self.sensor.codes(record))

    def train(self, record, gain=None):
        u = min(self.events / 999, 1)
        gain = 0.1 + 0.9 * (3*u*u - 2*u*u*u) if gain is None else gain
        codes = self.codes(record)
        targets = T.tensor([*record, EOS])
        if self.fast:
            from fast import normal_gradients
            grad = normal_gradients(self.model, codes, targets)
        else:
            grad, _, _ = self.model.gradients(codes, targets, [str(self.events)] * len(targets))
        m = self.model
        m.clock += 1
        for key, p in m.p.items():
            m.m[key].mul_(.9).add_(grad[key], alpha=.1)
            m.v[key].mul_(.999).addcmul_(grad[key], grad[key], value=.001)
            delta = -.001 * (m.m[key] / (1-.9**m.clock) /
                             (m.v[key] / (1-.999**m.clock)).sqrt().add(1e-8)) - .001*.0001*p
            p.add_(delta if gain == 1 else delta * gain)
            assert T.isfinite(p).all()
        self.events += 1

    def generate(self, prefix):
        if self.fast and len(prefix) < 64:
            p = self.sensor.prefix_state(prefix)
        else:
            p = np.zeros(88)
            for byte in prefix:
                p = (p*.95)[self.sensor.perm] + self.sensor.R[byte]
        out = []
        for _ in range(32):
            x = T.from_numpy(self.sensor.encode(p*.95)[None])
            symbol = int(self.model.forward(x)[0][0].argmax())
            out.append(symbol)
            if symbol == EOS:
                break
            p = (p*.95)[self.sensor.perm] + self.sensor.R[symbol]
        return out

    def fork(self, seed):
        other = Life(seed, self.sensor, fast=self.fast)
        other.model.restore(self.model.state())
        other.events = self.events
        return other

    def loss(self, record):
        logits = self.model.forward(self.codes(record))[0]
        targets = T.tensor([*record, EOS])
        return (-logits.log_softmax(1)[T.arange(len(targets)), targets] / np.log(2)).tolist()


def new_sensor(*, fast=True):
    sensor = load("sensor").Sensor()
    if not fast:
        return sensor
    from fast import FastSensor
    return FastSensor(sensor)
