"""V216 NORMAL learning with a single width variation for engineering."""

from hashlib import sha256
from math import sqrt
from pathlib import Path
import sys


class Backend:
    def __init__(self, archive):
        self.archive = Path(archive).resolve()
        source_dir = self.archive / "minifly_v216_experience_pretraining_design_20261104"
        sys.path.insert(0, str(source_dir))
        from engine import Life, T, new_sensor, source
        from fast import normal_gradients

        self.Life, self.T, self.source = Life, T, source
        self.sensor = new_sensor()
        self.normal_gradients = normal_gradients

    def birth(self, world, width):
        life = self.Life(world, self.sensor)
        if width == 64:
            return life
        if width != 128:
            raise ValueError(width)
        m, T = life.model, self.T
        rng = T.Generator().manual_seed(self.source.seed(f"{world}|T128-extra"))
        feedback_rng = T.Generator().manual_seed(self.source.seed("T128-fixed-feedback-extra"))
        replacements = {
            "T": T.cat((m.p["T"], T.randn((64, 64), generator=rng) / 8)),
            "Tb": T.zeros(128),
            "O": T.cat((m.p["O"], T.randn((257, 64), generator=rng) / 8), dim=1) / sqrt(2),
        }
        for key, value in replacements.items():
            m.p[key] = value
            m.m[key], m.v[key] = T.zeros_like(value), T.zeros_like(value)
        m.ft = T.cat((m.ft, T.randn((257, 64), generator=feedback_rng) / sqrt(257)), dim=1)
        return life

    def clone(self, life, world, width):
        branch = self.birth(world, width)
        branch.model.restore(life.model.state())
        branch.events = life.events
        return branch

    def checkpoint(self, life, path, world, width):
        self.T.save({"state": life.model.state(), "events": life.events, "world": world, "width": width}, path)

    def restore(self, path):
        data = self.T.load(path, map_location="cpu", weights_only=True)
        life = self.birth(data["world"], data["width"])
        life.model.restore(data["state"])
        life.events = data["events"]
        assert life.events == life.model.clock
        return life


def state_hash(life):
    digest = sha256(str(life.events).encode())
    state = life.model.state()
    for key in ("weights", "moment1", "moment2"):
        for name, value in sorted(state[key].items()):
            digest.update(f"{key}|{name}|{tuple(value.shape)}".encode())
            digest.update(value.detach().numpy().tobytes())
    digest.update(state["dropout_rng"].numpy().tobytes())
    digest.update(str(state["clock"]).encode())
    return digest.hexdigest()


def parameter_count(life):
    return sum(value.numel() for value in life.model.p.values())
