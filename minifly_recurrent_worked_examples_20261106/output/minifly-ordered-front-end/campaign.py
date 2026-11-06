"""Locked ordered-front-end development roster with resumable per-life workers."""
import argparse
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import resource
import time

import numpy as np

from qualify import ARMS, D, Backend, Encoder, EOS, Record, parameter_count, score, source_receipt, state_hash
from revised_panel import BLOCKS, LENGTHS, TARGET_SYMBOLS, certify, held, material, stream

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "campaign"
STEPS = ("cold", "A_10k", "A_100k", "B", "A_prime", "noedit")
NORMAL_CLOCKS = (0, 10000, 100000, 110000, 120000, 120000)
PHASES = ("A", "B", "A_prime")


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def sources(archive):
    result = source_receipt(archive)
    result["campaign_files"] = {name: checksum(ROOT / name)
                                for name in ("campaign.py", "qualify.py", "revised_panel.py", "CAMPAIGN_DESIGN.md")}
    return result


class CachedCodes:
    """Bounded cache over the fixed rendered roster. Exact feature values."""
    def __init__(self, base):
        self.base = base
        self.rows = {}
        self.prefixes = 0
        self.prefix_state = base.stock.prefix_state
        self.encode = base.stock.encode
        self.perm, self.R = base.stock.perm, base.stock.R

    def reset(self):
        self.rows.clear()

    def codes(self, record):
        packed = self.rows.get(record)
        if packed is None:
            packed = np.packbits(self.base.codes(record) > 0, axis=1)
            if len(self.rows) < 8192:
                self.rows[record] = packed
            self.prefixes += 1
        return np.unpackbits(packed, axis=1, count=D).astype(np.float32)


def builder(backend, arm, world):
    encoder = Encoder(arm, backend.sensor)
    cached = CachedCodes(encoder)

    class Life(backend.Life):
        def generate(self, prefix):
            if arm == "stock":
                return super().generate(prefix)
            output = []
            for _ in range(32):
                # This is exactly the last row of Encoder.codes, without discarded rows.
                raw = prefix[-64:]
                if arm == "bag" and len(raw) > 8:
                    raw = bytes(sorted(raw[:-8])) + raw[-8:]
                slots = np.arange(len(raw))[:, None] * 80 + encoder.book[np.frombuffer(raw[::-1], dtype=np.uint8)]
                x = np.zeros((1,D), dtype=np.float32)
                x[0,slots] = 1
                y = int(self.model.forward(backend.T.from_numpy(x))[0][0].argmax())
                output.append(y)
                if y == EOS:
                    break
                prefix += bytes((y,))
            return output

    return cached, lambda: Life(world, cached)


def save(backend, lives, path, cpu_seconds):
    tmp = path.with_suffix(".tmp")
    backend.T.save({"states": [x.model.state() for x in lives],
                    "events": [x.events for x in lives], "cpu_seconds": cpu_seconds}, tmp)
    tmp.replace(path)


def load(backend, lives, path):
    data = backend.T.load(path, weights_only=True, map_location="cpu")
    for life, state, events in zip(lives, data["states"], data["events"]):
        life.model.restore(state)
        life.events = events
        assert life.model.clock == life.events
    return data["cpu_seconds"]


def phase_score(lives, block, key, banks, sham=False):
    output = []
    for i, life in enumerate(lives):
        law = i
        measures = {}
        if key in ("cold", "A_10k", "A_100k", "B", "A_prime", "noedit"):
            laws = range(2) if sham else [law ^ (key == "A_prime")]
            for variant in laws:
                measures[f"held_A_law{variant}"] = score(life, held(block, "A", variant))
        if key in ("B", "A_prime", "noedit"):
            measures["held_B"] = score(life, held(block, "B", law))
        if key in ("A_10k", "A_100k"):
            # Every index here occurs within the first complete teaching epoch.
            bank = banks["A"]
            ids = [i * 61 for i in range(64)]
            rows = [Record(bank[j][0], bank[j][1][law]) for j in ids]
            measures["taught_A"] = score(life, list(zip(rows[::2],rows[1::2])))
        output.append(measures)
    return output


def profile(backend, root):
    from qualify import make_life
    out = {}
    block = BLOCKS[0]
    bank = material(block, "A")
    for arm in ARMS:
        cache, make = builder(backend, arm, block)
        first = make()
        assert parameter_count(first) == 37313
        ref = backend.birth(block, 64) if arm == "stock" else make_life(backend, arm)
        # The rehearsal helper uses WORLD. Copy this campaign birth and RNG into it.
        ref.model.restore(first.model.state())
        sample = [bank[i * 17][0] + bytes((bank[i * 17][1][i % 2],)) for i in range(16)]
        for record in sample:
            assert np.array_equal(cache.codes(record), ref.codes(record).numpy())
            first.train(record)
            ref.train(record)
            assert state_hash(first) == state_hash(ref)
        for p in held(block, "A", 0)[:3]:
            assert all(first.generate(x.prefix) == ref.generate(x.prefix) for x in p)
        checkpoint = root / f"profile-{arm}.pt"
        save(backend, [first], checkpoint, 0)
        fresh = make()
        load(backend, [fresh], checkpoint)
        first.train(sample[0])
        fresh.train(sample[0])
        assert state_hash(first) == state_hash(fresh)
        checkpoint.unlink()

        t = time.process_time()
        for _ in range(8):
            for row in sample:
                first.train(row)
        stage = {"cached": (time.process_time() - t) / (8 * len(sample))}
        cache.reset()
        t = time.process_time()
        for prefix, ys, *_ in bank[128:256]:
            first.train(prefix + bytes((ys[0],)))
        stage["cold"] = (time.process_time() - t) / 128
        out[arm] = {"cpu_seconds_per_event": stage, "parity_updates":16,
                    "parity_generation":6, "restart":True}
    return out


def worker(archive, block, arm, sham):
    lock = read(RUN / "LOCK.json")
    assert lock["sources"] == sources(archive) and lock["panel"]["status"] == "PANEL_QUALIFIED"
    assert block in BLOCKS and arm in ARMS
    name = f"{block}-{arm}" + ("-sham" if sham else "")
    root = RUN / name
    root.mkdir(exist_ok=True)
    backend = Backend(archive)
    cache, make = builder(backend, arm, block)
    lives = [make() for _ in range(1 if sham else 2)]
    assert len({state_hash(x) for x in lives}) == 1
    begin, cpu_begin = time.monotonic(), time.process_time()
    cumulative_cpu = 0
    banks = {phase: material(block, phase) for phase in (("A",) if sham else PHASES)}
    jobs = [("cold","A",0,0),("A_10k","A",0,10000),("A_100k","A",10000,100000)]
    if not sham:
        jobs += [("B","B",0,10000),("A_prime","A_prime",0,10000),("noedit","A_prime",0,10000)]
    progress_path = root / "STATUS.json"
    completed = 0
    for key, phase, start, finish in jobs:
        done = root / f"{key}.json"
        if done.exists():
            completed += (finish-start) * len(lives)
            continue
        if key != "cold":
            parent = {"A_10k":"cold","A_100k":"A_10k","B":"A_100k","A_prime":"B","noedit":"B"}[key]
            mid = root / f"{key}-partial.pt"
            cpu = load(backend, lives, root / f"{parent}.pt")
            offset = start
            if mid.exists():
                cpu = load(backend, lives, mid)
                offset += lives[0].events - read(root / f"{parent}.json")["clock"]
                assert start <= offset <= finish
            # Each stage consumes a fixed literal prefix of a registered stream.
            if key not in ("A_100k", "noedit"):
                cache.reset()
            bank = banks[phase]
            seq = islice(stream(block, phase, bank), offset, finish)
            stage_clock = time.process_time()
            for step, j in enumerate(seq, offset + 1):
                prefix, ys, stay, move = bank[j]
                for law, life in enumerate(lives):
                    if sham:
                        # Opposing outcomes alternate on each exact rendering by epoch.
                        y = (stay, move)[(j + (step - 1)//len(bank)) % 2]
                    else:
                        y = ys[law ^ (key == "A_prime")]
                    life.train(prefix + bytes((y,)))
                if step % 2000 == 0 or step == finish:
                    cumulative_cpu = time.process_time() - cpu_begin
                    save(backend, lives, mid, cpu + time.process_time() - stage_clock)
                    write(progress_path, {"stage":key,"stage_records_per_life":step,
                                         "updates_this_worker":completed + (step-start)*len(lives),
                                         "clock":lives[0].events, "session_cpu_seconds":cumulative_cpu,
                                         "session_wall_seconds":time.monotonic()-begin})
            if key != "noedit":
                save(backend, lives, root / f"{key}.pt", cpu + time.process_time() - stage_clock)
            completed += (finish-start) * len(lives)
        else:
            save(backend, lives, root / "cold.pt", 0)
        scored = phase_score(lives, block, key, banks, sham)
        write(done, {"stage":key, "clock":lives[0].events, "updates_this_worker":completed,
                     "state_hashes":[state_hash(x) for x in lives], "panels":scored})
        if key != "cold":
            mid.unlink(missing_ok=True)
        print(name, key, lives[0].events, flush=True)
    write(progress_path, {"stage":"COMPLETE", "updates_this_worker":completed,
                         "session_cpu_seconds":time.process_time()-cpu_begin,
                         "session_wall_seconds":time.monotonic()-begin,
                         "max_rss_bytes_macos":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "computed_feature_batches_this_session":cache.prefixes})


def prepare(archive):
    RUN.mkdir(exist_ok=True)
    path = RUN / "LOCK.json"
    assert not path.exists()
    panel = certify()
    timings = profile(Backend(archive), RUN)
    write(path, {"sources":sources(archive), "panel":panel, "profile":timings,
                 "blocks":BLOCKS,"arms":ARMS,"targets_per_event":TARGET_SYMBOLS,
                 "doses":LENGTHS,"main_events":1560000,"main_symbols":1560000*TARGET_SYMBOLS,
                 "matched_no_teaching_control_if_positive":True})
    print("Panel and native parity qualified", panel["registered_shortcut_ceiling"], timings, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive",type=Path)
    ap.add_argument("mode",choices=("prepare","worker","sham"))
    ap.add_argument("block",nargs="?")
    ap.add_argument("arm",nargs="?")
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.archive)
    else:
        worker(a.archive,a.block,a.arm,a.mode == "sham")


if __name__ == "__main__":
    main()
