"""V216 fixed eight-family development roster and engineering admission."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import resource
import time

from engine import EOS, Life, T, new_sensor
from environment import Episode
from run import score

DOSES = (1000, 10000, 100000)
LIMITS = dict(events=1600000, observed_bytes=60000000, target_sites=64000000,
              cpu=28800, wall=36000, rss=1 << 30, output=256 << 20)


def stable(*parts):
    return int.from_bytes(sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def episode(actors, places, op, attach, motif):
    a,b,c,d = actors
    left,right,dest = places
    initial = bytes((79,a,left,79,b,right))
    focal = bytes((op,a,b))
    distractor = bytes((88,c,d))
    if motif == "train":
        middle = distractor + focal
    else:
        middle = focal + distractor
        if motif == "long":
            middle += bytes((89,d,c))
    return Episode(initial + middle + bytes((77,a,dest,81,b)),
                   bytes((dest if op == attach else right,)))


def read_truth(prefix, attach):
    a,b = prefix[1], prefix[4]
    seen = [prefix[i] for i in range(6,len(prefix)-5,3)
            if prefix[i+1:i+3] == bytes((a,b))]
    assert len(seen) == 1 and prefix[-5:-3] == bytes((77,a))
    assert prefix[-2:] == bytes((81,b))
    return bytes((prefix[-3] if seen[0] == attach else prefix[5],))


def training(key, n, worlds, attach, start=128):
    identities = Random(stable(key,"identities"))
    actors = list(range(start,start+32))
    panels = [tuple(identities.sample(actors,4)) for _ in range(worlds)]
    assert len(set(panels)) == worlds
    obs = Random(stable(key,"obs"))
    for i in range(n):
        panel = panels[i % worlds]
        op = b"XY"[((i % worlds) + (i // worlds)) % 2]
        places = obs.sample(b"01234567",3)
        if start == 128 and i % 32 == 0:
            places[-1] = places[1]
        yield episode(panel, places, op, attach, "train")


def held(key, attach, invariant=False):
    rng = Random(key)
    panel = []
    for i in range(16):
        ids = rng.sample(range(192,224),4)
        places = rng.sample(b"01234567",3)
        if invariant:
            places[-1] = places[1]
        motif = "short" if i < 8 else "long"
        pair = [episode(ids, places, op, attach, motif) for op in b"XY"]
        assert pair[0].prefix[-8:] == pair[1].prefix[-8:]
        assert (pair[0].target == pair[1].target) == invariant
        assert all(read_truth(p.prefix,attach) == p.target for p in pair)
        panel.extend(pair)
    return panel


class Count:
    def __init__(self):
        self.rows = defaultdict(Counter)
        self.lookup = {}
        self.labels = Counter()
        self.canonical = defaultdict(Counter)
        self.rule = defaultdict(Counter)

    def observe(self, e):
        prefix = b""
        for symbol in (*e.record, EOS):
            for order in range(4):
                self.rows[prefix[-order:] if order else b""][symbol] += 1
            if symbol != EOS:
                prefix += bytes((symbol,))
        self.lookup[e.prefix] = e.target[0]
        self.labels[e.target[0]] += 1
        self.canonical[self.key(e.prefix)][e.target[0]] += 1
        self.rule[e.prefix[9]][e.target[0] == e.prefix[-3]] += 1

    @staticmethod
    def mode(row):
        return min(row, key=lambda c: (-row[c],c)) if row else None

    def byte3(self, prefix):
        answer = []
        for _ in range(32):
            token = next((value for size in (3,2,1,0)
                          if (value := self.mode(self.rows.get(prefix[-size:] if size else b""))) is not None), EOS)
            answer.append(token)
            if token == EOS:
                break
            prefix += bytes((token,))
        return answer

    @staticmethod
    def key(prefix):
        op = prefix[9] if len(prefix) == 17 and prefix[6] == 88 else prefix[6]
        return (prefix[2], prefix[5], op, prefix[-3])

    def report(self, panel):
        rows = defaultdict(list)
        for p in panel:
            op = p.prefix[6]
            structured = self.mode(self.rule.get(op))
            predictions = dict(exact_lookup=self.lookup.get(p.prefix),
                               normalized_lookup=self.mode(self.canonical.get(self.key(p.prefix))),
                               recent=p.prefix[-3], frequency=self.mode(self.labels),
                               structured=p.prefix[-3] if structured else p.prefix[5])
            for name,value in predictions.items():
                rows[name].append(value == p.target[0])
            rows["byte3"].append(self.byte3(p.prefix) == [*p.target,EOS])
        return {name:dict(exact=sum(row), both=sum(a and b for a,b in zip(row[::2],row[1::2])))
                for name,row in rows.items()}


def evaluate(life, panel, tally):
    tally["observed_bytes"] += sum(2*len(p.prefix)+1 for p in panel)
    result = score(life,panel)
    tally["target_sites"] += sum(len(p.record)+1 for p in panel) + sum(map(len,result["outputs"]))
    return result


def train(life,e,tally,gain=None):
    life.train(e.record,gain=gain)
    tally["observed_bytes"] += len(e.record)
    tally["target_sites"] += len(e.record)+1


def shadow(life,birth,panel,tally):
    fork = life.fork(birth)
    payload = eos = other = 0
    for probe in panel:
        loss = fork.loss(probe.record)
        payload += loss[-2]
        eos += loss[-1]
        other += sum(loss[:-2])
        tally["observed_bytes"] += len(probe.record)
        tally["target_sites"] += len(loss)
        train(fork,probe,tally)
    return dict(payload_bits=payload, eos_bits=eos, nonanswer_bits=other,
                bits_per_byte=(payload+other)/sum(len(e.record) for e in panel))


def adaptation(parent,birth,new,changed,old,tally,mode):
    life = parent.fork(birth)
    results = {0:evaluate(life,changed,tally)}
    for i,e in enumerate(new,1):
        if mode == "frozen":
            life.loss(e.record)
            tally["observed_bytes"] += len(e.record)
            tally["target_sites"] += len(e.record)+1
        else:
            train(life,e,tally,gain=1 if mode=="matched" else None)
        if i in (1,4,16,64):
            results[i] = evaluate(life,changed,tally)
    if mode=="frozen":
        assert all(T.equal(parent.model.p[k],life.model.p[k]) for k in life.model.p)
    return dict(changed=results, old_after=evaluate(life,old,tally) if old else None)


def family(index, output, trial=False):
    started = time.process_time()
    label = "V216|PREFLIGHT" if trial else "V216|DEV"
    key = stable(label,index)
    attach = b"XY"[index%2]
    swapped = b"YX"[index%2]
    birth = stable(key,"birth")
    sensor = new_sensor()
    panel = held(stable(key,"shared"),attach)
    changed = held(stable(key,"changed"),swapped)
    adapt = list(training(stable(key,"adapt"),64,4,swapped,start=160))
    assert all(read_truth(e.prefix,swapped)==e.target!=read_truth(e.prefix,attach) for e in adapt)
    n = 256 if trial else DOSES[-1]
    tally = Counter()
    cold = Life(birth,sensor)
    result = dict(index=index, trial=trial, cold=evaluate(cold,panel,tally), arms={})
    result["cold_adapt"] = {mode:adaptation(cold,birth,adapt,changed,[],tally,mode)
                            for mode in ("matched","natural")}
    for name,worlds in (("FEW",4),("MANY",32)):
        learner = Life(birth,sensor)
        ref = Count()
        rows = {}
        seen = set()
        counts = Counter()
        old = []
        train_cpu = 0.0
        for i,event in enumerate(training(stable(key,"train"),n,worlds,attach),1):
            assert len(event.record)==18 and read_truth(event.prefix,attach)==event.target
            counts[event.prefix[9]] += 1
            seen.add(event.record)
            if event.prefix[5] == event.prefix[-3] and len(old)<8:
                old.append(event)
            clock = time.process_time()
            train(learner,event,tally)
            ref.observe(event)
            train_cpu += time.process_time()-clock
            if i % 256 == 0:
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 480 * (1<<20), "worker RSS cap"
                assert time.process_time()-started < 3500, "worker CPU cap"
            tally["events"] += 1
            # Reference is charged for its own record application.
            tally["observed_bytes"] += len(event.record)
            tally["target_sites"] += len(event.record)+1
            if i in (n,) if trial else i in DOSES:
                assert counts[88] == counts[89]
                rows[i] = dict(shared=evaluate(learner,panel,tally),
                               online_shadow=shadow(learner,birth,panel,tally),
                               references=ref.report(panel), unique_records=len(seen),
                               opcode_counts=dict(counts), worlds=worlds)
                if not trial:
                    path = output/f"state_{index}_{name}_{i}.pt"
                    tmp = path.with_suffix(".tmp")
                    T.save(dict(events=learner.events,state=learner.model.state()),tmp)
                    tmp.replace(path)
                write(output/f"progress_{index}_{name}.json",dict(index=index,arm=name,checkpoint=i,
                      tally=dict(tally),cpu_seconds=time.process_time()-started))
        result["arms"][name] = dict(checkpoints=rows, train_cpu=train_cpu, old_before=evaluate(learner,old,tally),
                                     adaptation={mode:adaptation(learner,birth,adapt,changed,old,tally,mode)
                                                 for mode in ("matched","natural","frozen")})
    result.update(tally=dict(tally),cpu_seconds=time.process_time()-started,
                  rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(output/f"family_{index}.json",result)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("index",type=int,choices=range(8))
    p.add_argument("output",type=Path)
    p.add_argument("--trial",action="store_true")
    a = p.parse_args()
    family(a.index,a.output,a.trial)
