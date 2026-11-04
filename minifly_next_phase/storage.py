"""Four paired seeds on the recovered V40 SUPER arm and record stream."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'minifly_v39_v40_archive'
sys.path.insert(0, str(SOURCE))
import elm_fly_complementary_v40 as reference
import numpy as np

from functools import lru_cache
import resource

SEEDS = tuple(range(20261103501, 20261103505))
CELLS = [(128, 16384, 205), (512, 65536, 410)]

class Layer(reference.SuperposedLayer):
    def __init__(self, seed, units, k, dtype):
        super().__init__(seed, units, k)
        self.output = np.zeros((units, reference.OUTPUTS), dtype=dtype)
        self.cached = lru_cache(maxsize=4096)(super().encode)

    def encode(self, window):
        return self.cached(bytes(window))

    def observe(self, window, y):
        ids, values = self.encode(window)
        block = self.output[ids].astype(np.float64)
        used = values @ block
        norm = float(values @ values)
        if norm > 1e-15:
            error = -used
            error[y] += 1
            self.output[ids] = block + reference.RATE * np.outer(values/norm, error)
            self.updates += 1
        self.ops += self.k * reference.OUTPUTS + self.units * self.n_contacts + self.units

def run(job):
    seed, load, units, k, dtype = job
    start = time.perf_counter()
    stream = reference.workload(seed, keys_per_wave=load // 4)
    layer = Layer(seed, units, k, dtype)
    history = bytearray()
    checkpoints = {cp['record_count']: cp['name'] for cp in stream['checkpoints']
                   if cp['name'] in ('after_D', 'after_clutter') or cp['type'] == 'revision'}
    old = stream['untouched']
    taught = {row['raw'][:12] for row in stream['rows']}
    assert all(reference.context(stream['keys'][i], 2) not in taught for i in old)
    metrics = {}
    one_shot = []
    def accuracy(ids, aliases, revised=False):
        hits = [int(np.argmax(layer.scores(reference.context(stream['keys'][i], a))) ==
                    (stream['revisions'][i] if revised else stream['labels'][i]))
                for i in ids for a in aliases]
        return sum(hits) / len(hits)
    for n, row in enumerate(stream['rows'], 1):
        for pos, y in enumerate(row['raw']):
            layer.observe(history, y)
            if pos == reference.CONTEXT_BYTES and row['kind'] == 'acquisition' and row['exposure'] == 1:
                i = row['key_id']
                c = reference.context(stream['keys'][i], 2)
                assert c not in taught
                one_shot.append(int(np.argmax(layer.scores(c)) == y))
            history.append(y)
            if len(history) > reference.CONTEXT_BYTES:
                del history[0]
        if n in checkpoints:
            name = checkpoints[n]
            if name in ('after_D', 'after_clutter'):
                metrics[name] = {'old': accuracy(old, (0, 1)), 'unseen': accuracy(old, (2,))}
            else:
                metrics[name] = {'revision_unseen': accuracy(stream['revised'], (2,), True)}
    assert 'after_D' in metrics
    record = dict(seed=seed, load=load, units=units, active=k, stream_hash=stream['raw_sha256'],
                  metrics=metrics, one_shot=sum(one_shot)/len(one_shot),
                  costs=layer.cost(), dtype=dtype, byte_events=len(stream['raw']),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  cache=layer.cached.cache_info()._asdict(), seconds=time.perf_counter()-start)
    path = ROOT / ('%d-%d-%d-%d-%s.json' % job)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2)+'\n')
    tmp.replace(path)
    print(path.name, metrics['after_D'], flush=True)

if __name__ == '__main__':
    (ROOT/'STORAGE_PLAN.json').write_text(json.dumps(dict(seeds=SEEDS, cells=CELLS, dtypes=['float64','float16'], gate=dict(retention=.90,max_drop=.05), sources={
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')
    }), indent=2)+'\n')
    jobs = [(seed, *cell, dtype) for cell in CELLS for seed in SEEDS for dtype in ('float64','float16')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, jobs))
