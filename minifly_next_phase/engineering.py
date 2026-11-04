"""Three engineering cycles on protocol fixtures, without sealed worlds."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

def digest(value):
    return hashlib.sha256(value).hexdigest()

def parse(raw):
    if len(raw) != 7 or raw[:4] != b'par ' or raw[-1:] != b'=':
        raise ValueError('Expected a complete parity query prefix')
    return raw[4], raw[5]

def trajectory(dtype, seed, stop, state=None):
    rng = np.random.default_rng(seed)
    weights = np.zeros((64, 256), dtype=dtype) if state is None else state.copy()
    for n in range(32):
        ids = rng.choice(64, 8, replace=False)
        values = rng.normal(size=8)
        values /= np.linalg.norm(values)
        y = int(rng.integers(256))
        if n >= stop:
            break
        if state is not None and n < 16:
            continue
        block = weights[ids].astype(np.float64)
        error = -(values @ block)
        error[y] += 1
        weights[ids] = block + .3*np.outer(values,error)
    return weights

def job(spec):
    dtype,seed = spec
    weights = trajectory(dtype,seed,32)
    return [dtype,seed,digest(weights.tobytes()),weights.nbytes]

def restore(path, expected_hash):
    if digest(path.read_bytes()) != expected_hash:
        raise ValueError('Checkpoint digest mismatch')
    return np.load(path,allow_pickle=False)

if __name__ == '__main__':
    assert parse(b'par ab=') == (97,98)
    try:
        parse(b'par ab=<')
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted feedback in probe')
    for parity in (0,1):
        held = {2,10,3,9} if parity == 0 else {4,8,5,7}
        assert held == {12-d for d in held} and sum(d%2 for d in held)==2
    jobs = [(dtype,seed) for dtype in ('float64','float16') for seed in (101,102)]
    serial = list(map(job,jobs))
    assert serial[0][-1] == 4*serial[2][-1]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'checkpoint.npy'
        for dtype,seed in jobs:
            np.save(path,trajectory(dtype,seed,16),allow_pickle=False)
            checksum = digest(path.read_bytes())
            resumed = trajectory(dtype,seed,32,restore(path,checksum))
            assert digest(resumed.tobytes()) == job((dtype,seed))[2]
            path.write_bytes(path.read_bytes()+b'bad')
            try:
                restore(path,checksum)
            except ValueError:
                pass
            else:
                raise AssertionError('Accepted corrupt checkpoint')
    with ProcessPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(job,jobs)) == serial
    report = dict(status='FIXTURES_PASS_FULL_QUALIFICATION_OPEN',
                  source_sha256=digest(Path(__file__).read_bytes()),
                  cycle1='Byte-prefix boundary, balanced reverse-closed distance splits, and local update fixture passed.',
                  cycle2=serial,
                  cycle3='Storage fixture checkpoint/resume equality, corrupt checkpoint rejection and four-worker equality passed.')
    Path(__file__).with_name('ENGINEERING.json').write_text(json.dumps(report,indent=2)+'\n')
