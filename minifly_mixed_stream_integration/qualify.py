"""Routing and isolation fixtures for the mixed-stream design."""
import hashlib
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CHANNELS = ('predictor', 'fact', 'relation')

def stream(seed):
    # Framing is explicit. Hex payload prevents prose from impersonating records.
    for block in range(32):
        yield 'P', b'par ab= >\nF: fake record'
        yield 'F', bytes([97+block%8, 48+(block+seed)%10])
        yield 'S', bytes((97+(block+i)%12 for i in range(16)))
        yield ('R', b'ab>') if block < 16 else ('Q', b'ab')

def wire(kind, payload):
    return kind.encode()+b':'+payload.hex().encode()+b'\n'

def decode(raw):
    tag, payload = raw.rstrip(b'\n').split(b':',1)
    kind=tag.decode()
    data=bytes.fromhex(payload.decode())
    lengths={'F':2,'S':16,'R':3,'Q':2}
    if kind!='P' and (kind not in lengths or len(data)!=lengths[kind]):
        raise ValueError('Invalid frame')
    return kind,data

def job(seed, active=CHANNELS, start=0, end=128, state=None):
    if state is None:
        state={name:[] for name in active}
    for index,(tag,payload) in enumerate(stream(seed)):
        if index<start or index>=end:
            continue
        raw=wire(tag,payload)
        kind,data=decode(raw)
        if 'predictor' in active:
            state['predictor'].extend(raw)
        if 'fact' in active and kind=='F':
            state['fact'].append(tuple(data))
        if 'relation' in active and kind=='S':
            state['relation'].extend(('transition',a,b) for a,b in zip(data,data[1:]))
        if 'relation' in active and kind in ('R','Q'):
            state['relation'].append(('probe',*data[:2]))
            if kind=='R':
                state['relation'].append(('feedback',data[-1]))
    return state

def digest(data):
    return hashlib.sha256(data).hexdigest()

def restore(path):
    data=path.read_bytes()
    if digest(data)!=path.with_suffix('.sha').read_text():
        raise ValueError('Checkpoint digest mismatch')
    return json.loads(data)

if __name__=='__main__':
    if len(sys.argv)>1:
        path=Path(sys.argv[1])
        state=restore(path)
        # Compare using the same JSON representation across processes.
        print(json.dumps(job(101,start=64,state=state)))
        sys.exit()
    try:
        decode(wire('Q',b'ab>'))
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted a query carrying an answer')
    assert decode(wire('P',b'F:12\nR:ab>'))==('P',b'F:12\nR:ab>')
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel=list(pool.map(job,(101,102)))
    serial=[job(seed) for seed in (101,102)]
    assert parallel==serial
    for seed,merged in zip((101,102),serial):
        for name in CHANNELS:
            assert job(seed,(name,))[name]==merged[name]
        events=merged['relation']
        assert sum(e[0]=='feedback' for e in events)==16
        assert sum(e[0]=='transition' for e in events)==32*15
        tail=job(seed,start=64)['relation']
        assert sum(e[0]=='probe' for e in tail)==16
        assert not any(e[0]=='feedback' for e in tail)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'state.json'
        data=json.dumps(job(101,end=64)).encode()
        path.write_bytes(data)
        path.with_suffix('.sha').write_text(digest(data))
        cmd=[sys.executable,'-B',__file__,str(path)]
        assert json.loads(subprocess.check_output(cmd))==json.loads(json.dumps(serial[0]))
        path.write_bytes(data+b' ')
        proc=subprocess.run(cmd,capture_output=True,text=True)
        assert proc.returncode and 'Checkpoint digest mismatch' in proc.stderr
    report=dict(status='ROUTING_FIXTURES_PASS', source_sha256=digest(Path(__file__).read_bytes()),
                cycle_1='Explicit framing round-trip and answer-bearing query rejection',
                cycle_2='Two complete fixture worlds, identical routed event lists by channel, zero late relation labels',
                cycle_3='Parallel equality, fresh-process event-state resume and corrupt checkpoint rejection')
    Path(__file__).with_name('ENGINEERING.json').write_text(json.dumps(report,indent=2)+'\n')
