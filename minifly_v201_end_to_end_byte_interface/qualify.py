"""Byte parser and checkpoint qualification, plus a full development replay."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ESC=30
LENGTHS={ord('S'):16,ord('R'):3,ord('Q'):2,ord('F'):1,ord('T'):1}

class Parser:
    def __init__(self,state=None):
        self.state=state or dict(escaped=False,kind=None,payload=[],position=0)
    def feed(self,byte,teach=False):
        s=self.state
        s['position']+=1
        if s['kind'] is not None:
            kind=s['kind']
            final=len(s['payload'])+1==LENGTHS[kind]
            if teach != (final and kind in (ord('R'),ord('T'))):
                raise ValueError('Invalid teaching marker')
            s['payload'].append(byte)
            if final:
                payload=bytes(s['payload'])
                if kind in (ord('R'),ord('Q')) and not all(97<=b<=108 for b in payload[:2]):
                    raise ValueError('Invalid operands')
                if kind in (ord('S'),) and not all(97<=b<=108 for b in payload):
                    raise ValueError('Invalid structural symbols')
                if kind==ord('R') and byte not in (60,62) or kind==ord('T') and not 48<=byte<=57:
                    raise ValueError('Invalid teaching target')
                s.update(kind=None,payload=[])
                return [chr(kind),payload.hex()]
            return None
        if teach:
            raise ValueError('Unrouted teaching marker')
        if s['escaped']:
            s['escaped']=False
            if byte==ESC:
                return ['P','1e']
            if byte not in LENGTHS:
                raise ValueError('Unknown frame')
            s['kind']=byte
        elif byte==ESC:
            s['escaped']=True
        else:
            return ['P',bytes([byte]).hex()]
        return None
    def finish(self):
        if self.state['escaped'] or self.state['kind'] is not None:
            raise ValueError('Truncated frame')

def frame(kind,payload):
    raw=bytes([ESC])+kind.encode()+payload
    return [(b,kind in ('T','R') and i==len(raw)-1) for i,b in enumerate(raw)]

def consume(parser, items):
    return [event for byte,marker in items if (event:=parser.feed(byte,marker)) is not None]

def fixture(seed):
    prose=[(b,False) for b in b'prose\x1e\x1eR ab>']
    tape=prose+frame('S',b'abcdefghijkl ab'.replace(b' ',b'')+b'ab')+frame('Q',b'ab')
    tape+=frame('R',b'ab>')+frame('T',str(seed%10).encode())
    return tape

def case(seed):
    tape=fixture(seed)
    p=Parser()
    expected=consume(p,tape)
    p.finish()
    for split in range(len(tape)+1):
        p=Parser()
        assert consume(p,tape[:split])+consume(p,tape[split:])==expected
    return expected

def digest(data): return hashlib.sha256(data).hexdigest()

def development():
    # Feed the real development world's relation wire through the new parser,
    # then hand the parsed records to the existing model replay.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'minifly-integration'))
    import run
    import numpy as np
    prose,_=run.corpus()
    chunks,_,_,fact,reference=run.assemble(20100,prose)
    parser=Parser()
    reconstructed=[]
    count=0
    for kind,wire,tags,meta in chunks:
        if kind in ('S','Q','R'):
            payload=wire[2:-1] if kind=='S' else wire[5:7]+(wire[8:9] if kind=='R' else b'')
            events=consume(parser,frame(kind,payload))
            assert len(events)==1
            parsed,hexdata=events[0]
            data=bytes.fromhex(hexdata)
            count+=1
            wire=b'\x1eS'+data+b'\n' if parsed=='S' else b'\x1epar '+data[:2]+b'='+data[2:]+b'\n'
            reconstructed.append((parsed,wire,tags,data))
        else:
            reconstructed.append((kind,wire,tags,meta))
    parser.finish()
    integrated=run.replay(reconstructed,fact,('fact','relation'))
    assert integrated['relation']['state']==reference['final']
    assert integrated['relation']['scores']==reference['scores']
    prior=json.loads((run.ROOT/'world-20100.json').read_text())
    assert integrated==prior['integrated']
    return dict(parsed_relation_frames=count,fact=integrated['fact']['kept'],
                relation=integrated['relation']['scores']['change_128'])

if __name__=='__main__':
    if len(sys.argv)>1:
        path=Path(sys.argv[1])
        raw=path.read_bytes()
        if digest(raw)!=path.with_suffix('.sha').read_text():
            raise ValueError('Checkpoint digest mismatch')
        state,tail=json.loads(raw)
        print(json.dumps(consume(Parser(state),tail)))
        sys.exit()
    for bad in ([(ESC,False),(255,False)],[(ESC,True)],frame('Q',b'a!'),
                frame('R',b'ab>')[:-1]):
        try:
            p=Parser()
            consume(p,bad)
            p.finish()
        except ValueError:
            continue
        raise AssertionError('Malformed input accepted')
    serial=[case(s) for s in (101,102)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(case,(101,102)))==serial
    dev=development()
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'checkpoint.json'
        tape=fixture(101)
        for cut in (len(tape)-7,len(tape)-3):
            p=Parser()
            prefix=consume(p,tape[:cut])
            raw=json.dumps([p.state,tape[cut:]]).encode()
            path.write_bytes(raw)
            path.with_suffix('.sha').write_text(digest(raw))
            cmd=[sys.executable,'-B',__file__,str(path)]
            assert prefix+json.loads(subprocess.check_output(cmd))==serial[0]
            path.write_bytes(raw+b' ')
            proc=subprocess.run(cmd,capture_output=True,text=True)
            assert proc.returncode and 'Checkpoint digest mismatch' in proc.stderr
    report=dict(status='PARTIAL_ENGINEERING',source_sha256=digest(Path(__file__).read_bytes()),
                cycle1='Byte framing, malformed records, teaching marker and fragmentation checks',
                cycle2=dev,
                cycle3='Parser fixture fresh-process resume, corrupt checkpoint rejection, serial/parallel agreement')
    Path(__file__).with_name('ENGINEERING.json').write_text(json.dumps(report,indent=2)+'\n')
