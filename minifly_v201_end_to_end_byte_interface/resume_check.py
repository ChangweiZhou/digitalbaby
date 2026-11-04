"""Persist the resident state in the middle of a structural frame."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from live import Live, frame, prior, sha

ARRAYS=('mem','imp','maps','deploy','rel','eos','w','bias','projection')

def new():
    prose,_=prior.corpus()
    env,_=prior.fact_schedule(20100,prose)
    projection=np.random.default_rng(20100).normal(size=(2000,48))
    return Live(env['bag'],projection)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def state(model):
    return dict(digest=sha(*(getattr(model,a) for a in ARRAYS)),context=model.context.hex(),
                predictor_history=model.hist.hex(),position=model.parser.state['position'],
                updates=model.updates,output=[x.hex() for x in model.out],loss=model.loss_sum,
                bytes=model.bytes)

def rest(model):
    model.receive(b'ghijklabcd')
    model.receive(frame('R',b'ab>'),4)
    model.receive(frame('Q',b'ab'),probe=True)
    model.parser.finish()
    return state(model)

if __name__=='__main__':
    if len(sys.argv)>1:
        path=Path(sys.argv[1])
        if digest(path.read_bytes())!=path.with_suffix('.sha').read_text():
            raise ValueError('Checkpoint digest mismatch')
        model=new()
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['meta']))
            for a in ARRAYS:
                setattr(model,a,data[a])
        model.parser.state=meta['parser']
        model.context=bytes.fromhex(meta['context'])
        model.hist=bytes.fromhex(meta['history'])
        model.updates=meta['updates']
        model.out=[bytes.fromhex(x) for x in meta['out']]
        model.loss_sum=meta['loss']
        model.bytes=meta['bytes']
        print(json.dumps(rest(model)))
        sys.exit()
    model=new()
    model.receive(b'the code of mnopq is ')
    model.receive(frame('T',b'7'),2)
    model.receive(frame('S',b'abcdefghijklabcd')[:8])
    meta=dict(parser=model.parser.state,context=model.context.hex(),history=model.hist.hex(),
              updates=model.updates,out=[x.hex() for x in model.out],loss=model.loss_sum,bytes=model.bytes)
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'state.npz'
        np.savez(path,meta=json.dumps(meta),**{a:getattr(model,a) for a in ARRAYS})
        path.with_suffix('.sha').write_text(digest(path.read_bytes()))
        expected=rest(model)
        cmd=[sys.executable,'-B',__file__,str(path)]
        assert json.loads(subprocess.check_output(cmd))==expected
        with path.open('ab') as handle:
            handle.write(b'corrupt')
        error=subprocess.run(cmd,capture_output=True,text=True)
        assert error.returncode and 'Checkpoint digest mismatch' in error.stderr
    Path(__file__).with_name('RESUME_CHECK.json').write_text(json.dumps(dict(passed=True,fixture=expected),indent=2)+'\n')
