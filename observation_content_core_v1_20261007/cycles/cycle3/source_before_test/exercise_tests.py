import argparse,copy,gzip,hashlib,json
from pathlib import Path
import numpy as np
import config,environment,bench,checks,checkpoint_content
from candidate import LocalContentCore

def checkpoint_and_tamper(directory):
    results=[];raw=bench.fixture(822101)[1]['old'][:171]
    for name,constructor,cp in [('A',environment.native_core.ObservationCore,environment.native_checkpoint),('B',LocalContentCore,checkpoint_content)]:
        a=constructor();b=constructor()
        bench.consume(a,raw);bench.consume(b,raw[:101])
        at=b.t+config.DT;b.predict(at);path=directory/(name+'_pending.npz');cp.save(b,path);c=cp.load(path)
        assert c.state_digest()==b.state_digest()
        c.observe(raw[101],at);bench.consume(c,raw[102:]);assert c.state_digest()==a.state_digest()
        results.append(name+' checkpoint pending prediction exactly equals uninterrupted final content')
    # Actual B no-write operation may decay accessed content, but must not alter its learned visit count or write it.
    b=LocalContentCore();bench.consume(b,raw);b.plastic=False
    before=b.visits.copy();at=b.t+config.DT;b.predict(at);ids=b.cached['ids'].copy();f,s=b._view(ids,at)
    row=b.observe(49,at)
    assert not any(row['native_write_l1']) and np.array_equal(b.visits,before)
    assert np.array_equal(b.fast[ids],f) and np.array_equal(b.slow[ids],s)
    results.append('B disabled write count and arrays agree with direct lazy decay, not a flag')
    # Independent finite count table is a declared E1 countermodel, never used by A or B.
    h=b'';table={}
    for byte in raw:
        table.setdefault(h,np.zeros(4,dtype=int))[config.ALPHABET.index(byte)]+=1;h=(h+bytes([byte]))[-4:]
    assert len(table)<=256
    results.append('finite exact-context countermodel; E3 not claimed')
    original=directory/'B_pending.npz'
    bad=directory/'bad_bytes.npz';content=bytearray(original.read_bytes());content[-20]^=1;bad.write_bytes(content)
    bad.with_suffix('.sha256').write_text(original.with_suffix('.sha256').read_text())
    checks.rejects(lambda:checkpoint_content.load(bad))
    for name in ('source','pending','count','row_time'):
        with np.load(original,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        meta=json.loads(arrays['metadata'].tobytes())
        if name=='source':meta['source']='0'*64
        elif name=='pending':arrays['cached_p'][0]+=.01
        elif name=='count':arrays['visits'][0]=65536
        else:arrays['last'][0]=meta['t']+1
        arrays['metadata']=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype=np.uint8)
        bad=directory/(name+'_rehashed.npz')
        with bad.open('wb') as f:np.savez_compressed(f,**arrays)
        bad.with_suffix('.sha256').write_text(hashlib.sha256(bad.read_bytes()).hexdigest()+'\n')
        checks.rejects(lambda:checkpoint_content.load(bad))
    results.append('5 checkpoint tamper cases rejected, including forged integrity with wrong source/pending/count/time')
    return results

def receipt_tamper(r):
    mutations={
      'disabled_old_write':lambda b:b['arms']['A']['training']['old']['N_OLD'][0].update(plastic=True),
      'disabled_revision_write':lambda b:b['arms']['B']['training']['revised']['N_REV'][0].update(native_write_l1=[1.,0.,0.,0.]),
      'future_address':lambda b:b['arms']['B']['training']['old']['W'][0].update(context='3030'),
      'wrong_target':lambda b:b['arms']['A']['training']['old']['W'][0].update(byte=55),
      'nan_error':lambda b:b['arms']['B']['training']['new']['W'][0].update(signs=[float('nan')]*4),
      'drop_row':lambda b:b['arms']['A']['training']['new']['W'].pop(),
      'probe_learns':lambda b:b['arms']['B']['probes']['day3']['W']['rows'][0].update(plastic=True),
      'false_summary':lambda b:b['arms']['A']['probes']['day3']['W']['unchanged_metrics'].update(focal_accuracy=.123),
      'model_wrong_output':lambda b:b['arms']['A']['probes']['day3']['W']['rows'][0].update(emitted=99),
      'clock_shift':lambda b:b['arms']['B']['states']['day3/W'].update(clock=0),
      'wrong_birth':lambda b:b['arms']['A']['birth']['W'][0].update(canonical_B_sha256='0'*64),
      'science_namespace':lambda b:b.update(world=823001),
      'forged_raw':lambda b:b['raw'].update(old='00'+b['raw']['old'][2:]),
      'forged_mapping':lambda b:b['inputs']['old'].update({next(iter(b['inputs']['old'])):99}),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(r);mutate(bad);checks.rejects(lambda:checks.audit(bad))
    return list(mutations)

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--receipt',type=Path)
    args=p.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    result=dict(verdict='PASS',basic=checks.basic_tests(),checkpoint_checks=checkpoint_and_tamper(args.directory),science_worlds=0)
    if args.receipt:
        with gzip.open(args.receipt,'rt') as f:r=json.load(f)
        result['ledger_audit']=checks.audit(r);result['rejected_tampers']=receipt_tamper(r)
    (args.directory/'TEST_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
