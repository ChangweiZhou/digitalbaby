"""Exercise the production campaign end to end on synthetic records."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import campaign as run

def main():
    splits = {}
    for role, size in (('A',4),('B',2)):
        splits[role+'_train'] = [b'abc'+bytes([i])*69 for i in range(size)]
        splits[role+'_calib'] = [b'abcd']
        splits[role+'_held'] = [b'abcx', b'abcy']
    splits['NOEDIT_train'] = splits['A_train'][:2]
    splits['EDIT_train'] = [b'abd'+r[3:] for r in splits['NOEDIT_train']]
    meta = {'context_hex':b'ab'.hex(),'old':ord('c'),'new':ord('d'),
            'selected_train_ids':list(range(2))}
    grid = {'bigram_best':[0,2,.75],'grid':[[0,5,.75]]}
    worlds = {'fixture1':(1,2,dict(splits),meta),'fixture2':(3,4,dict(splits),meta)}
    expected = len(worlds)*2*sum(n*sum(len(r)+1 for r in splits[role+'_train'])
                                 for role,n in run.SCHEDULE)
    expected_records=len(worlds)*2*sum(n*len(splits[role+'_train']) for role,n in run.SCHEDULE)
    workdir = Path.cwd()
    with TemporaryDirectory() as tmp:
        run.os.chdir(tmp)
        try:
            run.OUT.mkdir()
            run.write('LAUNCH.json',{'source_sha256':{}})
            Path('local/text').mkdir(parents=True)
            for i in range(1,5):
                Path(f'local/text/{i}.txt').write_bytes(bytes([i]))
            manifest = {w:{'sources':{str(i):{'sha256':run.sha(bytes([i]))} for i in (a,b)},
                           'splits':{k:[run.sha(r) for r in rs] for k,rs in sp.items()}}
                        for w,(a,b,sp,_) in worlds.items()}
            Path('FRESH_ROSTER.json').write_text(json.dumps({'worlds':manifest}))
            Path('BASELINE_PREFLIGHT.json').write_text(json.dumps({'panels':{
                f'{w}_{s}':grid for w in worlds for s in ('A2','B3','EDIT2','NOEDIT2')}}))
            run.select = lambda: (None,worlds,None)
            run.main(expected_targets=expected,expected_records=expected_records)
            status = json.loads((run.OUT/'STATUS.json').read_text())
            assert status['status']=='COMPLETED'
            assert status['counts']['targets']==expected
            assert len(list(run.OUT.glob('*.pt')))==20
        finally:
            run.os.chdir(workdir)
    Path('CAMPAIGN_REHEARSAL.json').write_text(json.dumps({
        'status':'PASS','fixture_records':expected_records,'fixture_targets':expected,
        'science_records':0,'receipt_count':4,'checkpoint_count':20},indent=2)+'\n')

if __name__=='__main__':
    main()
