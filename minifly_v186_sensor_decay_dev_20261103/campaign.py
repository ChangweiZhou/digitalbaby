"""Run the locked V186 factorial once, retaining checkpoints and paired scores."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import zipfile

from qualify import DecaySensor, Learner, PARENT, T, np
from lock_roster import select, sha, corrected
from launch_checks import PrefixControl
from profile_roster import load_definitions
from generation import generate

sys.path.insert(0,str(PARENT/'official_ops'))
from metrics import score

OUT = Path('campaign')
START = time.monotonic()
CARRIED = 120
SCHEDULE = [('A',2),('B',3),('EDIT',2),('NOEDIT',2)]
counts = {'records':0,'targets':0,'completed_lives':0}

def write(name, data):
    path=OUT/name
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def resources():
    return {'cpu_seconds':time.process_time()+CARRIED,'wall_seconds':time.monotonic()-START+CARRIED,
            'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'output_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())}

def check():
    r=resources()
    assert all(r[k]<=v for k,v in dict(cpu_seconds=900,wall_seconds=1800,rss_bytes=2**30,output_bytes=2**28).items()),r
    assert counts['records']+314<=5000 and counts['targets']+22594<=600000
    return r

def pack(sensor,records,decay):
    return [[np.flatnonzero(row).astype(np.uint16) for row in sensor.codes(r,decay)] for r in records]

def step_pass(net, records, codes, batch, ids):
    x=T.zeros((batch,5177))
    labels,keys=[],[]
    for r, rows, key in zip(records,codes,ids):
        for y,row in zip([*r,256],rows):
            x[len(labels),row.astype(np.int64)]=1
            labels.append(y)
            keys.append(str(key))
            if len(labels)==batch:
                net.step(x,T.tensor(labels),keys,'NORMAL')
                labels.clear()
                keys.clear()
                x.zero_()
        counts['records']+=1
        counts['targets']+=len(r)+1
    if labels:
        net.step(x[:len(labels)],T.tensor(labels),keys,'NORMAL')
    check()

def compact_generation(net,sensor,records,decay):
    result=dict(prompts=len(records),exact=0,own_EOS=0,premature_EOS=0,truncated=0)
    for r in records:
        prefix=r[:3]
        def next_class(code):
            return int(net.forward(T.from_numpy(code[None,:]))[0].argmax())
        payload,eos=generate(next_class,sensor,prefix,decay)
        result['exact']+=int(eos and prefix+payload==r)
        result['own_EOS']+=int(eos)
        result['premature_EOS']+=int(eos and len(prefix+payload)<len(r))
        result['truncated']+=int(not eos)
    return result

def baseline_panels(splits, world, calibrated):
    env=load_definitions()
    result={}
    phases={'A2':['A'],'B3':['A','B'],'EDIT2':['A','B','EDIT'],'NOEDIT2':['A','B','NOEDIT']}
    for stage,roles in phases.items():
        history=[r for phase,n in SCHEDULE if phase in roles for _ in range(n) for r in splits[phase+'_train']]
        options=calibrated[world+'_'+stage]
        _,_,bigram=options['bigram_best']
        _,_,discount=min(row for row in options['grid'] if row[1]==5)
        uni=env['Unigram']().fit(history)
        bi=env['IKN'](2,bigram).fit(history)
        kn=env['IKN'](5,discount).fit(history)
        def probability(prefix,target):
            return uni.prob((),target)
        prefix=PrefixControl(probability).fit(history)
        result[stage]={}
        for name in ('A_held','B_held','EDIT_held'):
            records=splits[name]
            n=sum(len(r)+1 for r in records)
            panel={k:float(env['line_bits'](m,records).sum()/n) for k,m in [('unigram',uni),('bigram',bi),('KN5',kn)]}
            panel['exact_prefix']=sum(-math.log2(prefix.probability(r[:i],y)) for r in records for i,y in enumerate([*r,256]))/n
            result[stage][name]=panel
        check()
    return result

def suffix_panels(records, bits, history):
    result={}
    for width in range(1,6):
        seen={r[i-width:i] for r in history for i in range(width,len(r)+1)}
        mask=np.array([i>=width and r[i-width:i] in seen for r in records for i in range(len(r)+1)])
        result[str(width)]={name:{'targets':int(m.sum()),'bits':float(bits[m].mean()) if m.any() else None}
                            for name,m in [('seen',mask),('unseen_or_short',~mask)]}
    return result

def geometry(sensor, records, decay):
    overlaps={}
    for width in (1,2,4):
        preserve, change=[],[]
        for record in records:
            if len(record)<width+2:
                continue
            old=sensor.codes(record,decay)[-1]
            distant=bytes([record[0]^1])+record[1:]
            near=record[:-1]+bytes([record[-1]^1])
            preserve.append(float(np.dot(old,sensor.codes(distant,decay)[-1]))/max(1,int(old.sum())))
            change.append(float(np.dot(old,sensor.codes(near,decay)[-1]))/max(1,int(old.sum())))
        overlaps[str(width)]={'pairs':len(preserve),'same_suffix_overlap':float(np.mean(preserve)) if preserve else None,
                              'changed_suffix_overlap':float(np.mean(change)) if change else None}
    rows=[np.packbits(row != 0).tobytes() for r in records for row in sensor.codes(r,decay)[1:]]
    return {'overlaps':overlaps,'nonempty_prefix_sites':len(rows),
            'repeated_code_fraction':1-len(set(rows))/len(rows) if rows else None}

def main(expected_targets=253084, expected_records=4480):
    launch=json.loads((OUT/'LAUNCH.json').read_text())
    for name,digest in launch['source_sha256'].items():
        assert sha(Path(name).read_bytes())==digest,name
    _,worlds,_=select()
    locked=json.loads(Path('FRESH_ROSTER.json').read_text())
    calibrated=json.loads(Path('BASELINE_PREFLIGHT.json').read_text())['panels']
    sensor=DecaySensor()
    results,baselines={},{}
    for world,(a,b,splits,meta) in worlds.items():
        for i in (a,b):
            assert sha(Path(f'local/text/{i}.txt').read_bytes())==locked['worlds'][world]['sources'][str(i)]['sha256']
        assert {k:[sha(r) for r in rs] for k,rs in splits.items()}==locked['worlds'][world]['splits']
        rule=bytes.fromhex(meta['context_hex']),meta['old'],meta['new']
        for role in ('calib','held'):
            splits['EDIT_'+role]=[corrected(r,rule) for r in splits['A_'+role]]
        baselines[world]=baseline_panels(splits,world,calibrated)
        write('BASELINES.json',baselines)
        results[world]={}
        for decay in (.95,.5):
            cached={k:pack(sensor,rs,decay) for k,rs in splits.items()}
            write(f'{world}_{decay:.2f}_GEOMETRY.json',geometry(sensor,splits['A_held'],decay))
            for batch in (256,32):
                name=f'{decay:.2f}_{batch}'
                net=Learner(world)
                start_counts=counts.copy()
                panels={}
                checkpoint_index={}
                def snapshot(stage):
                    filename=f'{world}_{name}_{stage}.pt'
                    T.save({'decay':decay,'batch':batch,'state':net.state()},OUT/filename)
                    assert (OUT/filename).stat().st_size<=2**20
                    checkpoint_index[stage]=sha((OUT/filename).read_bytes())
                    if stage not in ('A2','B3','EDIT2','NOEDIT2'):
                        return
                    result={}
                    for panel in ('A_held','B_held','EDIT_held'):
                        alternate=splits['A_held'] if panel=='EDIT_held' else None
                        summary,site=score(net,splits[panel],cached[panel],alternate)
                        if alternate is not None:
                            old=np.array([v for r in alternate for v in [*r,256]])
                            changed=site['y']!=old
                            summary['changed_log_odds']=float(site['new_old_margin'][changed].mean())
                            summary['stable_bits']=float(site['bits'][~changed].mean())
                        history_roles={'A2':('A',),'B3':('A','B'),'EDIT2':('A','B','EDIT'),'NOEDIT2':('A','B','NOEDIT')}[stage]
                        history=[r for role in history_roles for r in splits[role+'_train']]
                        summary['suffix_seen']=suffix_panels(splits[panel],site['bits'],history)
                        result[panel]=summary
                    if stage!='NOEDIT2':
                        result['generation']=compact_generation(net,sensor,splits['A_held'][:16]+splits['B_held'][:16],decay)
                    panels[stage]=result
                snapshot('birth')
                fork=None
                for phase,passes in SCHEDULE:
                    if phase=='NOEDIT':
                        net.restore(fork)
                    for n in range(1,passes+1):
                        stage=phase+str(n)
                        write('STATUS.json',{'status':'RUNNING','world':world,'arm':name,'stage':stage,'counts':counts,**check()})
                        key=phase+'_train'
                        ids=meta['selected_train_ids'] if phase in ('EDIT','NOEDIT') else range(len(splits[key]))
                        step_pass(net,splits[key],cached[key],batch,ids)
                        snapshot(stage)
                    if phase=='B':
                        fork=copy.deepcopy(net.state())
                results[world][name]=panels
                counts['completed_lives']+=1
                write(f'{world}_{name}_RECEIPT.json',{'records':counts['records']-start_counts['records'],'targets':counts['targets']-start_counts['targets'],'checkpoint_sha256':checkpoint_index})
                write('METRICS.json',results)
    assert counts['records']==expected_records and counts['targets']==expected_targets and counts['completed_lives']==8
    decision={}
    for world,arms in results.items():
        long,short=arms['0.95_32'],arms['0.50_32']
        a,b=long['A2']['A_held'],short['A2']['A_held']
        sums=np.array(a['record_bit_sums'])-np.array(b['record_bit_sums'])
        n=np.array(a['record_target_counts'])
        draws=np.random.default_rng(186).integers(0,len(n),size=(10000,len(n)))
        gain=a['bits']-b['bits']
        gain256=arms['0.95_256']['A2']['A_held']['bits']-arms['0.50_256']['A2']['A_held']['bits']
        refs=baselines[world]['A2']['A_held']
        guards={'positive_gain':gain>0,'beats_unigram_and_bigram':b['bits']<min(refs['unigram'],refs['bigram']),
                'B3_A_and_B':all(short['B3'][k]['bits']<long['B3'][k]['bits'] for k in ('A_held','B_held')),
                'EDIT_stable':short['EDIT2']['EDIT_held']['stable_bits']<long['EDIT2']['EDIT_held']['stable_bits'],
                'correction':short['EDIT2']['EDIT_held']['changed_log_odds']>short['NOEDIT2']['EDIT_held']['changed_log_odds']}
        decision[world]={'gain32':gain,'gain256':gain256,'interaction':gain-gain256,
            'CI95_gain32':np.quantile(sums[draws].sum(1)/n[draws].sum(1),[.025,.975]).tolist(),
            'cadence_gain_at_long_decay':arms['0.95_256']['A2']['A_held']['bits']-a['bits'],
            'short_A_forgetting':short['B3']['A_held']['bits']-b['bits'],
            'short_A_KN5_gap':b['bits']-refs['KN5'],'guards':guards}
    passed=all(all(d['guards'].values()) for d in decision.values())
    verdict='REVIEW_ON_NEW_WORLDS' if passed else 'DID_NOT_PASS_PROGRESSION_GATES'
    write('DECISION.json',{'verdict':verdict,'worlds':decision,'counts':counts})
    (OUT/'REPORT.md').write_text('# V186 result\n\n'+verdict+'\n\nSee DECISION.json for paired contrasts and METRICS.json for native generation.\n')
    with zipfile.ZipFile(OUT/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(OUT.iterdir()):
            if file.suffix in ('.json','.md') and file.name!='STATUS.json':
                archive.write(file,file.name)
    for receipt in OUT.glob('*_RECEIPT.json'):
        data=json.loads(receipt.read_text())
        stem=receipt.name.removesuffix('_RECEIPT.json')
        for stage,digest in data['checkpoint_sha256'].items():
            assert sha((OUT/f'{stem}_{stage}.pt').read_bytes())==digest
    assert len(list(OUT.glob('*_RECEIPT.json')))==8
    with zipfile.ZipFile(OUT/'RESULT_BUNDLE.zip') as archive:
        assert archive.testzip() is None
    write('STATUS.json',{'status':'COMPLETED','verdict':verdict,'counts':counts,**check()})

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        write('STATUS.json',{'status':'FAILED','error':repr(error),'counts':counts})
        raise
