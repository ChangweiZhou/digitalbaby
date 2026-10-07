"""Validate real writes via inherited audit, then independent v2 binding checks."""
import numpy as np
from pathlib import Path
from runtime import frozen_auditor,SCIENTIFIC_ID,make_world,file_sha,Core,policies
from exporter import load_model
def audit(d,identity,folder,*,allow_short=False,load_export=True):
    if d['schema']!='NEXT_CORE_BUDGET_JOB_V2' or d['execution_source_identity']!=identity or d['source_identity']!=SCIENTIFIC_ID:raise ValueError('v2 receipt source')
    if d['stage'] not in ('qualification','screen','confirm') or d['development']!=(d['stage']=='qualification'):raise ValueError('stage/evidence mismatch')
    complete=d['complete']
    if not complete and (d['stage']!='qualification' or not allow_short):raise ValueError('partial receipt cannot be science')
    lens=dict(d,development=True)
    inherited=frozen_auditor.audit_receipt(lens,SCIENTIFIC_ID,complete=complete)
    w=make_world(d['world'],d['assay'])
    names=w['branches'] if complete else ['W']
    if d['limits']['branches']!=names or set(d['branches'])!=set(names):raise ValueError('branch list')
    required=list(w['clocks']) if complete else ['short_end']
    for branch,b in d['branches'].items():
        if complete and [p['name'] for p in b['probes']]!=required:raise ValueError('missing/duplicate phase probes')
        if [r['index'] for r in b['records']]!=list(range(d['limits']['records'])):raise ValueError('record sequence')
        total=sum(r['online_cpu_s'] for r in b['records'])+sum(p['online_api_cpu_s'] for p in b['probes'] if p['name']=='final')
        if abs(total-b['online_cpu_s'])>1e-9:raise ValueError('online CPU accumulation')
        if d['arm']=='ERROR':
            extra=sum(p['Q_HALF_additional_readout_cpu_s'] for p in b['probes'] if p['name']=='final')
            if extra<0 or abs(b['Q_HALF_online_cpu_s']-b['online_cpu_s']-extra)>1e-9:raise ValueError('Q_HALF learning/readout cost')
        for p in b['probes']:
            if complete and p['at']!=w['clocks'][p['name']]:raise ValueError('probe clock')
            for r in p['rows']:
                if d['assay']=='reuse':
                    vs=r['values'];want=ord('L') if vs[0]>=vs[1] else ord('R')
                    if r['emitted']!=want or r['correct']!=int(want==r['target']):raise ValueError('choice output')
                    hs=[]
                    for q in r['predictions']:
                        u=np.array(q['shared'])/1.4911274663291492+.5*np.array(q['private'])/1.3452365735750882
                        if q['extra']:u+=np.array(q['extra'])/1.4911274663291492
                        hs.append(float(u[1]-u[0]))
                    eh=ord('L') if hs[0]>=hs[1] else ord('R')
                    if r['Q_HALF_emitted']!=eh or r['Q_HALF_correct']!=int(eh==r['target']):raise ValueError('Q choice output')
                else:
                    q=r['prediction'];u=np.array(q['shared'])/1.4911274663291492+.5*np.array(q['private'])/1.3452365735750882
                    if q['extra']:u+=np.array(q['extra'])/1.4911274663291492
                    if r['policies']['Q_HALF']['emitted']!=48+int(np.argmax(u)):raise ValueError('Q output route')
    e=d['W_export'];path=Path(folder)/'models'/e['file']
    if e['sha256']!=file_sha(path) or e['state_digest']!=d['branches']['W']['final_state_digest']:raise ValueError('W model/receipt mismatch')
    if e['binding']!={'world':d['world'],'arm':d['arm'],'assay':d['assay'],'stage':d['stage'],'execution_source_identity':identity,'fixture_sha256':w['sha256'],'records':d['limits']['records'],'time':d['branches']['W']['final_time']}:raise ValueError('W export job binding')
    if load_export:load_model(path,e['binding'])
    return dict(inherited,stage=d['stage'],W_export_valid=True)


def audit_unavailable(d,identity,folder):
    from runtime import frozen_auditor,read_receipt
    if d['schema']!='DIAGNOSTIC_UNAVAILABLE_V2' or d['status']!='DIAGNOSTIC_NOT_QUALIFIED' or d['arm']!='S3_RAND' or d['reason']!='legal random swap pool exhausted':raise ValueError('unregistered diagnostic exclusion')
    if d['execution_source_identity']!=identity or d['source_identity']!=SCIENTIFIC_ID or d['stage'] not in ('qualification','screen','confirm'):raise ValueError('diagnostic binding')
    w=make_world(d['world'],d['assay']);cur=d['partial_cursor']
    if d['fixture_sha256']!=w['sha256'] or cur['fixture_sha256']!=w['sha256'] or cur['stage']!=d['stage']:raise ValueError('diagnostic fixture')
    yp=Path(folder)/'receipts'/f"{d['world']}_S3_CUE_{d['assay']}.json.gz"
    if d['yoke_sha256']!=file_sha(yp):raise ValueError('diagnostic yoke')
    audit(read_receipt(yp),identity,folder,allow_short=d['stage']=='qualification',load_export=False)
    if not cur['branches'] or not 0<=cur['branch_index']<len(cur['branch_names']):raise ValueError('diagnostic cursor')
    for branch,b in cur['branches'].items():
        if [r['index'] for r in b['records']]!=list(range(len(b['records']))):raise ValueError('diagnostic prefix')
        for e,r in zip(w['events'],b['records']):frozen_auditor.check_record(r,'S3_RAND',e,branch,8)
    return {'status':'DIAGNOSTIC_NOT_QUALIFIED','mechanism_claim_available':False}
