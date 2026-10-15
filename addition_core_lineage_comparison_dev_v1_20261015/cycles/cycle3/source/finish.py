"""Final TECHNICAL audit/report/package. Never selects/adopts or launches science."""
import json
import statistics
import time
import zipfile
from pathlib import Path
import core
import provenance
import analysis
import data
from audit import audit_job
from io_utils import write,read,now,digest


def finish():
    root=core.bootstrap.HERE
    provenance.verify(3)
    c1=read(root/'cycles/cycle1/TEST_RESULTS.json')
    c2=read(root/'cycles/cycle2/SUMMARY.json')
    c3=read(root/'cycles/cycle3/SUMMARY.json')
    assert c1['accepted'] and c2['accepted'] and c3['accepted']
    assert (c2['jobs'],c2['records'])==(5,2400)
    assert (c3['jobs'],c3['records'])==(10,522240)
    assert read(root/'cycles/cycle2/CONTINUITY_QUALIFICATION.json')['accepted']
    assert read(root/'cycles/cycle2/TAMPER_SUITE.json')['accepted']
    assert read(root/'cycles/cycle2/STATISTICS_FIXTURES.json')['accepted']
    assert read(root/'cycles/cycle2/SELECTION_FIXTURE.json')['accepted']
    assert read(root/'cycles/cycle2/LONG_CLOCK_FIXTURE.json')['accepted']
    jobs=[read(p) for p in sorted((root/'cycles/cycle3/receipts').glob('*.json.gz'))]
    cstar=c3['Cstar_training_CPU']
    assert cstar>0
    profiles=[]
    paths=[]
    for job in jobs:
        audit_job(job)
        byhistory={name:analysis.metrics(h,cstar) for name,h in job['histories'].items()}
        groups=data.prepare(job['world'],job['old_n'],job['new_n'])['groups']
        shortcut=analysis.shortcut_envelope(job['histories']['W']['rows'],groups['held'])
        profiles.append(dict(world=job['world'],version=job['version'],metrics=byhistory,
                             shortcut_envelope=shortcut['envelope'],shortcut_details=shortcut))
        if '151' in job['version']:
            h=job['histories']['W']
            paths.append(dict(world=job['world'],version=job['version'],
                final={group:analysis.accuracy(rows) for group,rows in h['metrics'][-1]['scores'].items()},
                erased={group:analysis.accuracy(rows) for group,rows in h['erasure'].items()},
                no_old_write={group:analysis.accuracy(rows) for group,rows in job['histories']['N_OLD']['metrics'][-1]['scores'].items()},
                output_path='Source Full151 association values, fixed common C16 code; no predictor matrix'))
    write(root/'OPERATIVE_PATH_AUDIT.json',dict(accepted=True,science=False,paths=paths,
           scope='Native output path and native erasure/no-old-write measurements; recurrence is an independent weight-learning route.'))
    resource_rows=[]
    for j in jobs:
        resource_rows.append(dict(world=j['world'],version=j['version'],wall_seconds=j['elapsed'],
            training_cpu=sum(h['resource']['production_cpu'] for h in j['histories'].values()),
            query_cpu=sum(h['query_cpu'] for h in j['histories'].values()),
            erasure_cpu=sum(h['erasure_cpu'] for h in j['histories'].values()),checkpoint_cpu=j['checkpoint_cpu'],
            peak_rss_bytes=j['peak_rss_bytes'],footprint=j['histories']['W']['footprint']))
    costs={v:statistics.median(r['wall_seconds'] for r in resource_rows if r['version']==v) for v in core.VERSIONS}
    worker_times=[0.,0.]
    for _ in range(16):
        for v in core.VERSIONS:
            index=min(range(2),key=lambda i:worker_times[i]);worker_times[index]+=costs[v]
    write(root/'RESOURCE_REPORT.json',dict(utc=now(),complete_technical_jobs=resource_rows,
           conditional_science_two_worker_wall_estimate_seconds=max(worker_times),
           estimated_8h_admission_windows_25pct_margin=__import__('math').ceil(max(worker_times)*1.25/28800),
           caveat='Estimate only; actual whole-job admission must respect measured maximum plus audit margin and the ten-hour hard boundary. Science is not authorized.',
           native_event_accounting='native_events counters are logical state events; centered target events execute three source evaluations, rest/query operations are separately timed. Native update and copying costs are included in production CPU.',
           receipt_bytes=sum(p.stat().st_size for p in (root/'cycles/cycle3/receipts').glob('*'))))
    summary=dict(utc=now(),science=False,accepted=True,cycles_completed=3,
          technical_world_version_jobs=15,correlated_histories=45,technical_job_record_applications=524640,
          development_fixture_record_applications=66,total_training_record_applications=524706,
          direct_native_source_event_fixtures=96,formal_science_jobs=0,
          Cstar_training_CPU=cstar,profiles=profiles,
          verdict='THREE_TECHNICAL_CYCLES_COMPLETE_NO_CORE_ADOPTION_NO_SCIENCE',
          caveat='Two complete technical worlds are descriptive only. No scientific comparison, significance or family exclusion is inferred.')
    write(root/'results/SUMMARY.json',summary)
    lines=['# Addition lineage comparison — three technical cycles','',
       '**Technical qualification only; no formal experiment or new-core adoption.**','',
       'Goal: exact own-byte unseen addition generalization per sample and compute. Only addition is graded; no arithmetic rules, attention, Transformers or ELM are supplied.', '',
       'N151/C151/RC151 retain native Full151 dynamics with explicit C16 output adaptation. KEEP/G62 are separately identified recurrent references. Fixed FE0/CONTENT are historical controls, not learned representations.', '',
       '## Counts', '',
       '15 complete technical world-version jobs; 45 correlated histories; 524,640 record applications in those jobs. Development fixtures add 66 record applications and 96 direct native source-event fixtures. Formal science: zero.', '',
       '## Complete-dose descriptive results', '',
       '| Version | OLD taught, after rest | Unseen after OLD+rest | Unseen final | J_sample | J_compute |',
       '|---|---:|---:|---:|---:|---:|']
    for version in core.VERSIONS:
        subset=[j for j in jobs if j['version']==version]
        means=[]
        for group,stage in [('old','OLD_REST'),('held','OLD_REST'),('held','NEW_REST')]:
            means.append(statistics.mean(analysis.accuracy(next(m for m in j['histories']['W']['metrics'] if m['stage']==stage)['scores'][group]) for j in subset))
        ms=[p['metrics']['W'] for p in profiles if p['version']==version]
        lines.append(f"| {version} | {means[0]:.1%} | {means[1]:.1%} | {means[2]:.1%} | {statistics.mean(m['J_sample'] for m in ms):.4f} | {statistics.mean(m['J_compute'] for m in ms):.4f} |")
    lines.extend(['',f'Common pre-science Cstar: {cstar:.3f} training CPU seconds, median of two KEEP technical W histories through OLD.', '',
        'Checkpoint curves, W/N_OLD/SH_OLD, own raw outputs and native erasure controls are retained in per-cycle receipts. These scores cannot establish significance or justify dropping a family.', '',
        '## Audit and revisions', '',
        'Cycle 1 corrected the no-write auditor’s gamma-vs-alpha decay constants without changing native equations, and filtered synthetic module filenames during source discovery. Cycle 2 checked full byte streams, branch permissions, checkpoint coverage, tamper rejection and save/restore continuity. Cycle 3 retained the same scientific models and doses; operations add checkpoint preservation, separated resource accounting and conservative window admission.', '',
        'A pass means this registered comparison can be executed and audited. Whether any version learns useful addition rules requires a separately authorized formal run. Poor absolute results remain visible; qualification is not substituted for the research objective.', '',
        '## Resource and operational limits', '',
        'See RESOURCE_REPORT.json for full-job timings, peak process RSS, model-state accounting, receipt size and window estimate. All repeated exposures, independent histories and fixture costs are charged. Checkpoint/query/audit costs are reported separately from production training CPU.', '',
        'No automatic science launch, GitHub upload or adopted-core registry change is permitted.'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (root/'cycles/cycle3/REVIEW.md').write_text('# Cycle 3 revision\n\nThe fixed two-world complete-dose qualification and independent receipt audits finished. Resource estimates and Cstar are now measured; sources and doses were not tuned to these scores. The revision is documentation of the actual feasible execution contract and evidence limits, not an additional model. No scientific run or adoption is authorized.\n')
    audit=dict(utc=now(),accepted=True,science=False,cycles=3,jobs=15,histories=45,
               record_applications=524640,formal_science_jobs=0,source_lock_verified=True,
               prerequisite_tests='cycle1 + continuity + statistics + tamper',automatic_science=False)
    write(root/'results/FINAL_AUDIT.json',audit)
    final_status=dict(utc=now(),state='THREE_CYCLES_COMPLETE',science=False,
          jobs=15,histories=45,records=524640,fixture_records=66,formal_science_jobs=0)
    package=root/'results/RESULT_BUNDLE.zip'
    temporary=package.with_name(package.name+'.pending')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(part in ('scratch','__pycache__','checkpoints') for part in p.parts):
                continue
            if p==package or p==root/'results/STATUS.json' or p.name.endswith('.pending') or p.name in ('PACKAGE.json',):
                continue
            z.write(p,p.relative_to(root))
        z.writestr('results/STATUS.json',json.dumps(final_status,indent=2)+'\n')
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:raise AssertionError('result archive CRC failure')
    temporary.replace(package)
    write(root/'results/PACKAGE.json',dict(complete=True,utc=now(),file=str(package),sha256=digest(package),
          bytes=package.stat().st_size,excluded='scratch caches and bulky in-flight checkpoints; all accepted receipts and source snapshots included'))
    write(root/'results/STATUS.json',final_status)
    return summary


if __name__=='__main__':
    print(json.dumps(finish(),indent=2),flush=True)
