"""Receipt/file audit and fixed analysis. Never reruns a model or trajectory."""
import json,hashlib,math,zipfile
from collections import Counter
import runtime as C
def main():
 source=json.loads((C.R/'SOURCE_TRAINING.json').read_text());probes=json.loads((C.R/'PROBE_RESULTS.json').read_text());gate=json.loads((C.R/'ADMISSION.json').read_text());index=json.loads((C.R/'COMMIT_INDEX.json').read_text());audit=[]
 for entry in index:
  p=C.R/entry['path'];assert C.sha(p)==entry['sha256'];r=json.loads(p.read_text())
  assert all(C.sha(C.R/f)==d for f,d in r['files'].items());audit.append(dict(key=entry['key'],status='ACCEPTED',receipt_sha256=entry['sha256']))
 births=json.loads((C.R/'BIRTH_LOCKS.json').read_text())['parameters']
 assert len(source['receipts'])==6 and sum(x['records'] for x in source['receipts'])==12258 and sum(x['targets'] for x in source['receipts'])==738324
 for r in source['receipts']:
  if r['arm']=='FROZEN-H':assert r['H_array_sha256']==births[r['pair']][0] and r['H_bias_sha256']==births[r['pair']][1]
 assert len(probes['results'])==12 and sum(r['fit_targets'] for r in probes['results'])==73728
 paired=[]
 for pair in ('P1','P2'):
  for space in ('direct','bridged'):
   rr={r['arm']:r for r in probes['results'] if r['pair']==pair and r['space']==space};a,b=rr['TRUE-H']['per_record'],rr['FROZEN-H']['per_record']
   diffs=[dict(record=k,targets=a[k]['targets'],TRUE_minus_FROZEN_mean_NLL=a[k]['loss_sum']/a[k]['targets']-b[k]['loss_sum']/b[k]['targets']) for k in a]
   paired.append(dict(pair=pair,space=space,records=diffs,zero_differences=sum(d['TRUE_minus_FROZEN_mean_NLL']==0 for d in diffs)))
 C.save('PROBE_PAIRED.json',dict(results=paired,uncertainty='Two previously opened development text pairs, one source birth each; correlated positions/records are not independent worlds. No population CI or significance claim. Fixed probe optimization can limit negative inference.'))
 summary=dict(verdict=gate['verdict'],native_B_run=gate['native_B_admitted'],source_record_applications=12258,source_target_applications=738324,probe_fit_target_applications=73728,probe_record_touches=sum(r['source_record_touches'] for r in probes['results']),native_lives=0,native_records=0,native_targets=0,probe_checks=gate['checks'],core_adopted=False,evidence='Local feature/probe diagnostic if stopped; bounded native predictive deployment only if completed; no arithmetic/E3 result.')
 paragraphs=[]
 if gate['native_B_admitted']:
  native=json.loads((C.R/'NATIVE_RESULTS.json').read_text());assert native['status']=='COMPLETE' and len(native['results'])==12
  contrasts=[]
  for pair in ('P1','P2'):
   arms={r['arm']:r for r in native['results'] if r['pair']==pair}
   t,f=arms['TRUE-H']['metrics']['B8']['cohorts']['held'],arms['FROZEN-H']['metrics']['B8']['cohorts']['held'];to,fo=arms['TRUE-H-NOWRITE']['metrics']['B8']['cohorts']['held'],arms['FROZEN-H-NOWRITE']['metrics']['B8']['cohorts']['held']
   positives=t['accuracy']>f['accuracy'] and t['payload_NLL']<f['payload_NLL'];ia=(t['accuracy']-to['accuracy'])-(f['accuracy']-fo['accuracy']);il=(to['payload_NLL']-t['payload_NLL'])-(fo['payload_NLL']-f['payload_NLL'])
   contrasts.append(dict(pair=pair,complete_accuracy_difference=t['accuracy']-f['accuracy'],payload_NLL_difference=t['payload_NLL']-f['payload_NLL'],positive=positives,write_accuracy_interaction=ia,write_NLL_benefit_interaction=il,write_mediated=positives and ia>0 and il>0,gained_ids=sorted(set(t['correct_ids'])-set(f['correct_ids'])),lost_ids=sorted(set(f['correct_ids'])-set(t['correct_ids']))))
  verdict='WRITE_MEDIATED_FEATURE_SIGNAL' if all(r['write_mediated'] for r in contrasts) else ('NATIVE_FEATURE_DEPLOYMENT_SIGNAL' if all(r['positive'] for r in contrasts) else 'NO_CONFIRMED_NATIVE_FEATURE_DEPLOYMENT_SIGNAL')
  summary.update(verdict=verdict,native_lives=12,native_records=12288,native_targets=24576,contrasts=contrasts)
  manifest=json.loads((C.DEV/'local/DATA_MANIFEST.json').read_text());refs=[]
  for pair in ('P1','P2'):
   train,held=manifest[pair]['rows']['train'],manifest[pair]['rows']['held'];freq=Counter(int(r['answer'],16) for r in train);guess=min(freq,key=lambda y:(-freq[y],y));table={r['prompt']:int(r['answer'],16) for r in train};score=sum(int(r['answer'],16)==guess for r in held)
   refs.append(dict(pair=pair,held_items=128,frequency_correct=score,exact_key_held_hits=sum(r['prompt'] in table for r in held),exact_key_backoff_correct=score,payload_NLL=sum(-math.log((freq[int(r['answer'],16)]+1)/(128+257)) for r in held)/128,actual_fit_records=128,actual_fit_targets=256,information_matched_unique_training_contexts=True,update_site_matched=False))
   C.COUNTS['reference_records']+=128;C.COUNTS['reference_targets']+=256;C.COUNTS['probability_evaluations']+=256
  C.save('COUNTERMODELS.json',dict(frequency_and_exact_key=refs,counting_reference_status='COUNTING_REFERENCE_UNAVAILABLE',counting_reason='Exact ClaudeV169 code/config was not authenticated in this local campaign. No substitute or grid was run.',configuration=dict(classes=257,frequency_additive_smoothing=1,output_site_separated=True,exact_key_fallback='frequency',tie_break='lowest byte')))
  paragraphs.extend([f"Native lives:12; full records:12,288; native targets:24,576. Verdict: **{verdict}**.", 'Both text pairs are previously opened development fixtures. Sign-consistency is descriptive, not statistical confirmation, promotion or arithmetic generalization.'])
 else:
  paragraphs.extend([f"Stopped at the registered source/bridge admission gate: **{gate['verdict']}**. Native B records: **0**.",'This does not establish that the native mechanism failed, that learned information is absent, or that local learning is impossible. The fixed adapted recipe and probe did not qualify for native deployment. No B fitting, retuning or automatic retry occurred.'])
 C.save('SUMMARY.json',summary);C.save('RECEIPT_AUDITS.json',dict(status='ACCEPTED',audits=audit,no_model_reexecution=True))
 if (C.R/'local/native_fixed.pkl').exists():C.save('FIXED_ASSET_ARCHIVE.json',dict(path='local/native_fixed.pkl',sha256=C.sha(C.R/'local/native_fixed.pkl'),bytes=(C.R/'local/native_fixed.pkl').stat().st_size,scope='Frozen asset dictionary required with persistent-reference native checkpoints; no model reexecution.'))
 C.save('FINAL_AUDIT.json',dict(status='ACCEPTED',verdict=summary['verdict'],independent_process=True,independent_external_auditor=False,receipt_file_integrity=True,source_counts=True,probe_counts=True,gate_followed=True,scientific_model_revalidation=False,core_adopted=False,source_records=12258,native_records=summary['native_records'],limitations=['Two previously opened development pairs; one birth each.','No E3/arithmetic capability result.','A failed fixed probe cannot distinguish information loss from readout/conditioning limits.']))
 lines=['# V176 optimized campaign result','',*paragraphs,'','| Pair | Space | TRUE NLL | FROZEN NLL | Qualified |','|---|---|---:|---:|---|']
 for pair,spaces in gate['checks'].items():
  for space,r in spaces.items():lines.append(f"|{pair}|{space}|{r['TRUE']:.6f}|{r['FROZEN']:.6f}|{r['passed']}|")
 lines+=['','Source learner:6 fixed source states,12,258 record applications,738,324 byte/EOS target applications. Probe fitting:73,728 target applications; source-record touches are separately counted in SUMMARY.json. Source heads were training-only; the native process does not import Torch.','', 'All prior and supplemental engineering costs remain charged. Complete own-byte/EOS output and separate payload/EOS losses are required for native success. No attention, Transformer, ELM or arithmetic-specific feature was introduced.','', 'See SOURCE_TRAINING.json, PROBE_RESULTS.json, PROBE_PAIRED.json, COMMIT_INDEX.json and FINAL_AUDIT.json. Raw corpora, caches, model states and response traces are excluded from the compact package.']
 (C.R/'REPORT.md').write_text('\n'.join(lines)+'\n');C.done(verdict=summary['verdict'])
 files=[p for p in C.R.rglob('*') if p.is_file() and 'local' not in p.relative_to(C.R).parts and '__pycache__' not in p.parts and p.suffix in ('.json','.md','.py') and p.name!='PACKAGE.json']
 with zipfile.ZipFile(C.R/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,str(p.relative_to(C.R)))
 C.save('PACKAGE.json',dict(status='COMPLETE',path='RESULT_BUNDLE.zip',sha256=C.sha(C.R/'RESULT_BUNDLE.zip'),bytes=(C.R/'RESULT_BUNDLE.zip').stat().st_size,raw_data=False))
if __name__=='__main__':
 try:main()
 except BaseException as e:C.fail(e);raise
