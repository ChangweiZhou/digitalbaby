"""Post-execution descriptive reduction of saved responses only. No model queries."""
from pathlib import Path
import json,time,resource,hashlib
import numpy as np
H=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def panel(tag):
 with np.load(H/'local'/f'{tag}.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
def summary(a):
 off=a['offsets'];y=a['y'];z=a['z'];payload=off[:-1];eos=off[1:]-1;continuation=np.asarray([j for lo,hi in zip(off[:-1],off[1:]) for j in range(lo+1,hi-1)],int);rival=a['rival'];first=z[payload];common=first.mean(0);contrast=first[np.arange(len(first)),y[payload]]-first[:,256];commonpart=common[y[payload]]-common[256];residualpart=contrast-commonpart
 out=dict(complete_correct=int(a['correct'].sum()),OLD_correct=int(a['correct'][:64].sum()),NEW_correct=int(a['correct'][64:].sum()),empty_EOS=int(np.sum(a['choices'][:,0]==256)),CAP=int(np.sum(~a['EOS'])),first_payload_correct=int(np.sum(z[payload].argmax(1)==y[payload])),continuation_correct=int(np.sum(z[continuation].argmax(1)==y[continuation])) if len(continuation) else 0,continuation_targets=len(continuation),true_EOS_correct=int(np.sum(z[eos].argmax(1)==256)),true_EOS_targets=len(eos),mean_first_payload_margin=float(a['margin'][payload].mean()),mean_true_EOS_margin=float(a['margin'][eos].mean()),first_payload_EOS_rival=int(np.sum(rival[payload]==256)),common_logit_argmax=int(common.argmax()),common_target_vs_EOS_mean=float(commonpart.mean()),residual_target_vs_EOS_mean=float(residualpart.mean()),total_target_vs_EOS_mean=float(contrast.mean()),common_residual_identity_max=float(np.max(abs(commonpart+residualpart-contrast))),clipped_gamma_count=int(a['clipping'][:,:2].sum()),clipped_MBON_count=int(a['clipping'][:,2:].sum()),gamma_or_MBON_opportunities_each=32*len(y))
 return out
def run():
 start=time.process_time();overall=load(H/'STATUS.json');assert overall['status']=='THREE_CYCLES_EXECUTED_AWAITING_REDUCTION'
 commits=[json.loads(line) for line in (H/'LEDGER.jsonl').read_text().splitlines() if json.loads(line)['event']=='PANEL_COMMIT']
 assert len(commits)==204
 for row in commits:assert sha(H/'local'/f"{row['tag']}.npz")==row['sha256']
 inputs=load(H/'local/INPUTS.json');feas=[];native=[];shared=[];proofs=[]
 for world in [202610255401,202610255402]:
  for front in ['I_FAST','P_SLOW']:
   key=f'{front}_{world}';d=inputs[str(world)];rows=d['panel'];fr=load(H/f'cases/{key}/FEASIBILITY_RESULT.json');mr=load(H/f'cases/{key}/MARGIN_RESULT.json');au=load(H/f'cases/{key}/AUDIT.json');assert au['status']=='ACCEPTED'
   feas.append({**{k:fr[k] for k in ['case','front','world','records','target_rows','feature_columns','construction','cpu']},'checks':[{k:v for k,v in cell.items() if k!='rollout'}|({'rollout':{k:v for k,v in cell['rollout'].items() if k!='rows'}} if 'rollout' in cell else {}) for cell in fr['checks']]})
   initial=panel(key+'_initial');prior=initial;events=[];I_sum=np.zeros(len(initial['y']));O_sum=I_sum.copy();W_sum=I_sum.copy();old_init=np.flatnonzero(initial['correct'][:64]);own_improved=own_harmed=other_improved=other_harmed=0;write_losses=idle_losses=observation_losses=0
   for j,row in enumerate(d['selected_NEW']):
    idle=panel(f'{key}_e{j:02d}_idle');obs=panel(f'{key}_e{j:02d}_observed');actual=panel(f'{key}_e{j:02d}_actual');I=idle['margin']-prior['margin'];O=obs['margin']-idle['margin'];W=actual['margin']-obs['margin'];I_sum+=I;O_sum+=O;W_sum+=W
    same=next(i for i,r in enumerate(rows) if r['id']==row['id']);lo,hi=actual['offsets'][same:same+2];own=np.zeros(len(W),bool);own[lo:hi]=True;other=~own
    oi=int(np.sum(W[own]>0));oh=int(np.sum(W[own]<0));pi=int(np.sum(W[other]>0));ph=int(np.sum(W[other]<0));own_improved+=oi;own_harmed+=oh;other_improved+=pi;other_harmed+=ph
    il=int(np.sum(prior['correct']&~idle['correct']));ol=int(np.sum(idle['correct']&~obs['correct']));wl=int(np.sum(obs['correct']&~actual['correct']));idle_losses+=il;observation_losses+=ol;write_losses+=wl
    event=dict(index=j,actual=summary(actual),same_required_margin_improved=oi,same_required_margin_worsened=oh,other_required_margin_improved=pi,other_required_margin_worsened=ph,idle_complete_losses=il,observed_complete_losses=ol,write_complete_losses=wl,write_complete_gains=int(np.sum(~obs['correct']&actual['correct'])),same_record_correct_no_write=bool(obs['correct'][same]),same_record_correct_written=bool(actual['correct'][same]),initially_correct_OLD_retained=int(actual['correct'][old_init].sum()),same_record_min_required_margin_before=float(obs['margin'][lo:hi].min()),same_record_min_required_margin_after=float(actual['margin'][lo:hi].min()))
    events.append(event);prior=actual
   final_idle=panel(key+'_final_idle');archived=panel(key+'_archived_final');res=float(np.max(abs(I_sum+O_sum+W_sum-(prior['margin']-initial['margin']))))
   result=dict(case=key,front=front,world=world,initial=summary(initial),continuation_final=summary(prior),post_idle=summary(final_idle),archived_V154_final=summary(archived),initially_correct_OLD=len(old_init),retention_measurable=bool(len(old_init)),initially_correct_OLD_retained_final=int(prior['correct'][old_init].sum()),initially_correct_OLD_retained_post_idle=int(final_idle['correct'][old_init].sum()),same_record_margin_improvements=own_improved,same_record_margin_worsening=own_harmed,other_record_margin_improvements=other_improved,other_record_margin_worsening=other_harmed,idle_complete_losses=idle_losses,observed_complete_losses=observation_losses,write_complete_losses=write_losses,mean_cumulative_idle=float(I_sum.mean()),mean_cumulative_observation=float(O_sum.mean()),mean_cumulative_write=float(W_sum.mean()),cumulative_telescoping_residual_max=res,events=events,parity=mr['parity'],initial_countermodels=mr['initial_countermodels'],final_countermodels=mr['events'][-1]['countermodels'])
   assert res<1e-9;native.append(result);shared.append(dict(case=key,initial=result['initial'],final=result['continuation_final'],post_idle=result['post_idle'],scope='panel common/residual decomposition at q=0; no policy subtraction'))
   proofs.append(dict(case=key,audit_sha256=sha(H/f'cases/{key}/AUDIT.json'),accepted=True))
 save(H/'FEASIBILITY.json',dict(status='COMPLETE',evidence='E0 finite evaluator decisions, not native realizability',cases=feas))
 save(H/'MARGIN_SUMMARY.json',dict(status='COMPLETE',evidence='E1 reused-world diagnostic only; no E3 score',cases=native))
 save(H/'SHARED_OUTPUT.json',dict(status='COMPLETE',cases=shared))
 counts=load(H/'COUNTS.json');c=counts['counts'];assert c['observed_records']==136 and c['native_pairs']==16320 and c['native_updating_records']==64 and c['native_inhibited_records']==64
 assert c['witness_queries']==1536 and c['technical_records']==8 and c['observed_targets']<=432 and c['evaluator_distributions']<=120832
 for p,h in load(H/'SOURCE_LOCK.json').items():assert sha(H.parent/p)==h
 prior=load(H.parent/'minifly_v156_design_20261025/PROVENANCE.json');arithmetic=next(x for x in prior['local_sources'] if x['path']=='PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json');assert sha(H.parent/arithmetic['path'])==arithmetic['sha256']
 output_bytes=sum(p.stat().st_size for p in H.rglob('*') if p.is_file());assert output_bytes<1073741824
 audit=dict(status='ACCEPTED',scope='independent saved-artifact case audits, aggregate arithmetic, exposure and source closure',cases=proofs,panel_commit_hashes_checked=len(commits),source_files_checked=len(load(H/'SOURCE_LOCK.json')),gradual_warmup_counters_preserved=True,local_native_teacher_unchanged=True,no_arithmetic_specific_candidate_structure=True,all_257_classes=True,no_forced_EOS=True,assessment_scored=False,new_core_adopted=False,three_cycles=3,observed_records=c['observed_records'],observed_targets=c['observed_targets'],native_query_pairs=c['native_pairs'],witness_queries=c['witness_queries'],evaluator_distributions=c['evaluator_distributions'],training_replayed=False,model_queries_for_reduction=0,postrun_reducer_CPU=time.process_time()-start,postrun_reducer_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,local_output_bytes=output_bytes)
 save(H/'FINAL_AUDIT.json',audit)
 print(json.dumps(dict(audit=audit,case_summaries=[{k:r[k] for k in ['case','initially_correct_OLD','initially_correct_OLD_retained_final','same_record_margin_improvements','same_record_margin_worsening','idle_complete_losses','observed_complete_losses','write_complete_losses','mean_cumulative_idle','mean_cumulative_observation','mean_cumulative_write']}|{'initial':r['initial']['complete_correct'],'final':r['continuation_final']['complete_correct']} for r in native]),indent=2))
if __name__=='__main__':run()
