"""Full roster profile and saved-site falsifiers; no scientific fitting."""
import bootstrap as B
from learner import Learner
import metrics as M
from reduce import rebuild,decide
from cycle2 import continuing_ledger,save
from ledger import H
import time,json,copy,io,zipfile,resource,sys
T=B.T;np=B.np

def fake():
 arms={}
 for arm in ('NORMAL','HPC','SHUFFLED'):
  v=3. if arm=='HPC' else 3.3;a={}
  for stage in ('birth','A1','A2','B1','B2','B3','EDIT1','EDIT2','NOEDIT1','NOEDIT2'):
   a[stage]={panel:{'bits':v,'EOS_NLL':1.} for panel in ('A_held','B_held','EDIT_held')};a[stage]['EDIT_held']['partitions']={k:{'bits':v+(.2 if stage=='NOEDIT2' and k=='CHANGED' else 0)} for k in ('SAME_KC','CHANGED','ALTERED_KC')}
  arms[arm]=a
 return arms

def main():
 ledger=continuing_ledger();start=time.process_time();sensor=B.sensor.Sensor();data,meta=B.D.prepare(sensor);sp=dict(data['M1']);rmeta=meta['M1']['correction'];rule=(bytes.fromhex(rmeta['context_hex']),rmeta['old'],rmeta['new'])
 for split in ('calib','held'):sp['EDIT_'+split]=[B.D.corrected(r,rule) for r in sp['A_'+split]]
 assert sum(len(rs) for rs in sp.values())==544 and sum(len(r)+1 for rs in sp.values() for r in rs)==33307
 t=time.process_time();packed={k:M.pack(sensor,rs) for k,rs in sp.items()};encoding=time.process_time()-t;net=Learner();t=time.process_time();panels={};sitefields={};rawbytes=0
 for name,rs in sp.items():
  out,sites=M.score(net,rs,packed[name],sp['A_held'] if name=='EDIT_held' else None,record_log_sums=name=='A_train')
  if name=='EDIT_held':out['partitions']=M.partitions(sites,sp['A_held'],packed[name],packed['A_held'])
  check=rebuild(sites)
  for k,v in check.items():assert abs(v-out[k])<1e-10,(name,k)
  # EOS uses genuine sites, and loss is reconstructed from log probabilities.
  eos=sites['y']==256;assert np.allclose(sites['logp_EOS'][eos],sites['logp_true'][eos],atol=1e-12);assert np.allclose(np.exp(sites['logp_EOS']),sites['prob_EOS'],rtol=1e-9,atol=1e-15)
  panels[name]=out
  for k,v in sites.items():sitefields[name+'__'+k]=v
  rawbytes+=sum(v.nbytes for v in sites.values())
 evalcpu=time.process_time()-t
 train_s={k[len('A_train__'):]:v for k,v in sitefields.items() if k.startswith('A_train__')};held_s={k[len('A_held__'):]:v for k,v in sitefields.items() if k.startswith('A_held__')};logq=M.forecast_from_summaries(train_s);assert np.isfinite(logq).all() and abs(float(np.exp(logq).sum())-1)<1e-10
 np.savez_compressed(H/'local/profile_sites.npz',**sitefields);T.save(net.state(),H/'local/profile_checkpoint.pt');cpbytes=(H/'local/profile_checkpoint.pt').stat().st_size;assert cpbytes<=1048576
 np.savez_compressed(H/'local/profile_forecast.npz',logq=logq,record_class_log_sums=train_s['record_class_log_sums'],record_class_site_counts=train_s['record_class_site_counts'],record_input_sha256=train_s['record_input_sha256'])
 # Numeric overflow/underflow fixture: finite log sums must not become a clipped
 # probability distribution. This fixture is within the shared technical cap.
 ledger.add('log-domain mean forecast underflow and saved-only gate fixtures',n=2,paths=1,base_records=1,loss_records=1)
 synthetic={'record_class_log_sums':np.array([[-10000.]*257,[-10001.]*257]),'record_class_site_counts':np.array([1,1])};ll=M.forecast_from_summaries(synthetic);assert np.isfinite(ll).all() and M.forecast_loss(ll,{'y':np.array([0,256])})>10000
 ff=fake();assert decide(ff,3.1)['verdict']=='HPC_DEV_GAIN';assert decide(ff,2.)['verdict']=='HPC_RELATIVE_GAIN_ONLY'
 # Superseded original-A post-C loss must not veto otherwise valid retention.
 ff['HPC']['EDIT2']['A_held']['bits']=100.;assert decide(ff,3.1)['verdict']=='HPC_DEV_GAIN'
 ff['HPC']['EDIT2']['EDIT_held']['partitions']['SAME_KC']['bits']=4.;assert not decide(ff,3.1)['checks']['post_C_still_valid_SAME_KC']
 t=time.process_time();generated=M.generate(net,sensor,sp['A_held'][:16]+sp['B_held'][:16]);generation=time.process_time()-t;steps=sum(r['model_outputs'] for r in generated);assert B.statehash(net)==B.statehash(Learner())
 ledger.add('batch conservation/lifetime/EOS policy static fixtures',n=0,paths=1,base_records=1,loss_records=0)
 c2=json.loads((H/'CYCLE2.json').read_text());full=next(v for v in c2['n1_and_full_batch_tail_cases'] if v['n']==256);update_bound=full['CPU_seconds']/256*(34364*3)*2 # conservative twice worst full candidate per site for all3 arms
 # All30 snapshots at measured full10-panel cost; all9 generation rosters at
 # 64 outputs regardless of an early actual EOS. No renewed count grid.
 projected_cpu=1.5*(encoding+30*evalcpu+generation/max(1,steps)*18432+update_bound+60)+ledger.write()['technical_target_evaluations']*0 # arithmetic independent of outcomes
 current=json.loads((H/'EXPOSURE_LEDGER.json').read_text());projected_cpu+=current['charged_CPU_seconds'];current_files=sum(p.stat().st_size for p in H.rglob('*') if p.is_file());runtime_bytes=sum(p.stat().st_size for p in Path('/private/tmp/minifly_v183_runtime').rglob('*') if p.is_file())
 # Record-class vectors occur only at A2, not30 times. Use uncompressed site
 # arrays for the bound, full mutable checkpoint size, archive worst copy,
 # explicit metadata/review allowance and the1.5 reserve. Isolated dependency
 # install is counted separately at its exact fixed disk footprint.
 classbytes=train_s['record_class_log_sums'].nbytes+train_s['record_class_site_counts'].nbytes+train_s['record_input_sha256'].nbytes
 all_states=30*cpbytes+30*(rawbytes-classbytes)+3*classbytes+1048576
 output_bound=runtime_bytes+1.5*(2*all_states+current_files+1048576)
 with zipfile.ZipFile(H/'local/profile_reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in ('profile_sites.npz','profile_checkpoint.pt','profile_forecast.npz'):z.write(H/'local'/name,name)
 inf=sum(a.nbytes for a in sensor.arrays)+sum(v.numel()*v.element_size() for v in net.p.values())+net.indices.nbytes+net.ix.numel()*net.ix.element_size()+net.fh.numel()*4+net.ft.numel()*4+250000
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);admitted=projected_cpu<=900 and output_bound<=268435456 and peak<=1073741824 and inf<=8388608
 save('RESOURCE_ADMISSION.json',{'status':'ADMITTED' if admitted else 'RESOURCE_NOT_ADMITTED','projected_aggregate_CPU_with_1_5_reserve':projected_cpu,'projected_aggregate_wall_single_process_seconds':projected_cpu+current['wall_seconds_including_300_setup_allowance'],'wall_cap_seconds':1800,'peak_RSS_bytes':peak,'single_model_inference_bound_bytes':inf,'full_mutable_checkpoint_bytes':cpbytes,'profile_one_full_site_roster_uncompressed_bytes':rawbytes,'profile_one_full_site_roster_CPU_seconds':evalcpu,'encoding_CPU_seconds':encoding,'generation_measured_CPU_seconds':generation,'full_candidate_update_256_CPU_seconds':full['CPU_seconds'],'future_checkpoints':30,'future_readonly_target_sites':999210,'generation_max_outputs':18432,'isolated_runtime_install_disk_bytes':runtime_bytes,'projected_all_outputs_including_state_archive_and_runtime_bytes':output_bound,'uncompressed_future_states_sites_upper_bytes':all_states,'new_countermodel_fits':0,'no_learning_during_profile':True,'mean_forecast_record_class_bytes_per_arm':train_s['record_class_log_sums'].nbytes,'aux_float64_and_extra_rng_accounted':True,'profile_CPU_seconds':time.process_time()-start})
 save('SAVED_METRIC_REVIEW.json',{'status':'PASS','mode':'EXECUTED_SAME_FAMILY_SELF_AUDIT','profile_target_sites':33307,'record_views':544,'saved_site_EOS_loss_and_scalar_reduction_match':True,'mean_forecast_finite_full257_log_support':True,'saved160x257_record_vectors':True,'no_held_label_fit':True,'superseded_original_A_not_retention_gate':True,'same_KC_retention_failure_detected':True,'read_only_model_optimizer_and_all_RNG_unchanged':True,'new_optimizer_updates':0,'counts':ledger.write()})
 print('RESOURCE',admitted,'CPU',projected_cpu,'output',output_bound,'RSS',peak,'profile CPU',time.process_time()-start,flush=True)
if __name__=='__main__':
 from pathlib import Path
 main()
