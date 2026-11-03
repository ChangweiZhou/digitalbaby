"""Full read-only snapshot profile and exact post-dropout sham validity census."""
import bootstrap as B
from learner import Learner
from data import prepare
import evaluate as E
import protocol as Q
import time,json,resource,sys,io,zipfile
from pathlib import Path
H=B.H;T=B.T;np=B.np

def save(n,v):(H/n).write_text(json.dumps(v,indent=2)+'\n')
def main():
 l=json.loads((H/'EXPOSURE_LEDGER.json').read_text());start=time.process_time();sensor=B.sensor.Sensor();sp,ids,manifest=prepare();t=time.process_time();packed={name:E.pack(sensor,rs) for name,rs in sp.items()};encoding=time.process_time()-t
 out={};arrays={};rawbytes=0;max_cp=0;armtimes={};norm={};totalviews=totalforward=0
 # Full snapshot cost at one cold state per arm. Published outputs retain only
 # engineering costs, not a frozen-state proxy efficacy result.
 for arm in ('NORMAL','HRES','NORM_SHAM'):
  net=Learner(arm);armtimes[arm]=0.;norm[arm]={'sites':0,'clipped_sites':0,'zero_denominator_sites':0,'harmless_both_zero_sites':0,'material_norm_mismatch_sites':0,'gain_max':0.,'max_abs_norm_error':0.}
  original=net.forward
  def tracked(*args,**kw):
   result=original(*args,**kw);d=net.last
   for k in ('sites','clipped_sites','zero_denominator_sites','harmless_both_zero_sites','material_norm_mismatch_sites'):norm[arm][k]+=d[k]
   for k in ('gain_max','max_abs_norm_error'):norm[arm][k]=max(norm[arm][k],d[k])
   return result
  net.forward=tracked;before=B.statehash(net);t=time.process_time()
  for name,rs in sp.items():
   score,site=E.score(net,rs,packed[name],sp['A_held'] if name=='EDIT_held' else None,record_log_sums=name=='A_train')
   if name=='EDIT_held':score['partitions']=E.partitions(site,sp['A_held'],packed[name],packed['A_held'])
   if name=='A_train':record_summaries=site
   # Saved-only loss/EOS reconstruction; never another model evaluation.
   eos=site['y']==256;assert np.allclose(site['logp_true'][eos],site['logp_EOS'][eos],atol=1e-12);assert abs(float((-site['logp_true']/np.log(2)).mean())-score['bits'])<1e-10
   assert abs(float((-site['logp_EOS'][eos]/np.log(2)).mean())-score['EOS_NLL'])<1e-10
   totalviews+=len(rs);totalforward+=len(site['y'])
   if arm=='HRES':
    for k,v in site.items():arrays[name+'__'+k]=v
  armtimes[arm]=time.process_time()-t;assert B.statehash(net)==before
  forecast=E.forecast_from_summaries(record_summaries);assert np.isfinite(forecast).all() and abs(float(np.exp(forecast).sum())-1)<1e-10
  buf=io.BytesIO();T.save(net.state(),buf);max_cp=max(max_cp,len(buf.getvalue()));assert len(buf.getvalue())<=1048576
 rawbytes=sum(v.nbytes for v in arrays.values());np.savez_compressed(H/'local/profile_sites.npz',**arrays);T.save(Learner('HRES').state(),H/'local/profile_checkpoint.pt')
 # The alignment diagnostic is label-blind in its mapping; two actual head-loss
 # routes are charged, plus one backbone pass. All state/RNG remains unchanged.
 t=time.process_time();n=Learner('HRES');alignment=E.alignment(n,sp['A_held'],packed['A_held']);alignment_cpu=time.process_time()-t;assert alignment['eligible_sites']<=alignment['held_sites'];save('ALIGNMENT_PROFILE.json',{k:v for k,v in alignment.items() if k not in ('aligned_bits','exchanged_bits','per_record_aligned_bit_sums','per_record_exchanged_bit_sums')})
 totalviews+=3*len(sp['A_held']);totalforward+=alignment['held_sites']+2*alignment['eligible_sites']
 t=time.process_time();gen=E.generate(n,sensor,sp['A_held'][:16]+sp['B_held'][:16]);generation_cpu=time.process_time()-t;outputs=sum(v['model_outputs'] for v in gen);assert B.statehash(n)==B.statehash(Learner('HRES'));totalviews+=len(gen);totalforward+=outputs
 c3=json.loads((H/'CYCLE3.json').read_text());batch=next(v for v in c3['all_tails'] if v['sites']==256);update_upper=batch['CPU_seconds']/256*103092*2
 profile_cpu=time.process_time()-start;qualification_cpu=l['CPU_seconds_including_15_setup_allowance']+profile_cpu+5
 projected=qualification_cpu+1.5*(encoding+10*sum(armtimes.values())+alignment_cpu+generation_cpu/max(1,outputs)*18432+update_upper+60)
 # Full raw site arrays at5 checkpoints/arm, not10; record-class summaries only
 # at A2. The bound is deliberately uncompressed and counts a second owner ZIP.
 classbytes=sum(arrays['A_train__'+k].nbytes for k in ('record_class_log_sums','record_class_site_counts','record_input_sha256'))
 all_artifact_bound=15*max_cp+15*(rawbytes-classbytes)+3*classbytes+2*1048576
 runtime=sum(p.stat().st_size for p in Path('/private/tmp/minifly_v183_runtime').rglob('*') if p.is_file());current=sum(p.stat().st_size for p in H.rglob('*') if p.is_file());output=runtime+1.5*(2*all_artifact_bound+current+1048576)
 inference=sum(a.nbytes for a in sensor.arrays)+sum(v.numel()*v.element_size() for v in n.p.values())+n.indices.nbytes+n.ix.numel()*n.ix.element_size()+n.fh.numel()*4+n.ft.numel()*4+254096
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);admitted=projected<=900 and projected<=1800 and output<=268435456 and peak<=1073741824 and inference<=8388608 and max_cp<=1048576
 save('RESOURCE_ADMISSION.json',{'status':'ADMITTED' if admitted else 'RESOURCE_NOT_ADMITTED','projected_aggregate_CPU_with1_5_reserve':projected,'projected_active_execution_wall_seconds':projected,'authoring_and_owner_wait_are_not_process_runtime':True,'peak_RSS_bytes':peak,'model_specific_inference_bound_bytes':inference,'full_mutable_checkpoint_bytes':max_cp,'raw_full_snapshot_array_bytes':rawbytes,'future_full_states':15,'future_full_site_snapshots':15,'all10_summary_checkpoints_each_arm':True,'future_readonly_primary_panels_target_sites':999210,'future_generation_outputs_max':18432,'alignment_extra_backbone_sites':alignment['held_sites'],'alignment_extra_scored_target_evaluations':2*alignment['eligible_sites'],'full_snapshot_CPU_by_arm':armtimes,'encoding_CPU_seconds':encoding,'alignment_CPU_seconds':alignment_cpu,'generation_CPU_seconds_for32_prompts':generation_cpu,'candidate_256_update_CPU_seconds':batch['CPU_seconds'],'profile_CPU_seconds':profile_cpu,'qualified_CPU_carried_upper':qualification_cpu,'projected_outputs_including_owner_archive_runtime_and_reserve_bytes':output,'fixed_reused_runtime_disk_bytes':runtime,'all_future_artifact_raw_bound_bytes':all_artifact_bound,'norm_census_common_fresh_birth':norm,'any_actual_control_mismatch_forces_CONTROL_NOT_QUALIFIED':True,'no_new_countermodel_fits':True,'read_only_profile_record_views':totalviews,'read_only_profile_forward_sites_including_alignment_generation':totalforward,'no_optimizer_updates_during_profile':True})
 save('SAVED_SITE_REVIEW.json',{'status':'PASS','true_EOS_loss_from_actual_saved_fields':True,'full257_log_mean_forecast_from_record_summaries':True,'alignment_matching_position_other_record_and_state_fixed':True,'profile_is_NOT_scientific_efficacy':True,'cross_family_review':'PENDING','formal_MRK1':'UNAVAILABLE'})
 # Append read-only costs to the same ledger without rerunning a qualification.
 l['rows'].append({'label':'full3-arm cold snapshot profile plus alignment/generation; no fitted outcomes','category':'read_only','record_views':totalviews,'supervised_target_evaluations':0,'optimizer_updates':0,'forward_sites':totalforward});l['totals']['read_only_record_views']=totalviews;l['totals']['read_only_forward_sites']=totalforward;l['profile_CPU_seconds']=profile_cpu;l['qualified_CPU_carried_upper']=qualification_cpu;(H/'EXPOSURE_LEDGER.json').write_text(json.dumps(l,indent=2)+'\n')
 print(json.dumps({'resource':'ADMITTED' if admitted else 'RESOURCE_NOT_ADMITTED','CPU':projected,'output_bound':output,'RSS':peak,'checkpoint':max_cp,'alignment_eligible':alignment['eligible_sites'],'norm_sham_invalid_real_sites':norm['NORM_SHAM']['material_norm_mismatch_sites']}),flush=True)
if __name__=='__main__':main()
