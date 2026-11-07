"""E13 D steps1–3. No refitting before the true-heading ceiling decision."""
import json,csv,time
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
import path_integration as P
F=P.first;R=F.ROOT

def digest(p):return sha256(p.read_bytes()).hexdigest()
def save(name,d):F.save(name,d)
@torch.no_grad()
def main():
 torch.set_num_threads(1)
 if (R/'PI_FOLLOWUP_RESULTS.json').exists():raise RuntimeError('Existing diagnostic results; do not overwrite')
 lock=F.load('PI_FOLLOWUP_LOCK.json')
 for n,h in lock['files'].items():assert digest(R/n)==h,n
 graph=F.read_graph();model=P.get_model(graph);nets=P.get_arm(graph,model);loop=F.Loop(model,'native');panels=P.datasets();prior=F.load('PI_LOCK.json');old=F.load('PI_RESULTS.json');records=F.load('PI_CURVES.json')['records'];rows=[];episode_rows=[];start=time.monotonic()
 for seed in F.SEEDS:
  c=P.get_compass(prior,seed); caches={k:P.cache(loop,c,panels[k]) for k in ('main','returning')}
  returns=panels['returning'];truth_heading=F.tensor(returns[0])[:,None]+F.tensor(returns[1]).cumsum(1)
  turn_error=torch.rad2deg(F.wrap(F.decode(caches['returning'][0][:,109]@loop.observer)-truth_heading[:,109]))
  outbound=P.DT*returns[2][:,:50].sum(1)
  for arm,net in nets.items():
   record=next(x for x in records if x['seed']==seed and x['arm']==arm);params=F.tensor([record['parameters'][k] for k in P.NAMES]);scores={}
   for name in ('main','returning'):
    e,speed,truth,loss=caches[name];data=(*net.pfns(e,speed),truth,loss);out=outbound if name=='returning' else None
    native=P.measure(net,data,params,out,prior);reference=old['scores'][str(seed)][f'{arm}_fitted'][name]
    for key in ('angle_count','median_angle_deg','median_magnitude_ratio') if name=='main' else ('median_residual_over_outbound',):
     assert np.isclose(native[key],reference[key],atol=1e-7,rtol=0),(seed,arm,name,key)
    a,yaw,sp=panels[name];head=F.tensor(a)[:,None]+F.tensor(yaw).cumsum(1)
    # Invert only the fixed geometric observer, not a learned gain or task score.
    # Correction converges on uniformly sampled headings as well as all panel headings.
    phase=head.clone()
    for _ in range(80):
     bump=c[11]*F.phase_bump(loop.phase,phase)
     xy=bump@loop.observer
     err=F.wrap(torch.atan2(xy[...,1],xy[...,0])-head)
     phase=phase-.7*err
    exact=c[11]*F.phase_bump(loop.phase,phase)
    # Encode the supplied true heading with the original cue's bump family/amplitude after inverting the fixed observer.
    enc_error=F.wrap(torch.atan2((exact@loop.observer)[...,1],(exact@loop.observer)[...,0])-head)
    assert float(torch.rad2deg(enc_error.abs()).max())<1e-6, "True-heading clamp not exact"
    clamp_data=(*net.pfns(exact,speed),truth,loss*0)
    clamped=P.measure(net,clamp_data,params,out,prior)
    scores[name]=dict(own_compass=native,true_heading_clamped=clamped,max_clamp_encoding_error_deg=float(torch.rad2deg(enc_error.abs()).max()))
    if name=='returning':
     pred=net.predict(data[:2],params)[:,-1].numpy();unit=np.column_stack((np.cos(a),np.sin(a)));perp=np.column_stack((-np.sin(a),np.cos(a)))
     along=(pred*unit).sum(1)/outbound;across=(pred*perp).sum(1)/outbound
     decoded_turn=turn_error.numpy()
     for i in range(48):episode_rows.append(dict(seed=seed,arm=arm,episode=i,turn_direction=int(np.sign(yaw[i,50])),along_outbound_fraction=float(along[i]),perpendicular_fraction=float(across[i]),residual_fraction=float(np.linalg.norm(pred[i])/outbound[i]),turn_error_signed_deg=float(decoded_turn[i])))
     scores['return_decomposition']=dict(median_signed_along_fraction=float(np.median(along)),median_absolute_along_fraction=float(np.median(abs(along))),median_signed_perpendicular_fraction=float(np.median(across)),median_absolute_perpendicular_fraction=float(np.median(abs(across))),median_absolute_turn_error_deg=float(np.median(abs(decoded_turn))),max_absolute_turn_error_deg=float(abs(decoded_turn).max()))
   rows.append(dict(seed=seed,arm=arm,scores=scores))
 ceiling_stop=any(x['scores']['returning']['true_heading_clamped']['median_residual_over_outbound']>.30 for x in rows if x['arm']=='native')
 result=dict(lock_sha256=digest(R/'PI_FOLLOWUP_LOCK.json'),scope='Wiring provides an instantaneous vector; temporal accumulation is supplied by the model. No learning or refit in these diagnostics. The fixed observer is inverted numerically so the clamp decodes to true heading within1e-6degree; no learned gains are changed.',rows=rows,episode_rows=episode_rows,compass_return_schedule_episodes=48,diagnostic_seconds=time.monotonic()-start,original_PI_score_parity=True,clamped_return_stop=ceiling_stop,status='STOP_CLAMPED_RETURN_FAILED' if ceiling_stop else 'DIAGNOSTICS_COMPLETE_REFIT_REQUIRED')
 save('PI_FOLLOWUP_RESULTS.json',result)
 with (R/'PI_FOLLOWUP_RETURN_EPISODES.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(episode_rows[0]));w.writeheader();w.writerows(episode_rows)
 lines=['# E13 D: path-integration follow-up','',result['scope'],'','Steps1–3 delivered for all three seeds. Original native/rewired PI scores reproduced within1e-7. Existing main and48-episode return panels used unchanged. The 180° compass-only probe uses their same scripted in-place turn; no translation feeds the compass.', '', '|Seed|Arm|Own main /96|Clamped main /96|Own return fraction|Clamped return fraction|Median turn error deg|','|---|---|---:|---:|---:|---:|---:|']
 for row in rows:
  s=row['scores'];lines.append(f'|{row["seed"]}|{row["arm"]}|{s["main"]["own_compass"]["angle_count"]}|{s["main"]["true_heading_clamped"]["angle_count"]}|{s["returning"]["own_compass"]["median_residual_over_outbound"]:.5f}|{s["returning"]["true_heading_clamped"]["median_residual_over_outbound"]:.5f}|{s["return_decomposition"]["median_absolute_turn_error_deg"]:.3f}|')
 lines += ['', '## Residual decomposition','','|Seed|Arm|Median absolute along /outbound|Median absolute perpendicular /outbound|','|---|---|---:|---:|']
 for row in rows:
  d=row['scores']['return_decomposition'];lines.append(f'|{row["seed"]}|{row["arm"]}|{d["median_absolute_along_fraction"]:.5f}|{d["median_absolute_perpendicular_fraction"]:.5f}|')
 lines += ['', 'All288 episode points are in PI_FOLLOWUP_RETURN_EPISODES.csv. Signed components and errors are also in JSON. Clamping supplies the true heading encoded by the original initial-cue bump family/amplitude; encoding error is reported explicitly and required below1e-6degree. The invalid initial nominal-phase clamp is preserved in PI_FOLLOWUP_CYCLE1_INVALID_CLAMP; it was rejected before any refitting. This is an intervention on heading input, not new learned capability.', '', '## Disposition','', 'STOP: native clamped-heading return residual exceeds0.30 in at least one seed. As explicitly prescribed, no refit is run. The supplied accumulator/readout package fails even with true heading; compass drift alone cannot explain failure.' if ceiling_stop else 'Diagnostics complete. Refit branch must be locked from these measurements before any gradient step. No refit has been run.']
 (R/'PI_FOLLOWUP_REPORT.md').write_text('\n'.join(lines)+'\n')
 save('PI_FOLLOWUP_STATUS.json',dict(status=result['status'],steps_1_to_3_complete=True,refit='NOT_RUN',prescribed_stop=ceiling_stop))
 print(json.dumps({k:v for k,v in result.items() if k not in ('rows','episode_rows')},indent=2),flush=True)
if __name__=='__main__':main()
