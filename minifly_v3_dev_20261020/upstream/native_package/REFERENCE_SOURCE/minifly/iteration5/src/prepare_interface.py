"""Freeze one same-specimen anatomical interface and unlabeled gain calibration."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd
from scipy.sparse import load_npz,save_npz,diags
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
SIDES=['left','right'];MTYPES=['MBON11','MBON14'];DTYPES=['PPL101','PPL106']

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2097152),b''):h.update(b)
 return h.hexdigest()

def patterns(n,F,seed,exclude=()):
 rng=np.random.default_rng(seed);seen={tuple(np.flatnonzero(p)) for p in exclude};out=[]
 while len(out)<n:
  ids=tuple(sorted(rng.choice(F,8,replace=False).tolist()))
  if ids in seen:continue
  x=np.zeros(F);x[list(ids)]=1;out.append(x);seen.add(ids)
 return np.array(out)

def encode(B,pn_type_idx,kc_side,p):
 raw=np.asarray(B.T@(p[pn_type_idx])).ravel();x=np.zeros(len(raw))
 for s in [0,1]:
  ix=np.flatnonzero((kc_side==s)&(raw>0));n=min(len(ix),int(np.ceil(.05*np.sum(kc_side==s))))
  order=np.lexsort((ix,-raw[ix]));x[ix[order[:n]]]=1.
 return x

def main():
 m=pd.read_parquet(BASE/'data/flywire630/metadata.parquet');A=load_npz(BASE/'data/flywire630/graph.npz');kc=np.flatnonzero(m.cell_class.eq('Kenyon_Cell'));pn=np.flatnonzero(m.cell_class.eq('ALPN') & m.top_nt.eq('acetylcholine'))
 B=A[pn][:,kc];pn=pn[np.asarray(B.sum(axis=1)).ravel()>0];B=A[pn][:,kc]/.275;assert np.all(B.data>0)
 mass=np.asarray(B.sum(axis=0)).ravel();B=(B@diags(np.divide(1,mass,out=np.zeros_like(mass),where=mass>0))).tocsr();save_npz(ROOT/'data/pn_to_kc.npz',B)
 names=m.hemibrain_type.iloc[pn].fillna('').to_numpy();names=np.array([n if n else 'untyped:'+str(m.root_id.iloc[i]) for n,i in zip(names,pn)]);types,pi=np.unique(names,return_inverse=True);ks=m.side.iloc[kc].map(dict(left=0,right=1)).to_numpy();assert np.all(np.isfinite(ks))
 mgroups=[np.flatnonzero(m.hemibrain_type.eq(t)&m.side.eq(s)) for t in MTYPES for s in SIDES];dgroups=[np.flatnonzero(m.hemibrain_type.eq(t)&m.side.eq(s)) for t in DTYPES for s in SIDES]
 outputs=np.column_stack([np.asarray(abs(A[kc][:,ids]).sum(axis=1)).ravel()/.275 for ids in mgroups]);teachers=np.column_stack([np.asarray(abs(A[ids][:,kc]).sum(axis=0)).ravel()/.275 for ids in dgroups]);T=np.zeros_like(teachers)
 for role in [0,1]:
  cols=slice(2*role,2*role+2);den=teachers[:,cols].sum(axis=1);T[:,cols]=np.divide(teachers[:,cols],den[:,None],out=np.zeros_like(teachers[:,cols]),where=den[:,None]>0);T[:,cols]*=(outputs[:,cols].sum(axis=1)>0)[:,None]
 feedback=np.array([[float(abs(A[ii][:,jj]).sum()/.275) for jj in dgroups] for ii in mgroups]);routing=feedback.copy()
 for role in [0,1]:
  rows=slice(2*role,2*role+2);den=routing[rows].sum(axis=0);routing[rows]=np.divide(routing[rows],den[None,:],out=np.zeros_like(routing[rows]),where=den[None,:]>0)
 mm=np.array([[float(abs(A[ii][:,jj]).sum()/.275) for jj in mgroups[2:]] for ii in mgroups[:2]]);den=mm.sum(axis=0);mm=np.divide(mm,den[None,:],out=np.zeros_like(mm),where=den[None,:]>0)
 dev=patterns(16,len(types),241501);evalbank=patterns(32,len(types),241502,dev);seen={tuple(np.flatnonzero(x)) for x in np.vstack([dev,evalbank])};overlaps=[]
 for i in range(8):
  parent=np.flatnonzero(evalbank[i]);shared=parent[:4];other=[t for t in np.flatnonzero(evalbank[(i+16)%32]) if t not in parent][:4]
  if len(other)<4:other+=list(t for t in range(len(types)) if t not in parent and t not in other)[:4-len(other)]
  row=np.zeros(len(types));row[np.r_[shared,other]]=1;assert np.sum(row*evalbank[i])==4;assert tuple(np.flatnonzero(row)) not in seen;seen.add(tuple(np.flatnonzero(row)));overlaps.append(row)
 bank=np.vstack([evalbank,overlaps]);activity=np.array([encode(B,pi,ks,p) for p in dev]);base=np.divide(outputs,outputs.sum(axis=0)[None,:],out=np.zeros_like(outputs),where=outputs.sum(axis=0)[None,:]>0);mean_drive=np.mean(activity@base,axis=0);assert np.all(mean_drive>0);Q=base/mean_drive
 np.savez(ROOT/'data/interface.npz',output_weights=Q,output_synapses=outputs,teacher_routes=T,teacher_synapses=teachers,feedback_routes=routing,feedback_synapses=feedback,mbon_routes=mm,pn_type_index=pi,kc_side=ks,kc_parent_index=kc,pn_parent_index=pn,development_patterns=dev,evaluation_patterns=bank)
 pd.DataFrame(dict(channel=np.arange(len(types)),historical_type=types)).to_csv(ROOT/'data/pn_channels.csv',index=False)
 for name,ix in [('coding_units',kc),('input_neurons',pn)]:m.iloc[ix][['root_id','parent_index','cell_class','hemibrain_type','side','top_nt']].to_csv(ROOT/f'data/{name}.csv',index=False)
 allx=np.array([encode(B,pi,ks,p) for p in bank]);np.save(ROOT/'data/evaluation_kc_activity.npy',allx)
 rows=[]
 for i in range(len(bank)):
  rows.append(dict(cue=i,active_PN_types=int(bank[i].sum()),active_KCs=int(allx[i].sum()),**{f'input_mass_{j}':float(allx[i]@Q[:,j]) for j in range(4)}))
 pd.DataFrame(rows).to_csv(ROOT/'results/input_activity_inventory.csv',index=False)
 overlaps=[]
 for i in range(len(bank)):
  for j in range(i+1,len(bank)):
   inter=np.sum((allx[i]>0)&(allx[j]>0));union=np.sum((allx[i]>0)|(allx[j]>0));overlaps.append(dict(cue1=i,cue2=j,shared_PN_types=int(np.sum(bank[i]*bank[j])),shared_KCs=int(inter),KC_jaccard=float(inter/union) if union else 0.))
 pd.DataFrame(overlaps).to_csv(ROOT/'results/cue_overlap.csv',index=False)
 files=[BASE/'data/flywire630/metadata.parquet',BASE/'data/flywire630/graph.npz',BASE/'iteration4/results/fitted_kernels.json',BASE/'paper_refinement/sources/huang_model/data_and_parameters/Dx_steady_state_nonlinear_3_27-Mar-2023_2modules.mat']
 raw_nt=m.top_nt.iloc[kc].value_counts().to_dict();neg=int(np.sum(np.asarray(A[kc][:,np.concatenate(mgroups)].sum(axis=1)).ravel()<0));summary=dict(specimen='FlyWire materialization 630 only',coding_units=len(kc),input_neurons=len(pn),input_type_channels=len(types),development_patterns=16,evaluation_patterns=len(bank),calibration_mean_unadapted_input=mean_drive.tolist(),fixed_gain=(1/mean_drive).tolist(),mutable_float64_values=4*len(kc),raw_KC_transmitter_labels=raw_nt,KC_semantic_transmitter='acetylcholine (class-level literature correction, raw graph unchanged)',KC_correction_source='https://www.nature.com/articles/s41586-024-07968-y',negative_signed_KC_output_rows_reinterpreted_as_unsigned_counts=neg,parameters_fitted_to_learning_outcomes=0,sources=[dict(path=str(p.relative_to(BASE)),sha256=digest(p)) for p in files])
 (ROOT/'results/interface_calibration.json').write_text(json.dumps(summary,indent=2))
 handoff=Path('/Users/pencilbard/Downloads/MiniFly_bounded_integration_handoff.md')
 if handoff.exists() and not (ROOT/'sources/handoff_proposal.md').exists():shutil.copyfile(handoff,ROOT/'sources/handoff_proposal.md')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
