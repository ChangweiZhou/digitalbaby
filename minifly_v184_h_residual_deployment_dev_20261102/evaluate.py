"""Exact same head, saved-site metrics and frozen residual alignment diagnostic."""
import bootstrap as B
import importlib.util,sys
from pathlib import Path
from protocol import derangement
spec=importlib.util.spec_from_file_location('v184_source_metrics',B.ROOT/'minifly_v183_h_credit_dev_20261102/metrics.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
score=M.score;pack=M.pack;partitions=M.partitions;generate=M.generate;forecast_from_summaries=M.forecast_from_summaries;forecast_loss=M.forecast_loss
T=B.T;np=B.np

def alignment(net,records,codes):
 before=B.statehash(net);hs=[];us=[];labels=[]
 for r,code in zip(records,codes):
  x=T.zeros((len(code),5177))
  for i,row in enumerate(code):x[i,T.tensor(row.astype(np.int64))]=1
  logits,cache=net.forward(x);hs.append(cache[2]);us.append(cache[3].clamp_min(0));labels.append([*r,256])
 mapping=derangement([len(r)+1 for r in records],[B.hashlib.sha256(r).hexdigest() for r in records]);sums={'aligned':np.zeros(len(records)),'exchanged':np.zeros(len(records))};counts=np.zeros(len(records),np.int64);site_rows=[]
 # One evaluation per eligible site/route; fixed T/O and no dropout at inference.
 for ri in range(len(records)):
  positions=[pos for pos in range(len(labels[ri])) if (ri,pos) in mapping]
  if not positions:continue
  h=hs[ri][positions];other=T.stack([hs[mapping[(ri,pos)]][pos] for pos in positions]);u=us[ri][positions];y=T.tensor([labels[ri][pos] for pos in positions]);counts[ri]=len(y)
  local_logs={}
  for role,skip in [('aligned',h),('exchanged',other)]:
   logits=(u+net.alpha*skip)@net.p['O'].T+net.p['Ob'];lp=logits.double().log_softmax(1);true_log=lp[T.arange(len(y)),y].numpy();local_logs[role]=true_log;sums[role][ri]=float((-true_log/np.log(2)).sum())
  for k,pos in enumerate(positions):site_rows.append({'record':ri,'position':pos,'other_record':mapping[(ri,pos)],'y':labels[ri][pos],'aligned_logp':float(local_logs['aligned'][k]),'exchanged_logp':float(local_logs['exchanged'][k])})
 assert B.statehash(net)==before and counts.sum()>0
 return {'eligible_sites':int(counts.sum()),'held_sites':sum(len(r)+1 for r in records),'aligned_bits':float(sums['aligned'].sum()/counts.sum()),'exchanged_bits':float(sums['exchanged'].sum()/counts.sum()),'per_record_aligned_bit_sums':sums['aligned'].tolist(),'per_record_exchanged_bit_sums':sums['exchanged'].tolist(),'per_record_eligible_counts':counts.tolist(),'mapping':[dict(record=r,position=p,other_record=o) for (r,p),o in sorted(mapping.items())],'site_evaluations':site_rows,'label_blind_hash_mapping':True,'unchanged_T_and_O':True,'no_training':True}
