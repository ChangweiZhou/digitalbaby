"""Frozen label-blind alignment and paired-record bootstrap, no learning."""
import bootstrap as B
np=B.np

def derangement(record_lengths,record_hashes):
 # Cyclic successor in a fixed hash ordering; no target bytes or EOS status.
 mapping={}
 for pos in range(max(record_lengths)):
  eligible=[r for r,n in enumerate(record_lengths) if n>pos]
  if len(eligible)<2:continue
  ordered=sorted(eligible,key=lambda r:B.hashlib.sha256(f'V184|M1|A2|H-align|{pos}|{record_hashes[r]}|{r}'.encode()).hexdigest())
  for i,r in enumerate(ordered):mapping[(r,pos)]=ordered[(i+1)%len(ordered)]
 assert all(r!=other and record_lengths[other]>pos for (r,pos),other in mapping.items())
 return mapping

def paired_bootstrap(candidate_sums,control_sums,counts):
 a=np.asarray(candidate_sums);b=np.asarray(control_sums);n=np.asarray(counts);assert a.shape==b.shape==n.shape and len(a)==64
 rng=np.random.default_rng(B.P.seed('V184|M1|A2|paired-bootstrap|10000'));diff=[]
 for _ in range(10):
  ix=rng.integers(0,64,size=(1000,64));diff.extend(((a[ix]-b[ix]).sum(1)/n[ix].sum(1)).tolist())
 low,high=np.quantile(diff,[.025,.975],method='linear')
 return {'candidate_minus_control_pooled_bits':float((a-b).sum()/n.sum()),'CI95_low':float(low),'CI95_high':float(high),'replicates':10000,'unit':'paired record, not token or ecology','method':'percentile; pooled site denominator each replicate','pass_upper_lt_zero':bool(high<0)}

def decide(arms,primary_CIs,mean_reference,unigram,alignment):
 c=arms['HRES'];checks={};primary=c['A2']['A_held']['bits']
 for arm in ('NORMAL','NORM_SHAM'):checks['A2_CI_'+arm]=primary_CIs[arm]['pass_upper_lt_zero'] and primary<arms[arm]['A2']['A_held']['bits']
 checks['A2_below_own_mean']=primary<mean_reference;checks['A2_below_unigram']=primary<unigram;checks['H_alignment_eligible_only']=alignment['aligned_bits']<alignment['exchanged_bits']
 for arm in ('NORMAL','NORM_SHAM'):
  other=arms[arm]
  for stage,panel in [('B3','A_held'),('B3','B_held'),('EDIT2','B_held')]:checks[f'{stage}_{panel}_nonpositive_{arm}']=c[stage][panel]['bits']<=other[stage][panel]['bits']
  for p in ('SAME_KC','ALTERED_KC'):checks[f'EDIT2_{p}_nonpositive_{arm}']=c['EDIT2']['EDIT_held']['partitions'][p]['bits']<=other['EDIT2']['EDIT_held']['partitions'][p]['bits']
  for stage,panels in [('A2',('A_held','B_held')),('B3',('A_held','B_held')),('EDIT2',('EDIT_held','B_held'))]:
   for panel in panels:
    for metric in ('EOS_NLL','EOS_Brier','premature_EOS_rate'):checks[f'{stage}_{panel}_{metric}_{arm}']=c[stage][panel][metric]<=other[stage][panel][metric]
 checks['changed_label_log_odds_feedback_effect']=c['EDIT2']['EDIT_held']['partitions']['CHANGED']['new_old_log2_odds']>c['NOEDIT2']['EDIT_held']['partitions']['CHANGED']['new_old_log2_odds']
 return {'verdict':'RETAIN_EXACT_PACKAGE_FOR_NEW_GUARD_REVIEW' if all(checks.values()) else 'REJECT_V184_EXACT_PACKAGE','checks':checks,'payload_CIs_reported_separately_not_a_new_posthoc_gate':True,'record_CIs_not_ecology_replication':True,'core_adopted':False}
