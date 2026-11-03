"""Pure saved-metric gate canaries: no input records, model evaluation or training."""
import bootstrap as B
import protocol as Q
import json,copy
from pathlib import Path

def main():
 a={}
 for arm in ('NORMAL','HRES','NORM_SHAM'):
  value=3. if arm=='HRES' else 3.2;a[arm]={}
  for stage in ('birth','A2','B3','EDIT2','NOEDIT2'):
   a[arm][stage]={p:{'bits':value,'EOS_NLL':1.,'EOS_Brier':.1,'premature_EOS_rate':0.} for p in ('A_held','B_held','EDIT_held')};a[arm][stage]['EDIT_held']['partitions']={p:{'bits':value,'new_old_log2_odds':.2 if stage=='EDIT2' else 0.} for p in ('SAME_KC','ALTERED_KC','CHANGED')}
 ci={arm:{'pass_upper_lt_zero':True} for arm in ('NORMAL','NORM_SHAM')};align={'aligned_bits':3.,'exchanged_bits':3.1}
 assert Q.decide(a,ci,3.1,4.4,align)['verdict']=='RETAIN_EXACT_PACKAGE_FOR_NEW_GUARD_REVIEW'
 assert Q.decide(a,ci,2.9,4.4,align)['verdict']=='REJECT_V184_EXACT_PACKAGE'
 wrong=copy.deepcopy(a);wrong['HRES']['EDIT2']['A_held']['bits']=100.;assert Q.decide(wrong,ci,3.1,4.4,align)['verdict']=='RETAIN_EXACT_PACKAGE_FOR_NEW_GUARD_REVIEW'
 wrong=copy.deepcopy(a);wrong['HRES']['EDIT2']['EDIT_held']['EOS_Brier']=.2;assert Q.decide(wrong,ci,3.1,4.4,align)['verdict']=='REJECT_V184_EXACT_PACKAGE'
 assert Q.decide(a,ci,3.1,4.4,{'aligned_bits':3.1,'exchanged_bits':3.})['verdict']=='REJECT_V184_EXACT_PACKAGE'
 wrong=copy.deepcopy(a);wrong['NORMAL']['A2']['A_held']['bits']=2.9;assert Q.decide(wrong,ci,3.1,4.4,align)['verdict']=='REJECT_V184_EXACT_PACKAGE'
 p=B.H/'REDUCER_REVIEW.json';p.write_text(json.dumps({'status':'PASS','mode':'EXECUTED_SAME_FAMILY_SELF_AUDIT','pure_metadata_gate_canaries':6,'new_input_record_presentations':0,'new_supervised_target_evaluations':0,'new_model_predictions_or_updates':0,'cases':['positive full-vector','mean shortcut failure','obsolete original A afterC excluded','EOS Brier false-positive veto','H alignment reversal veto','claimed CI cannot override wrong-signed actual point loss'],'formal_MRK1':'UNAVAILABLE','independent_cross_family':'PENDING'},indent=2)+'\n');print('SAVED_GATE_REVIEW_PASS')
if __name__=='__main__':main()
