"""Data-only coverage of record-distinct groups, no predictions or optimizer work."""
import util as U
import pickle,hashlib

def main():
 rec=pickle.loads((U.H/'local/records.pkl').read_bytes());audit=U.load('DATA_AUDIT.json');out={}
 for w,sp in rec.items():
  out[w]={}
  for phase,name in [('A','A_train'),('B','B_train'),('C','EDIT_train')]:
   role='A' if phase!='B' else 'B';filename=[x for x in audit['assets'] if x in {'D1':('austen-persuasion.txt','austen-sense.txt'),'D2':('chesterton-brown.txt','melville-moby_dick.txt'),'G1':('shakespeare-hamlet.txt','shakespeare-macbeth.txt'),'G2':('milton-paradise.txt','bible-kjv.txt')}[w]][0 if role=='A' else 1];sha=audit['assets'][filename]['sha256'];ids=audit['worlds'][w]['edit']['revision_record_ids'] if phase=='C' else list(range(len(sp[name])));sites=[]
   for i,r in zip(ids,sp[name]):sites += [sha+':'+str(i)]*(len(r)+1)
   bad=bad_targets=batches=0;hist={}
   for start in range(0,len(sites),256):
    block=sites[start:start+256];u=sorted(set(block),key=lambda v:hashlib.sha256(('V180|micro|'+v).encode()).hexdigest());reliable=min(len(u[::2]),len(u[1::2]))>=2;batches+=1;bad+=not reliable;bad_targets+=len(block)*(not reliable);hist[str(len(u))]=hist.get(str(len(u)),0)+1
   out[w][phase]=dict(batches_per_pass=batches,unreliable_batches_per_pass=bad,unreliable_batch_fraction=bad/batches,unreliable_target_fraction=bad_targets/len(sites),distinct_record_histogram=hist,source_targets_per_pass=len(sites))
 U.save('SPLIT_COVERAGE_AUDIT.json',dict(worlds=out,record_id_convention='raw corpus SHA256 + colon + original preprocessed record index; C retains original A IDs',C_order='frozen hash-selection list; identical EDIT/NOEDIT order; no subsequent shuffling',no_model_scores=True));U.finish_process('data_only_split_coverage');print(out)
if __name__=='__main__':main()
