"""Data-only preflight; no learner scores or efficacy-world optimizer updates."""
import util as U
import zipfile,importlib.util,collections,json,time
import numpy as np
WORLD=[('D1','austen-persuasion.txt','austen-sense.txt'),('D2','chesterton-brown.txt','melville-moby_dick.txt'),('G1','shakespeare-hamlet.txt','shakespeare-macbeth.txt'),('G2','milton-paradise.txt','bible-kjv.txt')]
CORP=U.ROOT/'minifly_v176_feature_deployment_dev_20261101/assets/source/minifly_v148_pretrain_probe_dev_20261025/corpus.py'
s=importlib.util.spec_from_file_location('literal_corpus',CORP);C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
letter=lambda y:65<=y<=90 or 97<=y<=122

def edit_rule(w,rs):
 freq=collections.Counter(b for r in rs for b in r if letter(b));counts=collections.defaultdict(collections.Counter)
 for r in rs:
  for t in range(2,len(r)):counts[r[t-2:t]][r[t]]+=1
 modal={k:min(v,key=lambda y:(-v[y],y)) for k,v in counts.items()};eligible=[k for k,v in counts.items() if sum(v.values())>=32 and letter(modal[k]) and v[modal[k]]/sum(v.values())>=.6];eligible.sort(key=lambda k:(-sum(counts[k].values()),U.digest(b'V180|'+w.encode()+b'|edit|'+k)));keys=eligible[:4];top=sorted(freq,key=lambda b:(-freq[b],b))[:16];rule={};used=set()
 for k in keys:
  old=modal[k];new=min([b for b in top if b!=old and b not in used],key=lambda b:U.digest(b'V180|'+w.encode()+b'|replacement|'+k+bytes([b])));rule[k]=(old,new);used.add(new)
 return rule,len(eligible),{k.hex():dict(total=sum(counts[k].values()),share=counts[k][modal[k]]/sum(counts[k].values())) for k in keys}

def correct(r,rule):
 out=bytearray(r);changes=np.zeros(len(r)+1,bool)
 for t in range(2,len(r)):
  if r[t-2:t] in rule and r[t]==rule[r[t-2:t]][0]:out[t]=rule[r[t-2:t]][1];changes[t]=True
 return bytes(out),changes

def main():
 archive=U.H/'local/gutenberg.zip';blob=archive.read_bytes();assert U.digest(blob)=='2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611';assets={};worlds={};all_data={};problems=[];outdir=U.H/'local/corpora';outdir.mkdir(exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  for w,an,bn in WORLD:
   sp={};wm={}
   for role,name in [('A',an),('B',bn)]:
    raw=z.read('gutenberg/'+name);(outdir/name).write_bytes(raw);rs=C.records(raw);cut=next((i for i,r in enumerate(rs) if r.startswith(b'End of Project Gutenberg')),len(rs));rs=rs[:cut];nt=512 if role=='A' else 128;ss=dict(train=rs[:nt],calib=rs[nt:nt+128],held=rs[-128:]);dups=set(ss['train'])&set(ss['held']);dupsca=set(ss['train'])&set(ss['calib']);assets[name]=dict(sha256=U.digest(raw),bytes=len(raw),records_after_footer_exclusion=len(rs));wm[role]=dict(counts={k:dict(records=len(v),targets=sum(len(r)+1 for r in v),sha256=U.digest(b''.join(len(r).to_bytes(4,'big')+r for r in v))) for k,v in ss.items()},train_held_exact_duplicate_record_types=len(dups),train_held_duplicate_held_records=sum(r in dups for r in ss['held']),train_calib_duplicate_types=len(dupsca),duplicate_hashes=[U.digest(x) for x in sorted(dups)])
    if len(rs)<nt+256:problems.append(dict(world=w,role=role,issue='INSUFFICIENT_DISJOINT_RECORD_OFFSETS'))
    if dups:problems.append(dict(world=w,role=role,issue='EXACT_TRAIN_HELD_DUPLICATES',records=sum(r in dups for r in ss['held'])))
    for k,v in ss.items():sp[role+'_'+k]=v
   rule,ne,stats=edit_rule(w,sp['A_train']);ids=[i for i,r in enumerate(sp['A_train']) if correct(r,rule)[1].any()];ids=sorted(ids,key=lambda i:U.digest(f'V180|{w}|revision-record|{i}'.encode()))[:64];changed=int(sum(int(correct(r,rule)[1].sum()) for r in sp['A_held']));same_label=sum(len(r)+1-int(correct(r,rule)[1].sum()) for r in sp['A_held'])
   wm['edit']=dict(eligible_contexts=ne,selected=[dict(context=k.hex(),old=v[0],new=v[1],**stats[k.hex()]) for k,v in rule.items()],revision_record_count=len(ids),revision_record_ids=ids,changed_held_targets=changed,unchanged_label_held_targets=same_label,source_only_preflight=True)
   if len(rule)!=4:problems.append(dict(world=w,issue='FEWER_THAN_FOUR_ELIGIBLE_LETTER_CONTEXTS',actual=len(rule)))
   if len(ids)!=64:problems.append(dict(world=w,issue='FEWER_THAN_64_EDITED_RECORDS',actual=len(ids)))
   if changed<24:problems.append(dict(world=w,issue='INSUFFICIENT_CHANGED_HELD_TARGETS',actual=changed))
   sp['EDIT_train']=[correct(sp['A_train'][i],rule)[0] for i in ids];sp['NOEDIT_train']=[sp['A_train'][i] for i in ids]
   for k in ('calib','held'):sp['EDIT_'+k]=[correct(r,rule)[0] for r in sp['A_'+k]]
   all_data[w]=sp;worlds[w]=wm
 import pickle
 with (U.H/'local/records.pkl').open('wb') as f:pickle.dump(all_data,f,protocol=5)
 U.save('DATA_AUDIT.json',dict(status='PASS_PENDING_SENSORY_MASKS' if not problems else 'D1_DATA_CONTRACT_FAILED',assets=assets,worlds=worlds,problems=problems,corpus_source=dict(repo='nltk/nltk_data',commit='550b6625bcef1f2abff2ff770a5a0d272c9c6b2a',archive_git_blob='f99e4d857456daefc5ae075d10bdde5f57083a9c',archive_sha256=U.digest(blob)),no_model_scoring=True,no_main_training=True));U.finish_process('data_only_authentication');print(json.dumps(dict(problems=problems,world_edits={w:m['edit'] for w,m in worlds.items()}),indent=2))
if __name__=='__main__':main()
