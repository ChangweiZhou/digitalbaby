"""Run recovered code, preserving its behavior and adding read-only per-query diagnostics."""
import argparse, contextlib, hashlib, io, json, os, platform, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def evaluate(source,stream,mode='full',testfrom=0,span=1500):
    import sys
    original_argv=sys.argv[:]
    previous={key:os.environ.get(key) for key in ['TESTFROM','SPAN']}
    os.environ.update(TESTFROM=str(testfrom),SPAN=str(span))
    sys.argv=[str(source),str(stream),mode]
    src=Path(source).read_text()
    # Hook runs after prediction, before any feedback-driven updates.
    needle="        # learning from the observed answer token (part of the stream)"
    assert src.count(needle)==1
    src=src.replace(needle,"""        AUDIT.append({'line':AUDIT_LINE,'kind':kind,'gold':a,'prediction':top,
            'correct':top==a,'candidate_available':a in c,'raw_probability':pv,
            'raw_bits':-math.log2(max(pv,1e-6)),
            'empty_fallback':not recentp,'candidate_count':len(sc),
            'raw_total_mass':(0.85 if Z else 0)+0.15,
            'op_evidence':{op:[sig,OPREL[(sig,op)][:],v] for v,lst in c.items() for r,op,sig in lst}})
"""+needle)
    src=src.replace('for ln in lines:\n','for AUDIT_LINE,ln in enumerate(lines,1):\n')
    namespace={'__name__':'__recovered__','AUDIT':[]}
    stdout=io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout): exec(compile(src,str(source),'exec'),namespace)
    finally:
        sys.argv=original_argv
        for key,value in previous.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
    trace=namespace['AUDIT']
    scored=[r for r in trace if r['line']>=testfrom]
    def stats(rows):
        return {kind:{'n':len(v),'top1':sum(r['correct'] for r in v)/len(v),
                      'raw_answer_bits':sum(r['raw_bits'] for r in v)/len(v),
                      'candidate_coverage':sum(r['candidate_available'] for r in v)/len(v),
                      'gold_fallback_count':sum(r['empty_fallback'] for r in v)}
                for kind in ['per','obj'] if (v:=[r for r in rows if r['kind']==kind])}
    # Recovered Round 23 splits by 45% of line count, not 50% of questions.
    split=len(namespace['lines'])*0.45
    result={'native':json.loads(stdout.getvalue().splitlines()[0]),'metrics':stats(scored),
            'halves':{name:stats([r for r in scored if (r['line']>split)==second])
                      for name,second in [('first45pct',False),('last55pct',True)]},
            'total_questions':len(trace),'scored_questions':len(scored),
            'learned_operations':[{'signature':sg,'op':op,'hits':h,'trials':n}
                  for (sg,op),(h,n) in sorted(namespace['OPREL'].items())],
            'diagnostics':{'empty_fallback_count':sum(r['empty_fallback'] for r in scored),
                           'subnormalized_count':sum(r['raw_total_mass']<0.999 for r in scored)}}
    return result,trace

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();spec=json.loads(Path(a.spec).read_text())
    result,trace=evaluate(ROOT/spec['source'],ROOT/spec['stream'],spec['mode'],spec.get('testfrom',0),spec.get('span',1500))
    result['spec']=spec
    result['receipts']={'source_sha256':hashlib.sha256((ROOT/spec['source']).read_bytes()).hexdigest(), 'stream_sha256':hashlib.sha256((ROOT/spec['stream']).read_bytes()).hexdigest(), 'python_hash_seed':os.environ.get('PYTHONHASHSEED'), 'python':sys.version}
    out=Path(a.out);out.write_text(json.dumps(result,indent=2)+'\n')
    out.with_suffix('.trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
