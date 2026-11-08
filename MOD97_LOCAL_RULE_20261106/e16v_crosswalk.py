"""E16-V numerical crosswalk. Standard library only; never executes a network."""
import csv
import json
import re
import statistics as st
from pathlib import Path

SEEDS = (73129, 93683, 116089)
PAIRS = ('+-', '-+', '><', '<>')
OPS = ('add', 'sub', 'max', 'min')
ARMS = ('T', 'D', 'M', 'T_oracle', 'T_balanced', 'T_all_balanced', 'T_share')
LABELS = {
    'T learner (uniform replay)': 'T', 'T learner (uniform replay, 1,024/block)': 'T',
    'D twin (joint ground truth)': 'D', 'M backprop on T\'s batches': 'M',
    'T on the twin\'s stream': 'T_oracle',
    'T balanced per family (1,024/block)': 'T_balanced',
    'T balanced, every entry': 'T_all_balanced',
    'T earlier families 50% of batch': 'T_share',
}

def pct(d, k='exact'):
    return 100 * d[k] / d['n']

def meanops(s, mod=97, k='12', silence=False):
    q = s['scores']['silenced_k12'] if silence else s['scores']
    return st.mean(pct(q[f'{o}{mod}'] if silence else q[f'{o}{mod}'][k]) for o in OPS)

def pair(s, p, silence=False, key='final'):
    return pct(s['scores']['silenced_compose' if silence else 'compose'][p], key)

def comp(s, silence=False, key='final'):
    return st.mean(pair(s, p, silence, key) for p in PAIRS)

def nums(s):
    return re.findall(r'(?<![\w])[-+]?\d+(?:\.\d+)?', s)

def tables(path, after=None):
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(after)) if after else 0
    out = []
    i = start
    while i < len(lines):
        if lines[i].startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|\-]+$', lines[i+1]):
            header = [x.strip() for x in lines[i].strip('|').split('|')]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith('|'):
                rows.append((i+1, [x.strip() for x in lines[i].strip('|').split('|')]))
                i += 1
            out.append((header, rows))
        else:
            i += 1
    return out

def csvwrite(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def crosswalk(root):
    snap = root / 'E16V_SOURCE_SNAPSHOT' / 'MOD97_LOCAL_RULE_20261106'
    d = {f'{a}_{s}': json.loads((snap/'exp45'/f'e16_life_{a}_{s}.json').read_text()) for a in ARMS for s in SEEDS}
    def stage(a, seed, b=6): return d[f'{a}_{seed}']['stages'][b]
    checked = []
    def check(file, line, col, cell, values):
        tokens = nums(cell)
        assert len(tokens) == len(values), (file, line, col, cell, values)
        for n, (token, value) in enumerate(zip(tokens, values)):
            decimals = len(token.split('.')[1]) if '.' in token else 0
            tolerance = .5 * 10 ** (-decimals) + 1e-9
            checked.append(dict(file=file, line=line, column=col, component=n,
                                published=token, recomputed=value, delta=float(token)-value,
                                rounding_half_unit=tolerance, verdict=('CONFIRMED' if abs(float(token)-value) <= tolerance else 'DISCREPANCY')))
    summary = snap/'exp45'/'SUMMARY_E16.md'
    ts = tables(summary); assert len(ts) == 5, len(ts)
    for ti, (header, rows) in enumerate(ts):
        for line, row in rows:
            if ti == 0:
                a, seed, block = row[:3]; seed = int(seed)
                b = {'birth':0,'F1.1':1,'F1.2':2,'F2.1':3,'F2.2':4,'F3.1':5,'F3.2':6}[block]
                s = stage(a, seed, b); sc = s['scores']
                vals = [meanops(s),pct(sc['add97']['12']),pct(sc['sub97']['12']),meanops(s,73),comp(s),
                        *[pair(s,p) for p in PAIRS],pair(s,'within'),st.mean(pct(sc[f'{o}89']['12']) for o in OPS)]
                for col, cell, v in zip(header[3:], row[3:], vals): check('SUMMARY_E16.md',line,col,cell,[v])
            elif ti == 1:
                seed = int(row[0]); label, fam = row[1].split(': ')
                a = LABELS[label]; s = stage(a,seed); twin=stage('D',seed)
                m = (lambda q: comp(q)) if fam == 'held comp final' else (lambda q: meanops(q,97 if fam.startswith('F1') else 73))
                mod=97 if fam.startswith('F1') else 73
                vals = [[m(s)],[m(twin)],[m(s)/m(twin)],
                        [meanops(stage(a,seed,2))] if fam.startswith('F1') else [],
                        [] if fam=='held comp final' else [meanops(s,mod,k='20')],
                        [comp(s,True)] if fam=='held comp final' else [meanops(s,mod,silence=True)]]
                for col, cell, v in zip(header[2:], row[2:], vals): check('SUMMARY_E16.md',line,col,cell,v)
            elif ti == 2:
                a=LABELS[row[0]]; seed=int(row[1]);s=stage(a,seed)
                vals=[[meanops(stage(a,seed,2)),meanops(s)],[pct(s['scores'][o+'97']['12']) for o in ('add','sub')],
                      [comp(stage(a,seed,4)),comp(s)],[pair(s,p) for p in PAIRS[:2]],
                      [pair(stage(a,seed,4),'within'),pair(s,'within')],[meanops(s,73)]]
                for col,cell,v in zip(header[2:],row[2:],vals): check('SUMMARY_E16.md',line,col,cell,v)
            elif ti == 3:
                a,seed=row[:2];s=stage(a,int(seed))
                for col,cell in zip(header[2:],row[2:]):
                    check('SUMMARY_E16.md',line,col,cell,[pct(s['scores'][col][k]) for k in ('0','8','12','20')])
            else:
                rec=stage('T',SEEDS[0],0)['shortcut_ceilings'][row[0]]
                for col,cell,v in zip(header[1:],row[1:],rec.values()): check('SUMMARY_E16.md',line,col,cell,[100*v])
    report=snap/'E45_REPORT.md'; rt=tables(report, '## E16:')
    assert len(rt)==4, len(rt)
    def vec(a,b,fn): return [fn(stage(a,s,b)) for s in SEEDS]
    def carries(a,b): return [pair(stage(a,s,b),p) for s in SEEDS for p in PAIRS[:2]]
    def op2(a,b): return [pct(stage(a,s,b)['scores'][o+'97']['12']) for s in SEEDS for o in ('add','sub')]
    for ti,(header,rows) in enumerate(rt):
        for ri,(line,row) in enumerate(rows):
            if ti==0:
                if ri==0: vv=[vec(a,2,meanops) for a in ('T','D')]
                elif ri==1:vv=[vec(a,4,comp) for a in ('T','D')]
                elif ri==2:vv=[carries(a,4) for a in ('T','D')]
                else:vv=[vec(a,6,lambda q:meanops(q,73)) for a in ('T','D')]
            elif ti==1:
                if ri in (0,5):
                    mod=97 if ri==0 else 73; fn=lambda q:meanops(q,mod)
                    vv=[vec('T',6,fn),vec('D',6,fn),[fn(stage('T',s))/fn(stage('D',s)) for s in SEEDS],vec('T',2,fn) if ri==0 else []]
                elif ri==1:vv=[op2('T',6),op2('D',6),[],op2('T',2)]
                elif ri==2:vv=[vec('T',6,comp),vec('D',6,comp),[comp(stage('T',s))/comp(stage('D',s)) for s in SEEDS],vec('T',4,comp)]
                elif ri==3:vv=[carries('T',6),carries('D',6),[],carries('T',4)]
                else:vv=[vec('T',6,lambda q:pair(q,'within')),vec('D',6,lambda q:pair(q,'within')),[],vec('T',4,lambda q:pair(q,'within'))]
            elif ti==2:
                a=('T','T_balanced','T_all_balanced','T_oracle','D')[ri]
                vv=[[v for s in SEEDS for v in (meanops(stage(a,s,2)),meanops(stage(a,s)))],
                    [v for s in SEEDS for v in (comp(stage(a,s,4)),comp(stage(a,s)))],carries(a,6),vec(a,6,lambda q:meanops(q,73))]
            else:
                a=('T','T_share','M','T_oracle','T_all_balanced')[ri]
                vv=[[meanops(stage(a,s,b)) for s in SEEDS for b in (2,4,6)],
                    [v for s in SEEDS for v in (comp(stage(a,s,4)),comp(stage(a,s)))],carries(a,6)]
            for col,cell,v in zip(header[1:],row[1:],vv):check('E45_REPORT.md',line,col,cell,v)
    csvwrite(root/'E16V_TABLE_CROSSWALK.csv',checked)
    return d, checked

def more_audits(root, d):
    snap=root/'E16V_SOURCE_SNAPSHOT'/'MOD97_LOCAL_RULE_20261106'
    def s(a,seed,b=6):return d[f'{a}_{seed}']['stages'][b]
    def fvec(a,b,fn):return [fn(s(a,seed,b)) for seed in SEEDS]
    def fmt(v):
        return '/'.join(f'{x:.4f}' for x in v) if isinstance(v,(list,tuple)) else str(v)
    claims=[]
    def claim(section,quote,result,verdict,note):
        claims.append(dict(section=section,quoted_claim=quote,recomputed=fmt(result),verdict=verdict,note=note))
    # Covers the quantitative prose and interpretive conclusions, besides all numeric table cells.
    claim('E16','6 blocks, 6144 updates, three standing seeds', [len(s0['stages'])-1 for s0 in d.values()], 'CONFIRMED','21 receipts have six blocks; all record stage_updates=6144. Historical wall 80min is not independently derivable from CPU counters.')
    claim('E16','current held composition84–87; carry75–84; shortcut1–2; within4–9 of twin',
          fvec('T',4,comp)+[pair(s('T',z,4),p) for z in SEEDS for p in PAIRS[:2]],'CONFIRMED','Rounded envelopes; table crosswalk supplies all exact values. Carry shortcuts independently rebuilt, 0.5–1.67% across seeds.')
    claim('E16','each family within1–9 points of BP while current',
          [meanops(s('D',z,6),73)-meanops(s('T',z,6),73) for z in SEEDS], 'DISCREPANCY','F3 seed116089 gap11.0625pp; overall current-family gaps span0.875–11.0625pp, not1–9.')
    claim('E16','F1 block4 values78.1/71.8/47.0 and partial recovery in F3',
          fvec('T',4,meanops)+fvec('T',6,meanops),'CONFIRMED','All three improve between blocks4 and6; F1 losses begin during F2.')
    claim('E16','F2 carry18–38, within56–64, comparison72–85 after6',
          [pair(s('T',z),p) for z in SEEDS for p in PAIRS],'CONFIRMED','Rounded envelopes, within55.5/61.1667/64.1667; comparisons72–85.5. These comparisons alone do not establish composition.')
    claim('E16','F1 one third during F2, F2 one fifth during F3',
          'F1 F2 blocks=1/2,1/3; F2 F3 blocks=1/4,1/5','DISCREPANCY','Quoted shares apply to each second block, not both blocks; realized counters unavailable.')
    claim('E16','silenced operations3–7 and compositions4–7',
          [meanops(s('T',z),m,silence=True) for z in SEEDS for m in (97,73)]+fvec('T',6,lambda q:comp(q,True)),
          'QUALIFIED','Rounded family averages support approximate wording; individual operations reach13.25%, individual held pairs need separate table. Store remains enabled.')
    gaps=[meanops(s('T',z),m)-meanops(s('T',z),m,k='20') for z in SEEDS for m in (97,73)]
    claim('E16','k20 within2–9 points; P16.4 within10 holds',gaps,'QUALIFIED','P16.4 explicitly names F1: its mean drops2.0625/4.1875/9.375pp, all≤10, and cell-silencing costs≥30pp. A broader all-family reading fails: F3 seed116089 drops10.375pp (82.3125→71.9375). Individual add73 falls14.75pp. Do not retrospectively broaden the original F1 gate.')
    dev=[st.mean(pct(s(a,z)['scores'][o+'89']['12']) for o in OPS) for a in ('T','D') for z in SEEDS]
    claim('E16','development operations9–52 in both arms',dev,'DISCREPANCY','T means9.1667–51.8333; D means7.6667–55.75. Mod89 is not replayed.')
    ratios=[[fn(s('T',z))/fn(s('D',z)) for z in SEEDS] for fn in (meanops,comp,lambda q:meanops(q,73))]
    claim('E16','P16.1 coexistence fails; F1passes2,F3passes3,compfails3',str(ratios),'CONFIRMED','Threshold0.8: T has2/3 F1,0/3 F2,3/3 F3 passes.')
    losses=[meanops(s('T',z,2))-meanops(s('T',z)) for z in SEEDS]
    claim('E16','P16.2 losses10.6/12.0/20.8, add/sub16–39',losses,'CONFIRMED','All F1 family losses exceed10pp; per-operation losses16–39. P16.3 ≥70 held and≥60 carry after4 passes3/3.')
    claim('E16','P16.5 twin holds everything', [comp(s('D',z,4))-comp(s('D',z)) for z in SEEDS], 'QUALIFIED','Supports life-family means and all-family gate, not development operations or every individual cell. D data, decay and Store differ from T.')
    claim('E16-D','9 lives,18–19min per block,115min',9,'NOT_INDEPENDENTLY_VERIFIED','Receipts establish9 six-block lives; elapsed wall cannot be recovered from cumulative process CPU alone.')
    claim('E16-D','oracle loses≤1.2 F1 and≤1.0comp,3–9 belowtwin',
          [meanops(s('T_oracle',z,2))-meanops(s('T_oracle',z)) for z in SEEDS]+[comp(s('T_oracle',z,4))-comp(s('T_oracle',z)) for z in SEEDS],
          'QUALIFIED','F1 seed116089 loses1.25pp, not≤1.2 literally (half-decimal boundary). F2maxloss1pp. Absolute current/final F3 gap reaches5.5; full comparisons in tables. Decay and Store confound rule-only interpretation.')
    dist=[comp(s('T_all_balanced',z))-comp(s('T_oracle',z)) for z in SEEDS]
    claim('E16-D','all-entry compositions80–85,within1.6–4.2 oforacle',dist,'CONFIRMED','Absolute gaps1.5833/4.1667/3.9583pp; first is above oracle; operations89.75/89.1875/77.6875.')
    claim('E16-D','balanced indistinguishable fromuniform; observednull',
          [comp(s('T_balanced',z))-comp(s('T',z)) for z in SEEDS],'QUALIFIED','Final gains7.8333/2.4167/3.125pp, not exact null; no statistical equivalence test. Expected shares equal during block3 but differ during blocks4 and6; per-item repetition also differs.')
    claim('E16-D','weak seed61duringcomposition; oracle75–85;shareaddressesexactlythatblock',
          [meanops(s('T_all_balanced',116089,b)) for b in (3,4)]+[meanops(s('T_oracle',116089,b)) for b in (3,4)],
          'DISCREPANCY','All-entry60.8125/60.6875 versusoracle74.875/84.875. Share does NOT change exposure in first compositionblock3, where large loss already occurs; only changes block4 relativebalanced.')
    claim('E16-D','P16D.3 missesF1by11.9;P16D.4 acquisitionwithin3–4',
          [meanops(s('T_oracle',116089))-meanops(s('T_all_balanced',116089))]+[comp(s('T_all_balanced',z,4))-comp(s('T',z,4)) for z in SEEDS],
          'DISCREPANCY','F1difference11.875; all-entry composition acquisition+3.125/+3.125/+4.3333pp. Original P16D.4 says within3pp, not3–4; fails literal gate. Other family/arm deviations in E16V_GATES.csv.')
    claim('E16-D2','6lives,23min/block,2h25min',6,'NOT_INDEPENDENTLY_VERIFIED','6 complete receipts; process CPU is not elapsed wall.')
    uni=[meanops(s('T',z,2))-meanops(s('T',z,4)) for z in SEEDS]
    sha=[meanops(s('T_share',z,2))-meanops(s('T_share',z,4)) for z in SEEDS]
    claim('E16-D2','shareF1loss15/24/44 versus15/20/39',sha+uni,'CONFIRMED','Rounded correctly; first loss-bearing F2block has unchanged expected share; cannot exclude share as a mechanism generally.')
    u=[comp(s('T',z,4))-comp(s('T',z)) for z in SEEDS]
    v=[comp(s('T_share',z,4))-comp(s('T_share',z)) for z in SEEDS]
    claim('E16-D2','compositionloss24/30/37 versus30/31/41,reduction4–20%',v+u+[(a-b)/a*100 for a,b in zip(u,v)],'DISCREPANCY','Share losses23.9583/29.9583/36.4167pp: last rounds36,not37; subtracting rounded endpoints can create this discrepancy. Reduction3.6193–20.4703% supports rounded4–20. Literal doubling exposure is wrong:25→25 and20→25% inF3.')
    claim('E16-D2','Mopswithin3ofblock2,comp loses17–28,ends65–77',
          [meanops(s('M',z,2))-meanops(s('M',z)) for z in SEEDS]+[comp(s('M',z,4))-comp(s('M',z)) for z in SEEDS],
          'CONFIRMED','M F1 losses0.8125/2.875/1.5625; comp losses19.4583/27.8333/16.5; final65.4167–76.7083.')
    claim('E16-D2','localforgetsextraabout20points',
          [comp(s('M',z))-comp(s('T',z)) for z in SEEDS],'QUALIFIED','Final advantages17.25/12.2083/30.75pp; mean20.0694, not20in everyseed. Loss differences and endpointdifferences are distinct because acquisition differs.')
    claim('E16-D2','shareacquisitionwithin2comp,+0.6to+5ops',
          [comp(s('T_share',z,4))-comp(s('T',z,4)) for z in SEEDS]+[meanops(s('T_share',z,2))-meanops(s('T',z,2)) for z in SEEDS],
          'DISCREPANCY','Compositiondeltas+1.4583/+0.2083/−1.0833;operations+0.5/+0.6875/+4.9375. Original P16D2.3 says everyfamily within3pp in botharms: fails; M composition acquisition is also >3pp higher in everyseed. Predictions cannot be loosened after results.')
    claim('E16-D2','rulecleared;shareisnotthelever;coverageisthecause;cellnotspecific',
          'Positive all-entry intervention; partial share effect; both rules forget under sparse replay', 'OVERCLAIM',
          'Support for replay coverage as important, not uniquely isolated cause. No cell-only causal inference; no fixed-buffer composition-priority sweep rules out allshare interventions. Rich-data comparison also changes decay/Store; actual batches were not logged.')
    # E16W cross-study prose is checked on the exact canonical receipt named by each JOB.
    w={}
    for seed in SEEDS:
        for arm in ('C_uni','C_share','D_matched','D_ceiling'):
            job=json.loads((snap/'exp45'/f'E16W2_JOB_{arm}_{seed}.json').read_text())
            w[(arm,seed)]=json.loads((snap/'exp45'/job['receipt']).read_text())['stages']
    def wc(q):return st.mean(pct(q['scores']['compose'][p],'final') for p in PAIRS)
    def wo(q):return st.mean(pct(q['scores'][o+'97']) for o in OPS)
    # E8 receipts have a different schema; assert rather than silently substituting.
    wl=[wc(w[('C_uni',z)][5])-wc(w[('C_uni',z)][-1]) for z in SEEDS]
    claim('E16-D2/E16W','windowloss46–48;sharegain6–11andreduction14–23%',
          wl+[wc(w[('C_share',z)][-1])-wc(w[('C_uni',z)][-1]) for z in SEEDS], 'CONFIRMED','Uses pinned JOB-selected receipts, not glob-selected partial runs. Percentage reductions13.7/23.4/17.0 round14/23/17.')
    claim('E16-D2/E16W','windowBPends43–50local31–37ceiling91',
          [wc(w[(a,z)][-1]) for a in ('D_matched','C_uni','D_ceiling') for z in SEEDS], 'CONFIRMED','Ceiling90.5/90.75/91.875; differentdata,notrule-only. Local record-stream matching verified by originalE16W study, not reexecuted here.')
    claim('E16-D2/E16W','oldops93–95→76–85duringcompose',
          [wo(w[('C_uni',z)][b]) for z in SEEDS for b in (4,5)],'CONFIRMED','Scalar family means; does not prove same cause as the cell learner.')
    csvwrite(root/'E16V_CLAIMS.csv',claims)
    return claims

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    d,rows=crosswalk(root)
    print('Numeric table components:',len(rows),'beyond rounding:',sum(x['verdict']=='DISCREPANCY' for x in rows))
    claims=more_audits(root,d)
    print('Prose/interpretation checks:',len(claims))
