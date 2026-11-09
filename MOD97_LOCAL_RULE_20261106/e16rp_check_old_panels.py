# E16-RP: verifies that e16rp_same_weights.old_panels() rebuilds E16-R3's panels: scores the births on them and compares with the *_oldpanels receipts' stage-0 compose counts (result 9 Nov 14:50 UTC: identical in all three seeds).
import sys, json, torch as T
sys.path.insert(0, '/home/user/digitalbaby/MOD97_LOCAL_RULE_20261106')
import e16rp_same_weights as W
H = W.H
for s in (73129, 93683, 116089):
    m = H.E.new_net(s); m.load_state_dict(T.load(f'exp45/e11_dev_M_{s}.pt'))
    oc, _, _ = W.old_panels(s)
    got = {n: (H.score(m, it)['final'], H.score(m, it)['exact']) for n, it in oc.items()}
    st0 = json.load(open(f'exp45/e16_life_T_raw65536_{s}_oldpanels.json'))['stages'][0]['scores']['compose']
    ref = {n: (v['final'], v['exact']) for n, v in st0.items()}
    print(s, 'identical' if got == ref else 'DIFFER', got if got != ref else '', ref if got != ref else '')
