"""Resource model and causal recitation-control fixtures."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import resource
import time

from qualify import Learner, T, np, DecaySensor, teach
from generation import generate
from profile_roster import load_definitions

class PrefixControl:
    def __init__(self, fallback):
        self.counts = defaultdict(Counter)
        self.fallback = fallback

    def fit(self, records):
        for record in records:
            for index,target in enumerate([*record,256]):
                self.counts[record[:index]][target] += 1
        return self

    def probability(self, prefix, target):
        fallback = self.fallback(prefix,target)
        counts = self.counts.get(prefix)
        if counts is None:
            return fallback
        return (counts[target]+fallback)/(sum(counts.values())+1)

def main():
    started = time.process_time()
    control = PrefixControl(lambda _prefix,_target:1/257).fit([b'ab',b'ab',b'ac'])
    assert control.probability(b'ab',256)>0.5
    assert control.probability(b'unknown',256)==1/257
    for prefix in (b'',b'a',b'ab',b'unknown'):
        assert math.isclose(sum(control.probability(prefix,i) for i in range(257)),1)
    sensor = DecaySensor()
    timings={}
    checkpoint_sizes=[]
    for batch in (32,256):
        model = Learner()
        t=time.process_time()
        targets=teach(model,sensor,.5,batch,[bytes(range(127))]*8)
        timings[batch]={'combined_encode_and_update_cpu_seconds':time.process_time()-t,
                        'targets':targets,'updates':model.clock}
    t=time.process_time()
    generated=0
    def force_payload(code):
        logits,_ = model.forward(T.from_numpy(code[None,:]))
        return int(logits[0,:256].argmax())
    for _ in range(32):
        payload,eos=generate(force_payload,sensor,b'abc',.5)
        assert not eos and len(payload)==64
        generated+=len(payload)
    decode_cpu=time.process_time()-t
    x=T.zeros((256,5177))
    x[:,::32]=1
    t=time.process_time()
    for _ in range(50):
        logits,_=model.forward(x)
        assert T.isfinite(logits.double().log_softmax(1)).all()
    score_cpu=time.process_time()-t
    profile=json.loads(Path('ROSTER_PROFILE.json').read_text())
    roster=profile['roster']
    projection={
        'encode_and_train':roster['targets']*max(v['combined_encode_and_update_cpu_seconds']/v['targets'] for v in timings.values()),
        'generation':roster['generation_sites_max']*decode_cpu/generated,
        'score_tensor_only':roster['score_sites']*score_cpu/(50*256),
        'baseline_and_preflight':profile['profile_cpu_seconds'],
        'local_fixture':time.process_time()-started+1.392999,
    }
    report={'status':'RESOURCE_PROJECTION_ONLY','prefix_fixture':'PASS','cpu_component_estimates':projection,
            'subtotal_cpu_seconds':sum(projection.values()),'cpu_cap':900,
            'omitted_from_subtotal':['score packing and output materialization','filesystem and final package audit'],
            'timings':timings,'additional_synthetic_records':16,'additional_synthetic_targets':2048,
            'fixture_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    Path('RESOURCE_PROJECTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)
if __name__=='__main__':
    main()
