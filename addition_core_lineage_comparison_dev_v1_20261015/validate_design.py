"""Static protocol/data/port checks only. No learner imports or trajectories."""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def split(world):
    rng = random.Random(world)
    diagonal = [(d, d) for d in range(10)]
    original = [(a, b) for a in range(10) for b in range(a+1, 10)]
    for trial in range(1, 10001):
        off = original.copy()
        rng.shuffle(off)
        held, new, old = off[:10], off[10:20], diagonal + off[20:]
        answers = {a+b for a,b in old}
        degrees = [sum(d in pair for pair in old) for d in range(10)]
        if answers == set(range(19)) and min(degrees) >= 3:
            return dict(old=old, new=new, held=held, rejection_attempts=trial)
    raise ValueError("pre-model data split failed")

def ordered(orbits):
    return sorted({p for a,b in orbits for p in ((a,b),(b,a))})

def validate():
    d = json.loads((HERE/"DESIGN.json").read_text())
    c = json.loads((HERE/"OUTPUT_CODEBOOK.json").read_text())
    versions=d["versions"]
    assert len(versions)==5
    assert [v["native_store_count"] for v in versions]==[16,16,32,0,0]
    assert len({v["id"] for v in versions})==5
    assert all("-C16" in v["id"] for v in versions)
    assert all((ROOT/p).is_file() for p in d["sources"].values())
    assert d["implementation"]["new_training_records"]==0
    assert not d["implementation"]["implemented"]
    words=c["words_by_class"]
    rng=random.Random(c["seed"])
    canonical=[w for w in range(65536) if w.bit_count()==8 and w<(w^65535)]
    selected=rng.sample(canonical,129)
    expected=[x for w in selected[:128] for x in (w,w^65535)]+[selected[128]]
    rng.shuffle(expected)
    assert words==expected and len(set(words))==257
    codes=[[1 if (w>>j)&1 else -1 for j in range(16)] for w in words]
    assert all(sum(row)==0 for row in codes)
    assert all(abs(sum(row[j] for row in codes))==1 for j in range(16))
    assert all({row[j] for row in codes}=={-1,1} for j in range(16))
    minimum=16
    for i,w in enumerate(words):
        logits=[16-2*(w^other).bit_count() for other in words]
        assert logits[i]==16 and max(range(257),key=logits.__getitem__)==i
        for j in range(i):
            minimum=min(minimum,(w^words[j]).bit_count())
    # Exact rational four-class identity, not a trained-model result.
    from fractions import Fraction
    k4=[[2*int(i==j)-1 for j in range(4)] for i in range(4)]
    mu=[sum(Fraction(row[j]) for row in k4)/4 for j in range(4)]
    for i in range(4):
        for j in range(4):
            assert (1-k4[i][j])/2==int(i!=j)
            assert (mu[j]-k4[i][j])/2 == Fraction(1,4)-int(i==j)
    values=[Fraction(1,3),Fraction(-2,5),Fraction(7,4),Fraction(-11,6)]
    centered=[v-sum(values)/4 for v in values]
    decoded=[sum(Fraction(c)*u for c,u in zip(row,centered))/2 for row in k4]
    assert len({decoded[i]-values[i] for i in range(4)})==1
    born_rng=random.Random(2026101443001)
    splits=[]
    for world in range(d["science"]["world_start"],d["science"]["world_end"]+1):
        s=split(world)
        assert s==split(world)
        sets=[set(map(tuple,s[key])) for key in ("old","new","held")]
        assert [len(x) for x in sets]==[35,10,10]
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        forms=[ordered(s[key]) for key in ("old","new","held")]
        assert list(map(len,forms))==[60,20,20]
        assert len(set().union(*[set(x) for x in forms]))==100
        assert all((a,b) not in forms[0]+forms[1] for a,b in forms[2])
        assert {(0,0),(0,1),(1,0)} <= set(forms[0])
        s.update(world=world,order_seed_old=(world<<8)+1,order_seed_new=(world<<8)+2,
                 shuffle_seed=(world<<8)+3,neural_birth_seed=born_rng.getrandbits(32))
        splits.append(s)
    n=len(splits); h=len(d["histories"]); v=len(versions)
    per=d["data"]["old_records"]+d["data"]["new_records"]
    assert n==16 and per==17408
    assert n*v==d["science"]["complete_world_version_jobs"]==80
    assert n*v*h==d["science"]["correlated_histories"]==240
    assert n*v*h*per==d["science"]["external_record_applications"]==4177920
    assert v*h*(128+32)==d["technical_plan"]["cycle2"]["record_applications"]==2400
    assert 2*v*h*per==d["technical_plan"]["cycle3"]["record_applications"]==522240
    plan={"schema":"PLANNED_ADDITION_DATA_SPLITS_V1","models_executed":0,
          "worlds":splits,"not_scientific_receipts":True}
    (HERE/"PLANNED_DATA_SPLITS.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
    result={"status":"PASS_STATIC_ONLY","source_paths_present":len(d["sources"]),
            "versions":v,"native_store_counts":[16,16,32,0,0],
            "output_code_self_decode":257,"output_code_min_hamming_distance":minimum,
            "four_class_opponent_and_centering_identity":True,
            "planned_data_splits":n,"complete_world_version_jobs":n*v,
            "correlated_histories":n*v*h,"science_record_applications":n*v*h*per,
            "technical_cycle2_record_applications":2400,
            "technical_cycle3_record_applications":522240,
            "new_training_records":0,"scientific_jobs":0,
            "native_forward_parity_tested":False,"recurrent_gradient_parity_tested":False,
            "three_cycles_completed":False,"runtime_measured":False,
            "warning":"Static arithmetic, codec and dataset checks are not implemented-core qualification or learning evidence."}
    (HERE/"STATIC_VALIDATION.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":
    validate()

