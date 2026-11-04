"""V214 technical driver rehearsal. The official Round 20 backend is absent."""
import json
from pathlib import Path
import random
import time
from zipfile import ZipFile,ZIP_DEFLATED

from protocol import ARMS,CHECKPOINTS,ByteCounter,efficiency,score,sha

OUT=Path("rehearsal")

def seed(label):
    return int(sha(label.encode())[:16],16)

def fixture(seed_value):
    r=random.Random(seed_value)
    history=[]
    for i in range(24):
        names=[bytes(r.randrange(97,123) for _ in range(6)) for _ in range(3)]
        history.append(b"person "+names[0]+b" in "+names[1]+b".\nobject "+names[2]+b".\n")
    place=bytes(r.randrange(97,123) for _ in range(7))+b"|"
    target=b"Where is the person?|"+place+b"Where is the object?|"+place
    a=target.index(place)
    b=target.rindex(place)
    return history,target,((a,a+len(place),"person"),(b,b+len(place),"object"))

def family(history,target,spans):
    arms={}
    for arm in ARMS:
        m=ByteCounter()
        totals=[]
        hist_bytes=0
        cells={}
        for k in range(25):
            if k in CHECKPOINTS:
                saved=m.snapshot()
                cloned=ByteCounter.restore(saved)
                r=score(cloned,target,spans)
                assert saved==m.snapshot() and cloned.total==m.total+len(target)
                assert not r.pop("predictions")
                cells[k]={**r,"checkpoint_sha256":sha(saved),"history_bytes":hist_bytes,
                          "history_bits":sum(totals)}
            if k<24:
                totals.append(score(m,history[k])["bits"])
                hist_bytes+=len(history[k])
        for r in cells.values():
            r.update(efficiency(cells[0]["bits"],r["bits"],r["history_bytes"],r["history_bits"]))
        arms[arm]=cells
    assert all(a==arms["BASE"] for a in arms.values())
    return arms

def write(name,obj):
    p=OUT/name
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    return sha(p.read_bytes())

def main():
    started=time.process_time()
    OUT.mkdir(exist_ok=False)
    roster=[seed(f"MINIFLY|V214|FAMILY|{i}") for i in range(16)]
    engineering=[seed(f"V214|TECH|{i}") for i in range(2)]
    assert len(set(roster+engineering))==18
    sources={p:sha(Path(p).read_bytes()) for p in (
        "DESIGN.md","parent_gate.py","protocol.py","test_protocol.py","rehearse.py")}
    refs={}
    physical=charged=0
    for i,key in enumerate(engineering):
        hist,target,spans=fixture(key)
        entry=family(hist,target,spans)
        h=sum(map(len,hist))
        physical+=h+len(target)
        charged+=len(ARMS)*(h+len(CHECKPOINTS)*len(target))
        name=f"TECH_{i}.json"
        refs[name]=write(name,dict(seed=key,backend="BYTE_COUNTER_TEST_DOUBLE",
                                  history_sha256=sha(b"".join(hist)),arms=entry))
    prose=[]
    for i in range(8):
        passage=f"Fixture prose line {i}. This is synthetic.\n".encode()
        rows={a:score(ByteCounter(),passage) for a in ARMS}
        assert all(rows[a]==rows["BASE"] for a in ARMS)
        prose.append(dict(slot=i,fixture_sha256=sha(passage),scores=rows))
        physical+=len(passage)
        charged+=len(ARMS)*len(passage)
    refs["PROSE_FIXTURE.json"]=write("PROSE_FIXTURE.json",prose)
    assert all(sha((OUT/p).read_bytes())==h for p,h in refs.items())
    upper=16*3*24*8192+16*3*4*8192+8*3*16384
    assert upper==11403264 and upper+charged<15500000
    write("CYCLE3_TEST.json",dict(status="DRIVER_PASS_PARENT_UNQUALIFIED",official_family_seeds=roster,
        official_streams_generated=0,official_worlds_opened=0,parent_byte_applications=0,
        fixture_physical_bytes=physical,fixture_charged_bytes=charged,official_upper_bytes_before_refs=upper,
        receipts=refs,source_sha256=sources,fixture_cpu_seconds=time.process_time()-started))
    paths=sorted(OUT.glob("*.json"))+[Path("CYCLE1_TEST.json"),Path("CYCLE2_TEST.json")]
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip","w",ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.name)
    with ZipFile(OUT/"TECHNICAL_BUNDLE.zip") as z:
        assert z.testzip() is None and all(z.read(p.name)==p.read_bytes() for p in paths)
    print(dict(fixture_physical_bytes=physical,fixture_charged_bytes=charged,
               bundle_sha256=sha((OUT/"TECHNICAL_BUNDLE.zip").read_bytes())))

if __name__=="__main__":
    main()
