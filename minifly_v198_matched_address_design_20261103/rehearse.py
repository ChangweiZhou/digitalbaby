"""V198 three-arm technical rehearsal and full-roster census."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from address import ARMS, codes
from source import PARENT, parent

OUT = Path("rehearsal")

def roster(world, fixture):
    return parent.roster(parent.seed(f"MINIFLY|V198|WORLD|{world}"), fixture)

def census(data):
    records = targets = marked = 0
    for _, _, _, stream in data:
        for phase in stream:
            for prefix, value, _ in phase:
                records += 1
                marked += value is not None
                targets += len(prefix) + 1 + (value is not None)
                assert len(prefix) + (value is not None) <= 64
    return {key: n * len(ARMS) for key,n in
            dict(records=records, targets=targets, marked=marked).items()}

def run():
    OUT.mkdir(exist_ok=False)
    full = [roster(i, False) for i in range(4)]
    totals = census(full)
    assert totals["records"] == 74880 and totals["marked"] == 59904
    fixture = roster(0, True)
    keys, values, edits, stream = fixture
    size = len(keys)//8
    prompts = [[form(key) for key in keys] for form in parent.FORMS]
    taught = {p for phase in stream for p,y,_ in phase if y is not None}
    assert all(p not in taught for group in prompts[2:] for p in group)
    code = {p: codes(p)[0] for group in prompts for p in group}
    by_prefix = {p: i for group in prompts[:2] for i,p in enumerate(group)}
    assert len(by_prefix) == 2 * len(keys)
    predictor = parent.Predictor()
    models = {a: parent.Memory("BASE", 198) for a in ARMS}
    stages = {a: [] for a in ARMS}
    observed = {}
    frequency = [0] * 256
    row_traffic = {a: 0 for a in ARMS}
    for wave, phase in enumerate(stream):
        for p, y, edit in phase:
            predictor.observe(p if y is None else p+bytes([y]), y is None)
            if y is not None:
                observed[by_prefix[p]] = y
                frequency[y] += 1
                for arm, model in models.items():
                    rows, features = code[p][arm]
                    model.teach(rows, features, y, edit)
                    row_traffic[arm] += len(rows)
        assert len(set(row_traffic.values())) == 1
        for arm, model in models.items():
            before = model.state_hash()
            items = []
            for wrapper, group in enumerate(prompts):
                for i,p in enumerate(group[:(wave+1)*size]):
                    corrected = i in edits and i//size < wave
                    target = edits[i] if corrected else values[i]
                    assert observed[i] == target
                    payload,eos = model.decode(*code[p][arm])
                    items.append(dict(wrapper=wrapper, item=i, cohort=i//size if corrected else None,
                                      exact=eos and payload==bytes([target]),
                                      byte=payload[:1]==bytes([target]),
                                      obsolete=corrected and payload[:1]==bytes([values[i]]),
                                      obsolete_complete=corrected and eos and payload==bytes([values[i]]),
                                      early=eos and not payload, truncated=not eos, extra=len(payload)>1,
                                      canonical_correct=observed[i]==target,
                                      exact_table_correct=p in taught,
                                      modal_correct=frequency.index(max(frequency))==target))
            assert model.state_hash() == before
            stages[arm].append(dict(items=items, state_sha256=before))
    hashes = {}
    decline = {}
    for arm,model in models.items():
        assert model.records == census([fixture])["marked"]//len(ARMS)
        path = OUT/f"{arm}_RECEIPT.json"
        parent.write(path,dict(arm=arm, stages=stages[arm], records=model.records, targets=model.targets,
                               motion=model.movement, fact_rows_read_and_written_each=row_traffic[arm]))
        hashes[path.name] = parent.sha(path.read_bytes())
        receipt = json.loads(path.read_bytes())
        final = {(r["wrapper"],r["item"]):r for r in receipt["stages"][-1]["items"]}
        transitions = []
        for cohort in range(7):
            own = [r for r in receipt["stages"][cohort+1]["items"] if r["wrapper"]>=2 and r["cohort"]==cohort]
            assert len(own)==size//2
            transitions.extend((r["exact"],final[r["wrapper"],r["item"]]["exact"]) for r in own)
        decline[arm] = dict(n=len(transitions), own=sum(x for x,_ in transitions),
                            final=sum(y for _,y in transitions), lost=sum(x and not y for x,y in transitions),
                            gained=sum(y and not x for x,y in transitions))
    lock = {str(p.name):parent.sha(p.read_bytes()) for p in
            (Path(__file__), Path("address.py"), Path("source.py"), Path("DESIGN.md"), PARENT, parent.PARENT)}
    summary = dict(status="TECHNICAL_PASS_BUDGET_BLOCKED", technical=census([fixture]), full_dry=totals,
                   full_roster_sha256=[parent.sha(repr(world).encode()) for world in full],
                   bridge_attempt_records=9216, source_lock=lock, receipts=hashes, technical_flips=decline,
                   resource=parent.resource_check(OUT))
    parent.write(OUT/"CYCLE3_TEST.json", summary)
    bundle = OUT/"TECHNICAL_BUNDLE.zip"
    with ZipFile(bundle,"w",ZIP_DEFLATED) as z:
        for p in OUT.glob("*.json"):
            z.write(p,p.name)
    with ZipFile(bundle) as z:
        assert z.testzip() is None
        for name, digest in hashes.items():
            assert parent.sha(z.read(name)) == digest
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    run()
