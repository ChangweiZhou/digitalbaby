"""Apply the prespecified E3 and delayed-reuse gates after the A comparison."""
from hashlib import sha256
from random import Random
import time

from official_worked import (ARMS, BLOCKS, EOS, MODES, OUT, Model, T, digest, e3,
                             fixture, read, receipt, sequence, update, write)
from qualify_worked import alphabet_for, work


def state(block,mode,arm,law):
    directory=OUT/f"{block}-{mode}-{arm}"
    model=Model(mode,block)
    opt=T.optim.Adam(model.parameters(),lr=.003)
    checkpoint=T.load(directory/f"law{law}-8192.pt",weights_only=True)
    model.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["optimizer"])
    assert digest(model)==read(directory/f"law{law}-8192.json")["state"]
    return model,opt


def null_stream(bank,block,orientation,arm):
    rng=Random(f"{block}:A:stream")
    traces=[[work(p,op) for _,p,_ in bank] for op in alphabet_for("A")]
    for epoch in range(16):
        order=rng.sample(range(len(bank)),len(bank))
        # Consume the same RNG stream as the matched fixed-law history.
        base=order[len(bank)//2:]+order[:len(bank)//2]
        while any(bank[i][0]==bank[j][0] for i,j in zip(order,base)):
            rng.shuffle(base)
        assigned={i:((bank[i][0]%64>=32)+epoch+orientation)%2 for i in order}
        source={}
        pair_rng=Random(f"{block}:{epoch}:null-derangement")
        for law in range(2):
            keys=[i for i in order if assigned[i]==law]
            perm=keys[1:]+keys[:1]
            while any(bank[i][0]==bank[j][0] for i,j in zip(keys,perm)):
                pair_rng.shuffle(perm)
            source.update(zip(keys,perm))
        for i in order:
            law=assigned[i]
            _,p,ys=bank[i]
            trace=traces[law][i if arm=="worked" else source[i]]
            yield list(p+b"?")+[ys[law],EOS],list(p+b"W"+trace)+[EOS]


def pair_flips(reports):
    out=[]
    for a,b in zip(*reports):
        pairs=list(zip(a["raw"],b["raw"]))
        assert all(x["prefix"]==y["prefix"] and x["flip"]==y["flip"] for x,y in pairs)
        out.append(dict(n=sum(x["flip"] for x,y in pairs),
                        both=sum(x["flip"] and x["output"]==x["target"] and y["output"]==y["target"]
                                 for x,y in pairs)))
    return out


def main():
    start=time.process_time()
    lock=read(OUT/"LOCK.json")
    assert lock["sources"]==receipt()
    summary=read(OUT/"SUMMARY.json")
    winners=summary["E2_eligible"]
    if not winners:
        print("E2 gate closed. No conditional jobs.")
        return
    ceilings=lock["adaptive_ceiling"]
    candidates=[]
    for mode,arm in winners:
        raw={}
        margins=[]
        for block in BLOCKS:
            p=[e3(state(block,mode,arm,law)[0],block,law,"A") for law in range(2)]
            margins.extend(row["balanced_exact"]-ceiling for report in p for row,ceiling in zip(report,ceilings))
            raw[str(block)]=dict(by_law=p,law_flip_both=pair_flips(p))
        write(OUT/f"E3-{mode}-{arm}.json",raw)
        if min(margins)>=.05:
            candidates.append((min(margins),-MODES.index(mode),-ARMS.index(arm),mode,arm,raw))
    if not candidates:
        summary["status"]="COMPLETE_STOPPED_AFTER_E3"
        write(OUT/"SUMMARY.json",summary)
        return
    _,_,_,mode,arm,raw=max(candidates)
    results={}
    attributed=True
    delayed=True
    for block in BLOCKS:
        a,_=fixture(block,"A")
        b,b_probes=fixture(block,"B")
        n=[]
        for law in range(2):
            control=Model(mode,block)
            opt=T.optim.Adam(control.parameters(),lr=.003)
            for row in null_stream(a,block,law,arm):
                update(control,opt,row,arm)
            ref=e3(control,block,law,"A")
            old=raw[str(block)]["by_law"][law]
            attributed &= all(x["balanced_exact"]-y["balanced_exact"]>=.10 for x,y in zip(old,ref))
            n.append(dict(sham=ref))
        results[str(block)]=n
    if attributed:
        from official_worked import e2
        for block in BLOCKS:
            bank,b_probes=fixture(block,"B")
            for law in range(2):
                model,opt=state(block,mode,arm,law)
                for row in sequence(bank,block,law,arm,"B",2048):
                    update(model,opt,row,arm)
                post=e3(model,block,law,"post_B")
                old=raw[str(block)]["by_law"][law]
                delayed &= all(y["balanced_exact"]>=cap+.05 and
                               x["balanced_exact"]-y["balanced_exact"]<=.05
                               for x,y,cap in zip(old,post,ceilings))
                results[str(block)][law].update(post_B=post,B_acquisition=e2(model,b_probes,law))
    write(OUT/"E3_CONTINUATION.json",results)
    summary.update(status="COMPLETE_CONDITIONAL",selected=[mode,arm],
                   prior_teaching_gate=attributed,post_B_gate=attributed and delayed,
                   conditional_cpu_seconds=time.process_time()-start,
                   conditional_pairs=4*8192+(4*2048 if attributed else 0))
    write(OUT/"SUMMARY.json",summary)


if __name__=="__main__":
    main()
