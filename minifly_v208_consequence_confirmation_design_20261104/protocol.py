"""V208 harness using the frozen V201 acting loop."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st
import sys

PARENT=Path(__file__).resolve().parent.parent/"minifly_v201_action_consequence_design_20261103"
sys.path.insert(0,str(PARENT))
from rehearse import ARMS, gates as parent_gates, rate, run
from environment import HELD, TRAIN, World, seed
spec=importlib.util.spec_from_file_location("v201_official",PARENT/"official.py")
parent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def manifest():
    lock=json.loads((PARENT/"official/LOCK.json").read_text())
    for p,h in lock["sources"].items():
        assert sha(PARENT/p)==h
    archive=PARENT/"official/RESULT_BUNDLE.zip"
    assert sha(archive)=="b2319c49896f328251f8af391d98250cb5f4d13fe829c7e6613ad84aa5649157"
    return lock

def interval(values):
    assert len(values)==32
    m=st.mean(values)
    half=2.03951345*st.stdev(values)/32**.5
    return [m-half,m+half]

def decision(lives):
    assert len(lives)==32 and len({r["world"] for r in lives})==32
    assert all(set(r["stages"])==set(parent.STAGES) and r["stats"]["R"][0]==256 for r in lives)
    # The parent's means and descriptive guards are unchanged. Its 16-world
    # confidence calculation is replaced below by the 32-world calculation.
    def values(stage,arm,held):
        return [rate(r["stages"][stage],arm,held) for r in lives]
    held=values("FINAL","ALIGNED",True)
    guards=dict(held_floor=st.mean(held)>=.75,
        trained_floor=st.mean(values("FINAL","ALIGNED",False))>=.80,
        acquisition=st.mean(values("A_END","ALIGNED",False))>=.80,
        reversal=st.mean(values("R_END","ALIGNED",False))>=.80 and
                 st.mean(values("R_END","ALIGNED",False))-st.mean(values("R0","ALIGNED",False))>=.30,
        retention=st.mean(values("R_END","ALIGNED",True))-st.mean(held)<=.10)
    panel=[x for r in lives for x in r["stages"]["FINAL"]["ALIGNED"] if x["held"]]
    guards["each_cue"]=all(st.mean(x["correct"] for x in panel if x["cue"]==c)>=.65 for c in (0,1))
    guards["each_wrapper"]=all(st.mean(x["correct"] for x in panel if tuple(x["wrapper"])==w)>=.65 for w in HELD)
    contrasts={}
    for arm in ARMS[1:]:
        d=[x-y for x,y in zip(held,values("FINAL",arm,True))]
        contrasts[arm]=dict(mean=st.mean(d),interval=interval(d))
    guards["paired"]=all(x["mean"]>=.15 and x["interval"][0]>0 for x in contrasts.values())
    safety=[r["stats"]["R"][1]/256 for r in lives]
    guards["live_safety"]=interval(safety)[0]>.5
    return dict(passed=all(guards.values()),guards=guards,contrasts=contrasts,
                live_safety=dict(mean=st.mean(safety),interval=interval(safety)))
