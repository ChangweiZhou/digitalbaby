"""Source-identical evaluator fixtures; separate name avoids frozen fixture imports."""
import importlib.util
import bootstrap
from io_utils import digest
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v2=module('_next_core_frozen_v2_fixture',bootstrap.V2/'v2_fixture.py')
latin=module('_next_core_frozen_latin_fixture',bootstrap.LATIN)
DT=v2.DT;RECORD_SECONDS=v2.RECORD_SECONDS
def make_world(s,assay):
    if assay in ('lifetime','reuse'):return v2.make_world(s,assay)
    if assay!='latin':raise ValueError(assay)
    d=latin.make_world(s);d.pop('sha256');d['assay']='latin'
    d['sets']={name:[{'cue_hex':ch,'outcome':y} for ch,y in zip(g['cues'],g['outcomes'])] for name,g in d['sets'].items()}
    d['branches']=['W','N_old','N_new']
    for i,e in enumerate(d['events']):e['index']=i
    d['clocks']={'old_end':d['old_end'],'new_end':d['new_end'],'final':d['final']}
    d['boundaries']={'192':['old_end'],'384':['new_end','final']}
    d['sha256']=digest(d);return d
BRANCH_MAP={'W':{'old':True,'new':True,'revision':True},
            'N_old':{'old':False,'new':True,'revision':True},
            'N_new':{'old':True,'new':False,'revision':True},
            'N_revision':{'old':True,'new':True,'revision':False}}
def permitted(branch,stage):return BRANCH_MAP[branch][stage]
