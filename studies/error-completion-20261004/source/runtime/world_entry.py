# SPDX-License-Identifier: GPL-3.0-or-later
import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_world import execute
r=execute(int(os.environ['SURVIVOR_WORLD']),'science' if os.environ['SURVIVOR_MODE']=='science' else 'technical',os.environ['SURVIVOR_DEST'])
print(json.dumps({'world':r['world'],'kind':r['kind'],'resources':r['resources'],'parts':len(r['parts'])}),flush=True)

if os.environ['SURVIVOR_MODE']=='cycle3' and os.environ['SURVIVOR_JOB']=='replay':
 from validate_receipt import load
 from integrity import ROOT,hashes
 from durable import create
 first=load(ROOT/'receipts/cycle3/pilot-320200',320200,hashes());second=load(os.environ['SURVIVOR_DEST'],320200,hashes())
 assert first['scientific_digest']==second['scientific_digest'],'independent-process scientific replay mismatch'
 create(ROOT/'receipts/cycle3/replay_comparison.json',{'passed':True,'first_manifest_sha256':first['manifest_sha256'],'replay_manifest_sha256':second['manifest_sha256'],'scientific_digest':first['scientific_digest'],'excluded_fields':['resources only'],'same_world':320200,'fresh_process':True})
