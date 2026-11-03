#!/usr/bin/env python3
"""Partial executable MiniFly meta-governance guard; NOT the missing MRK-1 checker."""
from __future__ import annotations
import argparse
import ast
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

EXIT_BAD_INPUT=2
EXIT_SCOPE_FAIL=20
EXIT_ADMISSION_FAIL=21
EXIT_RSS_KILLED=22
EXIT_TIME_KILLED=23
MODEL_PATH=re.compile(r"^minifly_v\d+(?:_[^/]*)?(?:/|$)",re.I)
META_ALLOWED=(".claude/skills/minifly-","tools/minifly_meta_guard/",
              ".github/workflows/minifly-meta-governance.yml","AGENTS.md",
              "PROJECT_DOCUMENTS/MINIFLY_META_","PROJECT_DOCUMENTS/KB/")

def output(path,payload):
    if path:
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        t=p.with_name(p.name+".tmp");t.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
        os.replace(t,p)
    print(json.dumps(payload,sort_keys=True))

def scope(paths):
    bad=[]
    for p in paths:
        if "\\" in p or p.startswith("/") or "/../" in p or p.startswith("../"):
            bad.append([p,"Unsafe path"])
        elif p=="PROJECT_DOCUMENTS/MINIFLY_VERSION_INDEX.json" or MODEL_PATH.search(p):
            bad.append([p,"Model/version changed during meta-governance"])
        elif not any(p==a or p.startswith(a) for a in META_ALLOWED):
            bad.append([p,"Outside strict meta-governance scope"])
    return {"status":"PASS_SCOPE_ONLY" if paths and not bad else "FAIL_SCOPE",
            "prohibited":bad,"changed_count":len(paths),"MRK1_certified":False}

def positive(d,k,zero=False):
    v=d.get(k)
    if type(v) is not int or v<(0 if zero else 1):
        raise ValueError(k+" must be a positive integral value")
    return v

def hazards(source):
    funcs=set()
    for n in ast.walk(ast.parse(source)):
        if isinstance(n,ast.Call):
            f=n.func
            if isinstance(f,ast.Attribute):funcs.add(f.attr)
            if isinstance(f,ast.Name):funcs.add(f.id)
    return ["Dense LP intermediate risk: toarray+hstack+linprog"] if {"toarray","hstack","linprog"}<=funcs else []

def solver(plan,source=None):
    budget=positive(plan,"rss_cap_bytes")
    reserve=plan.get("reserve_factor",1.5)
    if not isinstance(reserve,(float,int)) or not 1<=reserve<=10:raise ValueError("reserve_factor")
    strategy=plan.get("strategy")
    if strategy=="dense-multiclass":
        keys=("sites","competitors","classes","features_plus_bias","bytes_per_scalar")
        estimate=1
        for k in keys:estimate*=positive(plan,k)
    elif strategy=="bounded-sparse":
        estimate=(positive(plan,"max_nnz")*positive(plan,"bytes_per_nnz")
                  +positive(plan,"index_pointer_bytes",True)
                  +positive(plan,"max_solver_workspace_bytes")
                  +positive(plan,"baseline_rss_bytes"))
    else:raise ValueError("Unknown strategy")
    h=hazards(source) if source is not None else []
    status="ADMIT_FOR_TECHNICAL_ONLY"
    if plan.get("execution_guard") not in ("run-capped","kernel-hard-limit+run-capped"):
        status="FAIL_NO_RUNTIME_GUARD"
    elif estimate*reserve>budget:
        status="FAIL_RESOURCE_BOUND"
    elif h and strategy=="bounded-sparse":
        keys=("sites","competitors","classes","features_plus_bias","bytes_per_scalar")
        if not all(type(plan.get(k)) is int and plan[k]>0 for k in keys):
            status="FAIL_SOURCE_BOUNDS_MISMATCH"
        else:
            full=1
            for k in keys:full*=plan[k]
            if full*reserve>budget:status="FAIL_SOURCE_BOUNDS_MISMATCH"
    return {"status":status,"estimated_peak_bytes":estimate,
            "estimate_plus_reserve_bytes":int(estimate*reserve),"rss_cap_bytes":budget,
            "source_inspected":source is not None,"source_hazards":h,
            "warning":"Prospective technical filter only; neither actual RAM nor oracle correctness is certified.",
            "MRK1_certified":False}

def group_rss(pgid):
    p=subprocess.run(["ps","-eo","pgid=,rss="],capture_output=True,text=True,timeout=3)
    if p.returncode:raise RuntimeError("RSS monitor unavailable; stopping")
    total=0
    for line in p.stdout.splitlines():
        a=line.split()
        if len(a)>=2:
            try:
                if int(a[0])==pgid:total+=int(a[1])*1024
            except ValueError:pass
    return total

def kill_group(p):
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()

def run_capped(a):
    if os.name!="posix":raise ValueError("POSIX only")
    cap=int(a.rss_mib*1024*1024)
    if cap<16*1024*1024 or a.wall_seconds<=0:raise ValueError("Invalid limits")
    cmd=list(a.cmd)
    if cmd and cmd[0]=="--":cmd.pop(0)
    if not cmd:raise ValueError("No command specified")
    def preexec():
        if a.hard_as:resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    start=time.monotonic()
    p=subprocess.Popen(cmd,start_new_session=True,preexec_fn=preexec)
    peak=0;status="EXITED";rc=0
    try:
        while p.poll() is None:
            rss=group_rss(p.pid);peak=max(peak,rss)
            if rss>cap:
                status="RSS_CAP_EXCEEDED";rc=EXIT_RSS_KILLED;kill_group(p);break
            if time.monotonic()-start>a.wall_seconds:
                status="WALL_CAP_EXCEEDED";rc=EXIT_TIME_KILLED;kill_group(p);break
            time.sleep(0.05)
        if status=="EXITED":
            rc=p.wait()
            child_peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            peak=max(peak,child_peak*(1024 if sys.platform.startswith("linux") else 1))
    except BaseException:
        kill_group(p);raise
    output(a.report,{"status":status,"child_return_code":p.returncode,
          "max_observed_group_rss_bytes":peak,"rss_cap_bytes":cap,
          "wall_seconds":time.monotonic()-start,"wall_cap_seconds":a.wall_seconds,
          "hard_virtual_address_limit_enabled":a.hard_as,
          "limitation":"Sampling RSS is not a hard kernel memory limit. Strict execution requires OS/container isolation.",
          "MRK1_certified":False})
    return rc

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="act",required=True)
    s=sub.add_parser("verify-scope")
    s.add_argument("--mode",choices=["meta"],required=True)
    s.add_argument("--changed-file",action="append",default=[])
    s.add_argument("--base-ref");s.add_argument("--head-ref");s.add_argument("--report")
    d=sub.add_parser("admit-solver")
    d.add_argument("--plan",required=True);d.add_argument("--source-file");d.add_argument("--report")
    r=sub.add_parser("run-capped")
    r.add_argument("--rss-mib",type=float,required=True);r.add_argument("--wall-seconds",type=float,required=True)
    r.add_argument("--hard-as",action="store_true");r.add_argument("--report");r.add_argument("cmd",nargs=argparse.REMAINDER)
    a=p.parse_args(argv)
    try:
        if a.act=="verify-scope":
            paths=a.changed_file
            if not paths:
                if not a.base_ref or not a.head_ref:raise ValueError("Missing base/head refs")
                proc=subprocess.run(["git","diff","--name-only","--no-renames",a.base_ref+".."+a.head_ref],
                      capture_output=True,text=True,check=True)
                paths=[x.strip() for x in proc.stdout.splitlines() if x.strip()]
            v=scope(paths);output(a.report,v)
            return 0 if v["status"]=="PASS_SCOPE_ONLY" else EXIT_SCOPE_FAIL
        if a.act=="admit-solver":
            plan=json.loads(Path(a.plan).read_text())
            src=Path(a.source_file).read_text() if a.source_file else None
            v=solver(plan,src);output(a.report,v)
            return 0 if v["status"]=="ADMIT_FOR_TECHNICAL_ONLY" else EXIT_ADMISSION_FAIL
        return run_capped(a)
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as e:
        print("INVALID_INPUT_OR_MONITOR: "+str(e),file=sys.stderr)
        return EXIT_BAD_INPUT

if __name__=="__main__":sys.exit(main())
