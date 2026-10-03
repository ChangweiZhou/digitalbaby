# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure fake-tool tests of the checked-in collector; no external action."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run_checks():
 code=r'''
const fs=require('fs'),vm=require('vm');const ctx={};vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
(async()=>{
 const checks=[];
 for(const scenario of ['success','failure','host_loss']) {
  let launches=0,guards=0,waits=0,interrupts=0;const ids=[];
  const tools={exec_command:async arg=>{
   if(arg.cmd==='synthetic command'){launches++;return {session_id:71,output:'first '};}
   guards++;return {exit_code:scenario==='host_loss'?1:0,output:'guard'};
  },write_stdin:async arg=>{ids.push(arg.session_id);if(arg.chars==='\x03'){interrupts++;return {exit_code:130,output:''};}waits++;return {exit_code:scenario==='failure'?9:0,output:'second'};}};
  let value,error;try{value=await ctx.collectCommand(tools,'synthetic command',{root:'/synthetic',hostToken:'mock'});}catch(e){error=e;}
  if(launches!==1 || guards!==1 || ids.some(x=>x!==71))throw Error('duplicated/wrong session');
  if(scenario==='success' && (value!=='first second'||error||waits!==1||interrupts))throw Error('yield success');
  if(scenario==='failure' && (!error||waits!==1||interrupts))throw Error('yield failure');
  if(scenario==='host_loss' && (!error||waits||interrupts!==1))throw Error('host loss');
  checks.push('actual_transport_collector_'+scenario);
 }
 process.stdout.write(JSON.stringify(checks));
})().catch(e=>{process.stderr.write(String(e));process.exitCode=1;});
'''
 r=subprocess.run(['node','-e',code,str(ROOT/'operations/controller.js')],capture_output=True,text=True,timeout=20)
 assert r.returncode==0,r.stderr
 return json.loads(r.stdout)
if __name__=='__main__':print(json.dumps(run_checks()))
