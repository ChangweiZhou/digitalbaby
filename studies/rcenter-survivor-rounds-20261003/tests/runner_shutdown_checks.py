# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure fake-tool runner checks; checked-in runner/controller, no external writes."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run_checks():
 code=r'''
const fs=require('fs'),vm=require('vm');const ctx={};vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);const controller=fs.readFileSync(process.argv[2],'utf8');
(async()=>{const checks=[];
for(const scenario of ['normal','local_failure']) {
 const token='f'.repeat(48);let stopped=false,polled=false,backup=0,interrupt=0;
 const tools={exec_command:async({cmd})=>{
  const ok=output=>({exit_code:0,output:typeof output==='string'?output:JSON.stringify(output)});
  if(cmd.includes('secrets.token_hex'))return ok(token);
  if(cmd.includes('/operations/host_session.py'))return {session_id:91,output:JSON.stringify({ready:true,host_token:token})};
  if(cmd.includes("p=r/'operations/controller.js'"))return ok(JSON.stringify(controller));
  if(cmd.includes("--action 'plan'"))return ok({blocked:scenario==='local_failure'?['synthetic_local_failure']:[],pending_barriers:[],next_key:null});
  if(cmd.includes("'stop_host'")){stopped=true;return ok({stop_requested:true});}
  if(cmd.includes("'info'"))return ok({'HOST_WALL.json':{active:false,host_token:token,effective_s:7}});
  if(cmd.includes("'pending'"))return ok('null');
  if(cmd.includes("'assert_host'"))return ok({active:true});
  if(cmd.includes("'checkpoint'")){backup++;return ok({upload_needed:false});}
  throw Error('unexpected command '+cmd);
 },write_stdin:async a=>{if(a.session_id!==91)throw Error('wrong keeper session');if(a.chars==='\x03')interrupt++;if(!stopped)throw Error('polled before stop');polled=true;return {exit_code:0,output:'closed'};}};
 let result,error;try{result=await ctx.runProgramme(tools,{root:'/synthetic',mode:'cycle3',jobs:[]});}catch(e){error=e;}
 if(!stopped||!polled||interrupt)throw Error('graceful close not followed');
 if(scenario==='normal' && (error||!result.host_close.closed||backup))throw Error('normal close');
 if(scenario==='local_failure' && (!error||backup!==1))throw Error('failure private backup then close');
 checks.push('actual_runner_'+scenario+'_graceful_shutdown');
}process.stdout.write(JSON.stringify(checks));})().catch(e=>{process.stderr.write(String(e));process.exitCode=1;});
'''
 r=subprocess.run(['node','-e',code,str(ROOT/'operations/runner.js'),str(ROOT/'operations/controller.js')],capture_output=True,text=True,timeout=20)
 assert r.returncode==0,r.stderr
 return json.loads(r.stdout)
if __name__=='__main__':print(json.dumps(run_checks()))
