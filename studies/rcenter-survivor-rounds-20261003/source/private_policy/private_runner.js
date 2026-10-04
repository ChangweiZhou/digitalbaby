// SPDX-License-Identifier: GPL-3.0-or-later
// Fixed qualification only. This layer neither executes nor clears GitHub operations.
async function runPrivateQualification(tools,options) {
 const root=options.root;let keeper=null,ready=false,error=null;
 const whole=async cmd=>{let r=await tools.exec_command({cmd,yield_time_ms:1000,max_output_tokens:200000}),out=r.output||'';while(r.session_id&&r.exit_code===undefined){r=await tools.write_stdin({session_id:r.session_id,chars:'',yield_time_ms:1000,max_output_tokens:200000});out+=r.output||'';}if(r.exit_code!==0)throw new Error(out);return out.trim();};
 const token=await whole("python3 -c 'import secrets;print(secrets.token_hex(24))'");
 const env=`SURVIVOR_HOST_TOKEN='${token}'`;
 const helper=async(action,data={})=>JSON.parse(await whole(`${env} '${root}/operations/venv/bin/python' '${root}/source/private_policy/private_transport.py' '${action}' <<'RC_JSON'\n${JSON.stringify(data)}\nRC_JSON`));
 const supervise=async(action,key,reservation)=>JSON.parse(await whole(`${env} '${root}/operations/venv/bin/python' '${root}/source/private_policy/adapter.py' cycle3 --job '${key.split('/')[1]}' --action '${action}' ${reservation?`--reservation-id '${reservation}'`:''}`));
 const result={jobs:[],privateBarriers:[],githubSyncVerified:false,publicationDeferred:true};let sync=null;
 try {
  keeper=await tools.exec_command({cmd:`python3 '${root}/operations/host_session.py' '${token}'`,tty:true,yield_time_ms:1000,max_output_tokens:2000});if(!keeper.session_id||!keeper.output.includes(token))throw new Error('host keeper unavailable');ready=true;
  const source=await whole(`'${root}/operations/venv/bin/python' - <<'PY'\nimport sys,json,pathlib\nr=pathlib.Path('${root}');sys.path.insert(0,str(r/'source/private_policy'));import adapter;adapter.require_policy();print(json.dumps((r/'source/private_policy/private_controller.js').read_text()))\nPY`);
  sync=new Function('tools','options',JSON.parse(source)+'\nreturn privateController(tools,options);');
  // The completed pure operations receipt already exists; verify it privately without asserting public success.
  const completedKeys=await helper('completed');
  if(!completedKeys.includes('cycle3/operations'))throw new Error('completed pure operations prerequisite missing');
  // Refresh every genuine private barrier from the current verified archive; older local ZIP copies need not survive a reset.
  for(const key of completedKeys)await sync(tools,{root,hostToken:token,key});
  const allowed=new Set(options.jobs||['pilot','replay']);
  while(true){
   const plan=await supervise('plan','cycle3/pilot');
   if(plan.blocked.some(x=>x!=='persistence_barrier'))throw new Error('private admission block: '+JSON.stringify(plan.blocked));
   for(const p of plan.pending_barriers){const ack=await sync(tools,{root,hostToken:token,key:p.key});result.privateBarriers.push({key:p.key,...ack});}
   if(plan.pending_barriers.length)continue;
   if(!plan.next_key||!allowed.has(plan.next_key.split('/')[1]))break;
   const key=plan.next_key,res=await supervise('reserve',key);
   await sync(tools,{root,hostToken:token});await helper('reservation_ack',res);
   const completed=await supervise('run',key,res.attempt.reservation_id);
   if(completed.status!=='completed')throw new Error('fixed private qualification job failed');
   result.jobs.push({key,charged_s:completed.charged_s,receipt_sha256:completed.receipt_sha256});
   const ack=await sync(tools,{root,hostToken:token,key});result.privateBarriers.push({key,...ack});
   if(options.maxJobs&&result.jobs.length>=options.maxJobs)break;
  }
  if(options.verifyNoop){const first=await sync(tools,{root,hostToken:token});const second=await sync(tools,{root,hostToken:token});if(second.privateChanged)throw new Error('unchanged private checkpoint churn');result.noop={first,second};}
  return result;
 }catch(e){error=e;if(sync)try{await sync(tools,{root,hostToken:token});}catch(b){e.message+='; private failure checkpoint unverified: '+String(b);}throw e;
 }finally{
  if(ready)try{await helper('stop_host');let r=await tools.write_stdin({session_id:keeper.session_id,chars:'',yield_time_ms:1000,max_output_tokens:2000});const start=Date.now();while(r.session_id&&r.exit_code===undefined&&Date.now()-start<10000)r=await tools.write_stdin({session_id:keeper.session_id,chars:'',yield_time_ms:1000,max_output_tokens:2000});if(r.exit_code!==0)throw new Error('host exit unverified');const info=await helper('info');if(info.host.active||info.host.host_token!==token)throw new Error('host closure record mismatch');result.hostClosed=true;}catch(e){try{await tools.write_stdin({session_id:keeper.session_id,chars:'\x03',yield_time_ms:1000,max_output_tokens:2000});}catch(ignored){}if(!error)throw e;error.message+='; host close unverified: '+String(e);}
 }
}
