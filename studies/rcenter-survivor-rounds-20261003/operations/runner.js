// SPDX-License-Identifier: GPL-3.0-or-later
// Active authorized-tool-host runner. One completion -> both verified barriers -> next key.
async function runProgramme(tools, options) {
  const root=options.root,mode=options.mode;
  if(!/^\/[A-Za-z0-9_./-]+$/.test(root||'') || !['cycle3','science'].includes(mode)) throw new Error('invalid study root/mode');
  const whole=async cmd=>{
    let r=await tools.exec_command({cmd,yield_time_ms:1000,max_output_tokens:200000});let output=r.output||'';
    while(r.session_id && r.exit_code===undefined){r=await tools.write_stdin({session_id:r.session_id,chars:'',yield_time_ms:1000,max_output_tokens:200000});output+=r.output||'';}
    if(r.exit_code!==0)throw new Error(output||'supervised operation failed');return output.trim();
  };
  const token=(await whole("python3 -c 'import secrets; print(secrets.token_hex(24))'"));
  let keeper=null,ready=false,primaryError=null,synchronize=null;
  const env=`SURVIVOR_HOST_TOKEN='${token}'`;
  const local=async(action,data={})=>JSON.parse(await whole(`${env} python3 '${root}/operations/transport.py' '${action}' <<'RC_JSON'\n${JSON.stringify(data)}\nRC_JSON`));
  const supervise=async(action,key,reservation)=>{
    let tail=key==='analysis/final'?`--job analysis`:(mode==='science'?`--world '${key.split('/')[1]}'`:`--job '${key.split('/')[1]}'`);
    if(reservation)tail+=` --reservation-id '${reservation}'`;
    return JSON.parse(await whole(`${env} '${root}/operations/venv/bin/python' '${root}/operations/supervise.py' '${mode}' --action '${action}' ${tail}`));
  };
  const result={mode,jobs:[],barriers:[],automatic:true};
  try {
    keeper=await tools.exec_command({cmd:`python3 '${root}/operations/host_session.py' '${token}'`,tty:true,yield_time_ms:1000,max_output_tokens:2000});
    if(!keeper.session_id || !(keeper.output||'').includes(token))throw new Error('exclusive host keeper failed to start');ready=true;
    // Load exact reviewed controller bytes. Supervisor governance independently binds the whole closure before any reservation.
    const source=await whole(`python3 - <<'PY'\nimport json,hashlib,pathlib\nr=pathlib.Path('${root}');p=r/'operations/controller.js';b=p.read_bytes();a=json.loads((r/'audits/CYCLE3_SOURCE_REVIEW.json').read_text());assert a['accepted'] and a['source_hashes']['operations/controller.js']==hashlib.sha256(b).hexdigest();print(json.dumps(b.decode()))\nPY`);
    const reconcile=new Function('tools','options',JSON.parse(source)+'\nreturn reconcileTransport(tools,options);');
    const sync=new Function('tools','options',JSON.parse(source)+'\nreturn controller(tools,options);');synchronize=sync;
    const allowed=mode==='cycle3'?new Set(options.jobs||['operations','pilot','replay']):null;
    while(true){
      const plan=await supervise('plan',mode==='science'?'science/310001':'cycle3/operations');
      if(plan.blocked.includes('unresolved_transport')){const fixed=await reconcile(tools,{root,hostToken:token});if(!fixed.resolved)throw new Error('external reconciliation blocked: '+JSON.stringify(fixed));continue;}
      if(plan.blocked.some(x=>!['persistence_barrier'].includes(x)))throw new Error('recovery/admission block: '+JSON.stringify(plan.blocked));
      // This is the actual completion detector, also used on a restored complete-but-unsynced receipt.
      for(const pending of plan.pending_barriers){
        const key=pending.key;
        const ack=await sync(tools,{root,hostToken:token,includeCode:true,key,world:key.startsWith('science/')?Number(key.split('/')[1]):undefined,message:'Persist '+key+' verified R_center evidence'});
        result.barriers.push({key,manifest_sha256:pending.manifest_sha256,...ack});
      }
      if(plan.pending_barriers.length)continue;
      if(!plan.next_key)break;
      const key=plan.next_key;
      if(allowed && !allowed.has(key.split('/')[1]))break;
      const reservation=await supervise('reserve',key);
      // Off-host reservation precedes worker birth; an executor loss cannot forget its900s upper-bound charge.
      await sync(tools,{root,hostToken:token,privateOnly:true});
      await local('reservation_ack',reservation);
      const completed=await supervise('run',key,reservation.attempt.reservation_id);
      if(completed.status!=='completed')throw new Error('native/operations job did not complete');
      result.jobs.push({key,receipt_sha256:completed.receipt_sha256,charged_s:completed.charged_s});
      // Loop detects/validates the immutable completed receipt via the production planner before syncing it.
      if(options.maxJobs && result.jobs.length>=options.maxJobs){
        const finalPlan=await supervise('plan',mode==='science'?'science/310001':'cycle3/operations');
        for(const pending of finalPlan.pending_barriers){const k=pending.key;const ack=await sync(tools,{root,hostToken:token,includeCode:true,key:k,world:k.startsWith('science/')?Number(k.split('/')[1]):undefined,message:'Persist '+k+' verified R_center evidence'});result.barriers.push({key:k,manifest_sha256:pending.manifest_sha256,...ack});}
        break;
      }
    }
    if(options.verifyTransport){
      const first=await sync(tools,{root,hostToken:token,includeCode:true});
      const second=await sync(tools,{root,hostToken:token,includeCode:true});
      if(second.privateChanged||second.publicChanged)throw new Error('unchanged controller invocation was not a no-op');
      const restore=await local('restore');
      result.transport={first,second,restore};
    }
    return result;
  } catch(error) {
    primaryError=error;
    // Preserve failed local attempt evidence automatically when the host remains available.
    // A pending external phase makes controller stop before any retry or other mutation.
    if(synchronize)try{await synchronize(tools,{root,hostToken:token,privateOnly:true});}catch(backupError){error.message+='; private failure backup unverified: '+String(backupError);}
    throw error;
  } finally {
    if(ready && keeper?.session_id) {
      try {
        await local('stop_host');
        let closed=await tools.write_stdin({session_id:keeper.session_id,chars:'',yield_time_ms:1000,max_output_tokens:3000});
        const stopStart=Date.now();
        while(closed.session_id && closed.exit_code===undefined && Date.now()-stopStart<10000)
          closed=await tools.write_stdin({session_id:keeper.session_id,chars:'',yield_time_ms:1000,max_output_tokens:3000});
        if(closed.exit_code!==0)throw new Error('host keeper did not confirm graceful terminal exit');
        const info=await local('info'),wall=info['HOST_WALL.json'];
        if(!wall || wall.active!==false || wall.host_token!==token)throw new Error('host close record absent/stale');
        result.host_close={closed:true,effective_s:wall.effective_s};
      } catch(closeError) {
        // Last-resort interruption is exact-session only; an unclosed wall record remains conservative at startup.
        try {await tools.write_stdin({session_id:keeper.session_id,chars:'\x03',yield_time_ms:1000,max_output_tokens:3000});}catch(ignored){}
        if(!primaryError)throw closeError;
        primaryError.message+='; host closure unverified: '+String(closeError);
      }
    }
  }
}
