// SPDX-License-Identifier: GPL-3.0-or-later
// Only owned Library writes. No GitHub call, ref update or public-queue mutation.
async function privateCollect(tools,cmd,options,max=200000) {
 let r,session,output='',start=Date.now();
 try {
  r=await tools.exec_command({cmd,tty:true,yield_time_ms:1000,max_output_tokens:max});output=r.output||'';session=r.session_id;
  while(session&&r.exit_code===undefined){
   if(Date.now()-start>900000||output.length>16*1024*1024)throw new Error('private local collector bound');
   const guard=await tools.exec_command({cmd:`SURVIVOR_HOST_TOKEN='${options.hostToken}' python3 '${options.root}/operations/transport.py' assert_host`,yield_time_ms:1000,max_output_tokens:2000});
   if(guard.session_id||guard.exit_code!==0)throw new Error('private transport host lost');
   r=await tools.write_stdin({session_id:session,chars:'',yield_time_ms:1000,max_output_tokens:max});output+=r.output||'';
  }
  if(r.exit_code!==0)throw new Error(output||'private local command failed');return output;
 }catch(e){if(session&&r?.exit_code===undefined)await tools.write_stdin({session_id:session,chars:'\x03',yield_time_ms:1000,max_output_tokens:2000});throw e;}
}
async function privateController(tools,options) {
 const {root,hostToken}=options;
 const local=async(action,data={})=>JSON.parse(await privateCollect(tools,`SURVIVOR_HOST_TOKEN='${hostToken}' '${root}/operations/venv/bin/python' '${root}/source/private_policy/private_transport.py' '${action}' <<'RC_JSON'\n${JSON.stringify(data)}\nRC_JSON`,options));
 const unpack=r=>{if(r.isError)throw new Error(JSON.stringify(r));const d=r.structuredContent;if(!d)throw new Error('missing private transport result');return d.result||d;};
 const helper=async(path,id,attrs)=>privateCollect(tools,`python3 '${root}/operations/library_file_transfer.py' apply-xattrs '${path}' '${id}' <<'RC_JSON'\n${JSON.stringify(attrs||[])}\nRC_JSON`,options,2000);
 if(await local('pending'))throw new Error('uncertain private transport requires read-only reconciliation; no retry');
 await local('assert_host');const checkpoint=await local('checkpoint');let changed=false;
 if(checkpoint.upload_needed){
  await local('phase',{status:'running',phase:'private_upload',contentIdentity:checkpoint.meta.identity,sha256:checkpoint.meta.sha256,prior:checkpoint.prior});
  await local('assert_host');
  const d=unpack(await tools.mcp__codex_apps__library_replace_library_file({file:checkpoint.meta.path,library_file_id:checkpoint.prior.library_file_id,expected_current_version:checkpoint.prior.version,version_reason:'Private-only R_center checkpoint under owner-authorized publication deferral'}));
  await helper(checkpoint.meta.path,d.library_file_id,d.xattrs);
  await local('phase',{status:'running',phase:'private_readback',contentIdentity:checkpoint.meta.identity,sha256:checkpoint.meta.sha256,library:d});
  await local('preflight_io',{extra_bytes:2*checkpoint.meta.bytes});
  const back=unpack(await tools.mcp__codex_apps__library_prepare_materialize({items:[{library_file_id:d.library_file_id,file_id:d.file_id,file_name:d.file_name}],destination:{directory:root+'/backups/readback'}}));
  const t=back.transfers?.[0];if(!t?.workspace_path)throw new Error('private readback local materialization unavailable');
  await helper(t.workspace_path,d.library_file_id,t.xattrs);
  await local('private_ack',{meta:checkpoint.meta,library:d,readback_path:t.workspace_path});
  await local('phase',{status:'complete',phase:'private_verified',contentIdentity:checkpoint.meta.identity});changed=true;
 }
 let barrier=null;if(options.key)barrier=await local('barrier',{key:options.key});
 return {privateChanged:changed,privateReadbackVerified:true,githubSyncVerified:false,publicationDeferred:true,barrier};
}
