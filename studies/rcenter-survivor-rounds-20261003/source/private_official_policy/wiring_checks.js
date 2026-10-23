// SPDX-License-Identifier: GPL-3.0-or-later
// Mock tools only: no subprocess, native outcome, remote write or real state change.
const fs=require('fs'),assert=require('assert');
const dir=__dirname,root='/mock/study';
(async()=>{
 const seen=[];let plans=0;
 const stubController='async function privateController(tools,o){tools.record(o.key||"checkpoint");return {privateReadbackVerified:true,githubSyncVerified:false};}';
 const tools={record:x=>seen.push(x),exec_command:async({cmd})=>{
  seen.push(cmd);
  if(cmd.includes('secrets.token_hex'))return {exit_code:0,output:'abc123'};
  if(cmd.includes('operations/host_session.py'))return {session_id:42,output:'abc123'};
  if(cmd.includes("import adapter;adapter.verify_policy()")){assert(cmd.includes("r/'source/private_official_policy'"));return {exit_code:0,output:JSON.stringify(stubController)};}
  if(cmd.includes('private_transport.py')){const a=cmd.match(/private_transport.py' '([^']+)'/)[1];return {exit_code:0,output:JSON.stringify(a==='info'?{host:{active:false,host_token:'abc123'}}:{ok:true})};}
  if(cmd.includes('adapter.py')){assert(cmd.includes('adapter.py\' science'));let v;if(cmd.includes("--action 'plan'"))v={blocked:[],pending_barriers:[],next_key:plans++===0?'science/310001':null};else if(cmd.includes("--action 'reserve'")){assert(cmd.includes("--world '310001'"));v={attempt:{reservation_id:'r'}};}else{assert(cmd.includes("--action 'run'"));v={status:'completed',charged_s:1,receipt_sha256:'h'};}return {exit_code:0,output:JSON.stringify(v)};}
  throw Error('unexpected command '+cmd);
 },write_stdin:async()=>({exit_code:0,output:''})};
 const runner=new Function('tools','options',fs.readFileSync(dir+'/private_runner.js','utf8')+'\nreturn runPrivateOfficial(tools,options);');
 const result=await runner(tools,{root,maxJobs:1});assert.equal(result.jobs[0].key,'science/310001');assert.equal(result.hostClosed,true);
 const index=x=>seen.findIndex(s=>s.includes(x));
 assert(index("'prelaunch'")<index("--action 'reserve'"));
 assert(index("--action 'reserve'")<index("'reservation_ack'"));
 assert(index("'reservation_ack'")<index("--action 'run'"));
 assert(index("--action 'run'")<seen.indexOf('science/310001'));
 const local=[];let remoteWrites=0;
 const tools2={exec_command:async({cmd})=>{let v=null;
  if(cmd.includes('apply-xattrs'))return {exit_code:0,output:'{}'};
  const m=cmd.match(/private_transport.py' '([^']+)'/);assert(m);const a=m[1];local.push(a);
  if(a==='checkpoint')v={upload_needed:true,meta:{identity:'id',sha256:'sha',path:'/p',bytes:1},prior:{library_file_id:'libfile_x',version:1}};
  else if(a==='barrier')v={private_readback_verified:true};else if(a!=='pending')v={ok:true};
  return {exit_code:0,output:JSON.stringify(v)};
 },mcp__codex_apps__library_replace_library_file:async()=>{remoteWrites++;return {structuredContent:{library_file_id:'libfile_x',file_id:'file_x',file_name:'a.zip',xattrs:[]}};},mcp__codex_apps__library_prepare_materialize:async()=>({structuredContent:{transfers:[{workspace_path:'/back',xattrs:[]}]}})};
 const controller=new Function('tools','options',fs.readFileSync(dir+'/private_controller.js','utf8')+'\nreturn privateController(tools,options);');
 const c=await controller(tools2,{root,hostToken:'token',key:'science/310001'});assert.equal(remoteWrites,1);assert.equal(c.githubSyncVerified,false);assert(local.indexOf('private_ack')<local.indexOf('retention'));assert(local.indexOf('retention')<local.indexOf('barrier'));
 console.log(JSON.stringify({passed:true,checks:4,native_events:0,remote_mutations:0,scope:'actual JS runner selector/order/shutdown and controller verified-private completion ordering via mock tools'}));
})().catch(e=>{console.error(e);process.exit(1)});
