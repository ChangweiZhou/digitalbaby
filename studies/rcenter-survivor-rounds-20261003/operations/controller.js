// SPDX-License-Identifier: GPL-3.0-or-later
// Authenticated transport adapter, executed by the coordinator with its authorized tools.
// Content-triggered; every uncertain mutation stops for read-only reconciliation.
// Follow the exact yielded shell session; never repeat a command or detach a writer.
async function collectCommand(tools, cmd, options, max=200000) {
  const start=Date.now(); let output='', r, session;
  try {
    r=await tools.exec_command({cmd,tty:true,yield_time_ms:1000,max_output_tokens:max});
    output=r.output||''; session=r.session_id;
    while(session && r.exit_code===undefined) {
      if(Date.now()-start>900000 || output.length>16*1024*1024) throw new Error('bounded local transport collector exhausted');
      const guard=await tools.exec_command({cmd:`SURVIVOR_HOST_TOKEN='${options.hostToken}' python3 '${options.root}/operations/transport.py' assert_host`,yield_time_ms:1000,max_output_tokens:2000});
      if(guard.session_id || guard.exit_code!==0) throw new Error('tool host lost while collecting local transport session');
      r=await tools.write_stdin({session_id:session,chars:'',yield_time_ms:1000,max_output_tokens:max});output+=r.output||'';
    }
    if(r.exit_code!==0) throw new Error(output||'local transport command failed');
    return output;
  } catch(error) {
    // Exact session is a PTY, so interrupting it stops its local helper. External phase stays unresolved.
    if(session && r?.exit_code===undefined) await tools.write_stdin({session_id:session,chars:'\x03',yield_time_ms:1000,max_output_tokens:2000});
    throw error;
  }
}
async function controller(tools, options = {}) {
  const root = options.root;
  if(!/^\/[A-Za-z0-9_./-]+$/.test(root || '')) throw new Error('explicit consumer-local study root required');
  const repo = 'ChangweiZhou/digitalbaby', branch = 'study/rcenter-survivor-rounds-20261003';
  const run = (cmd,max=200000) => collectCommand(tools,cmd,options,max);
  const command = (action, data = {}) => `SURVIVOR_HOST_TOKEN='${options.hostToken}' python3 '${root}/operations/transport.py' '${action}' <<'RC_JSON'\n${JSON.stringify(data)}\nRC_JSON`;
  const local = async (action,data) => JSON.parse(await run(command(action,data)));
  const structured = r => {
    if(r.isError) throw new Error(JSON.stringify(r));
    const d=r.structuredContent;
    if(!d) throw new Error('missing structured result; stop and reconcile');
    return d.result || d;
  };
  const github = async url => {
    const d=structured(await tools.mcp__codex_apps__github_fetch({url}));
    return JSON.parse(d.content);
  };
  const helper = async (path,id,attrs) => run(`python3 '${root}/operations/library_file_transfer.py' apply-xattrs '${path}' '${id}' <<'RC_JSON'\n${JSON.stringify(attrs || [])}\nRC_JSON`,2000);
  const pending=await local('pending');
  if(pending && pending.status==='running') throw new Error('unfinished external mutation requires read-only reconciliation before resume');
  await local('assert_host');
  const checkpoint=await local('checkpoint'); let privateChanged=false;
  if(checkpoint.upload_needed) {
    await local('phase',{status:'running',phase:'private_upload',contentIdentity:checkpoint.meta.identity,sha256:checkpoint.meta.sha256,prior:checkpoint.prior});
    let r;
    await local('assert_host');
    if(checkpoint.prior) r=await tools.mcp__codex_apps__library_replace_library_file({file:checkpoint.meta.path,library_file_id:checkpoint.prior.library_file_id,expected_current_version:checkpoint.prior.version,version_reason:'Content-triggered R_center source and immutable receipt checkpoint'});
    else r=await tools.mcp__codex_apps__library_create_library_file({file:checkpoint.meta.path,library_artifact_type:'other'});
    const d=structured(r);
    await helper(checkpoint.meta.path,d.library_file_id,d.xattrs);
    await local('phase',{status:'running',phase:'private_readback',contentIdentity:checkpoint.meta.identity,sha256:checkpoint.meta.sha256,library:d});
    await local('preflight_io',{extra_bytes:2*checkpoint.meta.bytes});
    const back=structured(await tools.mcp__codex_apps__library_prepare_materialize({items:[{library_file_id:d.library_file_id,file_id:d.file_id,file_name:d.file_name}],destination:{directory:root+'/backups/readback'}}));
    const t=back.transfers?.[0];
    if(!t?.workspace_path) throw new Error('readback not materialized locally; use current Library flow before resuming');
    await helper(t.workspace_path,d.library_file_id,t.xattrs);
    await local('private_ack',{meta:checkpoint.meta,library:d,readback_path:t.workspace_path});
    privateChanged=true;
    await local('phase',{status:'complete',phase:'private_verified',contentIdentity:checkpoint.meta.identity});
  }
  if(options.privateOnly) return {privateChanged,publicChanged:false,privateOnly:true};
  const projection=await local('manifest',{include_code:!!options.includeCode});let publicChanged=false;
  let state=JSON.parse(await run(`cat '${root}/operations/GITHUB_STATE.json'`,3000));
  if(projection.changed.length) {
    const ref=await github(`https://api.github.com/repos/${repo}/git/ref/heads/${branch}`);
    if(ref.object.sha!==state.verified_head) throw new Error('remote head drift; reconcile before publishing');
    let tree=state.verified_tree;
    await local('phase',{status:'running',phase:'public_begin',includeCode:!!options.includeCode,publicIdentity:projection.identity,parent:state.verified_head,tree});
    for(let i=0;i<projection.changed.length;) {
      const names=[];let bytes=0;
      while(i<projection.changed.length && (names.length===0 || bytes+projection.files[projection.changed[i]].bytes<=180000)) {
        const name=projection.changed[i++];names.push(name);bytes+=projection.files[name].bytes;
      }
      const rows=await local('data',{identity:projection.identity,paths:names,include_code:!!options.includeCode});
      const entries=[];
      for(const row of rows) {
        if(!row.path.startsWith(projection.prefix)) throw new Error('public path out of authorized study');
        if(row.encoding==='base64') {
          await local('assert_host');
          const b=structured(await tools.mcp__codex_apps__github_create_blob({repository_full_name:repo,encoding:'base64',content:row.content}));
          if(b.sha!==row.git_sha) throw new Error('binary blob SHA mismatch');
          entries.push({path:row.path,mode:'100644',type:'blob',sha:b.sha});
        } else entries.push({path:row.path,mode:'100644',type:'blob',content:row.content});
      }
      await local('assert_host');
      tree=structured(await tools.mcp__codex_apps__github_create_tree({repository_full_name:repo,base_tree_sha:tree,tree_elements:entries})).sha;
      await local('phase',{status:'running',phase:'public_tree',includeCode:!!options.includeCode,publicIdentity:projection.identity,parent:state.verified_head,tree,processed:i});
    }
    await local('assert_host');
    const commit=structured(await tools.mcp__codex_apps__github_create_commit({repository_full_name:repo,parent_sha:state.verified_head,tree_sha:tree,message:options.message || 'Sync frozen R_center programme evidence'})).sha;
    await local('phase',{status:'running',phase:'public_commit',includeCode:!!options.includeCode,publicIdentity:projection.identity,parent:state.verified_head,tree,commit});
    const fresh=await github(`https://api.github.com/repos/${repo}/git/ref/heads/${branch}`);
    if(fresh.object.sha!==state.verified_head) throw new Error('remote changed while preparing commit; no ref mutation');
    await local('assert_host');
    structured(await tools.mcp__codex_apps__github_update_ref({repository_full_name:repo,branch_name:branch,sha:commit,force:false}));
    const verified=await github(`https://api.github.com/repos/${repo}/git/ref/heads/${branch}`);
    if(verified.object.sha!==commit) throw new Error('ref readback mismatch');
    const rootTree=await github(`https://api.github.com/repos/${repo}/git/trees/${tree}`);
    const studies=rootTree.tree.find(x=>x.path==='studies' && x.type==='tree');
    if(!studies) throw new Error('study tree absent');
    const first=await github(`https://api.github.com/repos/${repo}/git/trees/${studies.sha}`);
    const own=first.tree.find(x=>x.path==='rcenter-survivor-rounds-20261003' && x.type==='tree');
    const all=await github(`https://api.github.com/repos/${repo}/git/trees/${own.sha}?recursive=1`);
    if(all.truncated) throw new Error('tree listing truncated');
    const remote=new Map(all.tree.filter(x=>x.type==='blob').map(x=>[x.path,x.sha]));
    for(const [path,value] of Object.entries(projection.files)) if(remote.get(path)!==value.git_sha) throw new Error('public tree mismatch at '+path);
    await local('public_ack',{identity:projection.identity,include_code:!!options.includeCode,commit,tree});
    publicChanged=true;
    await local('phase',{status:'complete',phase:'public_verified',publicIdentity:projection.identity,commit,tree});
  }
  if(options.world!==undefined || options.key || options.prelaunch) await local('barrier',{world:options.world,key:options.key,prelaunch:!!options.prelaunch});
  return {privateChanged,publicChanged,publicIdentity:projection.identity,changedFiles:projection.changed.length};
}

// Read-only external reconciliation. It can acknowledge proven completed writes;
// it never retries an uncertain upload/ref mutation or clears an unproven phase.
async function reconcileTransport(tools, options) {
  const root=options.root,token=options.hostToken,repo='ChangweiZhou/digitalbaby',branch='study/rcenter-survivor-rounds-20261003';
  const run=cmd=>collectCommand(tools,cmd,options);
  const local=async(action,data={})=>JSON.parse(await run(`SURVIVOR_HOST_TOKEN='${token}' python3 '${root}/operations/transport.py' '${action}' <<'RC_JSON'\n${JSON.stringify(data)}\nRC_JSON`));
  const unpack=r=>{if(r.isError)throw new Error(JSON.stringify(r));return r.structuredContent.result||r.structuredContent;};
  const get=async url=>JSON.parse(unpack(await tools.mcp__codex_apps__github_fetch({url})).content);
  const info=await local('info'),p=info['TRANSPORT_PENDING.json'];if(!p)return {resolved:true,pending:false};
  if(p.phase==='private_upload'||p.phase==='private_readback') {
    const id=p.library?.library_file_id||p.prior?.library_file_id;
    if(!id)return {resolved:false,reason:'uncertain initial create has no confirmed Library identity; coordinator lookup required'};
    await local('preflight_io',{extra_bytes:2*48*1024*1024});
    const back=unpack(await tools.mcp__codex_apps__library_prepare_materialize({items:[{library_file_id:id}],destination:{directory:root+'/backups/reconcile-readback'}}));
    const t=back.transfers?.[0];if(!t?.workspace_path)return {resolved:false,reason:'supported local Library readback unavailable'};
    await run(`python3 '${root}/operations/library_file_transfer.py' apply-xattrs '${t.workspace_path}' '${id}' <<'RC_JSON'\n${JSON.stringify(t.xattrs||[])}\nRC_JSON`);
    return local('reconcile_private',{transfer:t});
  }
  if(p.phase==='public_commit' && p.commit) {
    const ref=await get(`https://api.github.com/repos/${repo}/git/ref/heads/${branch}`);
    if(ref.object.sha!==p.commit)return {resolved:false,reason:'target ref not confirmed; coordinator must review exact pending update before retry',expected:p.commit,observed:ref.object.sha};
    const rootTree=await get(`https://api.github.com/repos/${repo}/git/trees/${p.tree}`);
    const studies=rootTree.tree.find(x=>x.path==='studies'&&x.type==='tree');const outer=await get(`https://api.github.com/repos/${repo}/git/trees/${studies.sha}`);
    const own=outer.tree.find(x=>x.path==='rcenter-survivor-rounds-20261003'&&x.type==='tree');const all=await get(`https://api.github.com/repos/${repo}/git/trees/${own.sha}?recursive=1`);
    if(all.truncated)return {resolved:false,reason:'remote tree readback truncated'};
    return local('reconcile_public',{commit:p.commit,tree:p.tree,files:Object.fromEntries(all.tree.filter(x=>x.type==='blob').map(x=>[x.path,x.sha]))});
  }
  return {resolved:false,reason:'uncertain staging phase requires coordinator inspection; no automatic mutation retry',phase:p.phase};
}
