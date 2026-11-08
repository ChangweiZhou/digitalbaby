// Static interaction-logic check in a VM with a minimal inert DOM.
// This is not a browser, screenshot, or visual rendering check.
const fs=require('fs'), vm=require('vm'), path=require('path'), assert=require('assert');
const root=__dirname;
const text=fs.readFileSync(path.join(root,'mechanism-review.html'),'utf8');
const script=text.match(/<script>([\s\S]*)<\/script>/)[1];
const nodes={};
function make(){return {value:'',innerHTML:'',textContent:'',children:[],append(x){this.children.push(x)},addEventListener(){}}}
const document={getElementById(id){return nodes[id]??=make()},createElement(){return make()}};
const context=vm.createContext({document});
new vm.Script(script).runInContext(context,{timeout:10000});
const rows=()=>Number(nodes.count.textContent.split(' / ')[0]);
assert.equal(rows(),577);
assert.equal(nodes.src.children.length,11);
assert.equal(nodes.fam.children.length,31);
nodes.src.value='CATALOG136';vm.runInContext('render()',context);assert.equal(rows(),136);
nodes.src.value='';nodes.fam.value='F17';vm.runInContext('render()',context);assert.equal(rows(),3);
assert(nodes.family.innerHTML.includes('Displacement accumulation'));
nodes.fam.value='';nodes.q.value='Orb2';vm.runInContext('render()',context);assert(rows()>0&&rows()<577);
const orb2Count=rows();
nodes.reset.onclick();assert.equal(rows(),577);
nodes.q.value='unlikely_nonsense_7777777777';vm.runInContext('render()',context);assert.equal(rows(),0);
nodes.reset.onclick();nodes.decision.value='IN PROGRESS E15';vm.runInContext('render()',context);assert.equal(rows(),3);
nodes.reset.onclick();assert.equal(rows(),577);
assert(vm.runInContext("E('<script>')",context)==='&lt;script&gt;');
const result={mode:'INERT_DOM_LOGIC_ONLY',rows:577,source_options:11,family_options:31,
  checks:['initial rows','source filtering','secondary-family filtering','text search','reset','zero matches','decision filter','HTML escaping'],
  orb2_search_matches:orb2Count,passed:true,
  visual_preview:'NOT_PERFORMED: browser security policy blocked file URLs; no workaround attempted.'};
fs.writeFileSync(path.join(root,'VIEWER_LOGIC_CHECK.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
