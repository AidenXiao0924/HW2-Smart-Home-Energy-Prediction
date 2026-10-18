// Stage only C3's immutable allowlist. No model fitting or result generation.
import {readFile,writeFile,mkdir,readdir,copyFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=fileURLToPath(new URL('../',import.meta.url));
export const hash=b=>createHash('sha256').update(b).digest('hex');
export async function stage(){
 const source=path.join(root,'artifacts/deployment/mlr-1.0.0'),dest=path.join(root,'web/public/mlr-1.0.0');
 const manifestBytes=await readFile(path.join(source,'manifest.json')),manifest=JSON.parse(manifestBytes);
 const config=await readFile(path.join(root,'web/js/core/config.js'),'utf8');
 if(!config.includes(hash(manifestBytes)))throw Error('Pinned manifest mismatch');
 for(const [name,entry] of Object.entries(manifest.files)){
   if(!/^(?:charts\/|schemas\/)?[a-z_-]+(?:\.schema)?\.json$/.test(name))throw Error('Unsafe source');
   const bytes=await readFile(path.join(source,name));if(hash(bytes)!==entry.sha256||bytes.length!==entry.bytes)throw Error('C3 hash mismatch');
   await mkdir(path.dirname(path.join(dest,name)),{recursive:true});await copyFile(path.join(source,name),path.join(dest,name));
 }
 await writeFile(path.join(dest,'manifest.json'),manifestBytes);
 await writeFile(path.join(root,'web/.nojekyll'),'');
 return audit();
}
export async function audit(output='reports/web/m4a3'){
 const files=[];
 async function walk(dir){for(const e of await readdir(dir,{withFileTypes:true})){if(e.isSymbolicLink())throw Error('Symlink not public');const p=path.join(dir,e.name);if(e.isDirectory())await walk(p);else files.push(p);}}
 await walk(path.join(root,'web'));
 const approved=[/^index\.html$/, /^\.nojekyll$/, /^css\/styles\.css$/, /^public\/contracts\/csv-errors-1\.1\.0\.schema\.json$/, /^js\/(?:app\.js|core\/(?:config|loader|ui|research-data|demo-data|training-data)\.js|charts\/svg\.js|inference\/(?:engine|schema|time)\.js|csv\/(?:contracts|source|parser|adapter|analytics|batch|export)\.js|pages\/(?:research|future|simulator|csv|demo|training)\.js)$/];
 const source=path.join(root,'artifacts/deployment/mlr-1.0.0');
 const manifest=JSON.parse(await readFile(path.join(source,'manifest.json'),'utf8'));
 const entries=[];
 for(const file of files){
  const name=path.relative(path.join(root,'web'),file).split(path.sep).join('/'), bytes=await readFile(file);
  if(name.startsWith('public/training-v1/')){
   const n=name.slice('public/training-v1/'.length),prefix='artifacts/training/train-v1/';
   if(!['manifest.json','schema.json','summary.json','ranges.json'].includes(n))throw Error('Unallowlisted training export');
   if(!bytes.equals(await readFile(path.join(root,prefix,n))))throw Error('Training export modified');
   const config=await readFile(path.join(root,'web/js/core/config.js'),'utf8');if(!config.includes(hash(await readFile(path.join(root,prefix,'manifest.json')))))throw Error('Training pin mismatch');
  }else if(name.startsWith('public/test-demo-v1/')){
   const n=name.slice('public/test-demo-v1/'.length),prefix='artifacts/demo/test-v1/';
   if(!['manifest.json','samples.json','samples.schema.json','demo.csv'].includes(n))throw Error('Unallowlisted Demo');
   if(!bytes.equals(await readFile(path.join(root,prefix,n))))throw Error('Demo modified');
   const pin=await readFile(path.join(root,'web/js/core/config.js'),'utf8');if(!pin.includes(hash(await readFile(path.join(root,prefix,'manifest.json')))))throw Error('Demo pin mismatch');
  }else if(name.startsWith('public/mlr-1.0.0/')){
   const n=name.slice('public/mlr-1.0.0/'.length);if(n!=='manifest.json'&&!manifest.files[n])throw Error('Unallowlisted data');
   if(!bytes.equals(await readFile(path.join(source,n))))throw Error('Public data modified');
  }else if(!approved.some(re=>re.test(name)))throw Error('Unallowlisted web asset '+name);
  if(/(?:localhost|[A-Z]:[\\/]|\/Users\/|\/home\/|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{30})/.test(bytes.toString()))throw Error('Sensitive public content '+name);
  entries.push({path:name,bytes:bytes.length,sha256:hash(bytes)});
 }
 const result={status:'PASS',model_version:'mlr-1.0.0',deployment:'DEPLOYMENT_PENDING',assignment_ready:false,files:entries.sort((a,b)=>a.path.localeCompare(b.path)),prohibited:'No raw/processed data, joblib, venv, private manifests or user uploads'};
 await mkdir(path.join(root,output),{recursive:true});await writeFile(path.join(root,output,'public-assets.json'),JSON.stringify(result,null,2)+'\n');return result;
}
if(process.argv[1]===fileURLToPath(import.meta.url)){const result=await stage();console.log(JSON.stringify({status:result.status,files:result.files.length,bytes:result.files.reduce((a,f)=>a+f.bytes,0)}));}
