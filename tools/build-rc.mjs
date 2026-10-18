import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {audit,hash} from './web-assets.mjs';
const root=fileURLToPath(new URL('../',import.meta.url));
export async function buildRC({verify=false}={}){
 const assets=await audit('reports/web/m4a3'),files=assets.files;
 const tree_sha256=hash(Buffer.from(JSON.stringify(files)));
 const manifest={version:'web-rc3',format:1,model_version:'mlr-1.0.0',deployment:'DEPLOYMENT_PENDING',assignment_ready:false,files,total_files:files.length,total_bytes:files.reduce((s,f)=>s+f.bytes,0),tree_sha256};
 const dest=path.join(root,'releases/web-rc3'),mf=path.join(dest,'release-manifest.json');
 let existing;try{existing=await readFile(mf,'utf8');}catch(e){if(e.code!=='ENOENT')throw e;}
 if(existing&&existing!==JSON.stringify(manifest,null,2)+'\n')throw Error('Immutable RC differs; create a new reviewed version.');
 if(verify&&!existing)throw Error('RC not built');
 if(!existing){await mkdir(path.join(dest,'site'),{recursive:true});for(const f of files){const p=path.join(dest,'site',f.path);await mkdir(path.dirname(p),{recursive:true});await copyFile(path.join(root,'web',f.path),p);}await writeFile(mf,JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});}
 const {readdir}=await import('node:fs/promises');const actual=[];
 async function walk(d){for(const e of await readdir(d,{withFileTypes:true})){if(e.isSymbolicLink())throw Error('RC symlink rejected');const p=path.join(d,e.name);if(e.isDirectory())await walk(p);else actual.push(path.relative(path.join(dest,'site'),p).split(path.sep).join('/'));}}
 await walk(path.join(dest,'site'));if(actual.length!==files.length||actual.some(p=>!files.some(f=>f.path===p)))throw Error('Unexpected RC asset');
 for(const f of files){const b=await readFile(path.join(dest,'site',f.path));if(b.length!==f.bytes||hash(b)!==f.sha256)throw Error('RC hash mismatch: '+f.path);}
 return manifest;
}
if(process.argv[1]===fileURLToPath(import.meta.url)){const m=await buildRC({verify:process.argv.includes('--verify')});console.log(JSON.stringify({version:m.version,total_files:m.total_files,total_bytes:m.total_bytes,tree_sha256:m.tree_sha256,verified:true}));}
