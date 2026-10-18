// Node QA only: production loader integrity checks over the same frozen local bytes.
import {readFile} from 'node:fs/promises';
import {createLoader} from '../web/js/core/loader.js';
export function localLoader(){return createLoader({base:new URL('../artifacts/deployment/mlr-1.0.0/',import.meta.url),fetcher:async url=>{try{const b=await readFile(url);return {ok:true,arrayBuffer:async()=>Uint8Array.from(b).buffer};}catch{return {ok:false};}}});}
