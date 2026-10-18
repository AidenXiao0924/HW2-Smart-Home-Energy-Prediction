import {BASE,FEATURES,VERSION,TRAINING_MANIFEST_HASH} from './config.js';import {sha256} from './loader.js';import {assertSchema,ContractError} from '../inference/schema.js';import {sourceCalendar} from '../inference/time.js';
const verified=new WeakSet(),cache=new Map();const fail=()=>{throw new ContractError('TRAINING_ARTIFACT_INVALID');};
const freeze=x=>{if(x&&typeof x==='object'){Object.values(x).forEach(freeze);Object.freeze(x);}return x;};
export function loadTraining(kind='ranges',{fetcher=globalThis.fetch}={}){
 if(!['summary','ranges'].includes(kind))return Promise.reject(new ContractError('TRAINING_ARTIFACT_INVALID'));
 const useCache=fetcher===globalThis.fetch;if(useCache&&cache.has(kind))return cache.get(kind);
 const task=(async()=>{async function read(name,expected){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);try{const response=await fetcher(new URL('public/training-v1/'+name,BASE),{signal:controller.signal,credentials:'omit',cache:'no-cache'});if(!response.ok)fail();const b=await response.arrayBuffer();if(expected.bytes!==undefined&&b.byteLength!==expected.bytes||await sha256(b)!==expected.sha256)fail();return JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(b));}finally{clearTimeout(timer);}}
 const m=await read('manifest.json',{sha256:TRAINING_MANIFEST_HASH});if(m.model_version!==VERSION||Object.keys(m.files).toSorted().join('|')!=='ranges.json|schema.json|summary.json')fail();
 const [schema,data]=await Promise.all([read('schema.json',m.files['schema.json']),read(kind+'.json',m.files[kind+'.json'])]);assertSchema(m,schema.definitions.manifest);assertSchema(data,schema.definitions[kind]);
 if(JSON.stringify(data.feature_order)!==JSON.stringify(FEATURES)||data.provenance.split!=='TRAIN'||data.provenance.rows!==13814||JSON.stringify(data.provenance)!==JSON.stringify(m.provenance))fail();
 if(kind==='ranges'){data.ranges.forEach((r,i)=>{if(r.feature!==FEATURES[i]||!Number.isFinite(r.min)||!Number.isFinite(r.max)||r.min>r.max||r.valid_count+r.missing_count!==13814)fail();});}
 else{data.distributions.forEach((r,i)=>{if(r.feature!==FEATURES[i]||r.histogram.edges.length!==r.histogram.counts.length+1||r.histogram.counts.reduce((a,b)=>a+b,0)!==r.valid_count||r.valid_count+r.missing_count!==13814)fail();});if(data.correlation.matrix.length!==15||data.correlation.matrix.some(r=>r.length!==15||r.some(x=>!Number.isFinite(x)||Math.abs(x)>1.0000000001)))fail();if(data.importance.method!=='Permutation Importance'||data.importance.split!=='TRAIN_CV_INNER'||data.importance.mean.length!==27||data.importance.sources.length!==5)fail();}
 const bundle=freeze({data,manifest:m,manifest_sha256:TRAINING_MANIFEST_HASH});verified.add(bundle);return bundle;})();if(useCache){cache.set(kind,task);task.catch(()=>cache.delete(kind));}return task;
}
export const unavailableRange=(reason='SOURCE_UNAVAILABLE')=>({status:'UNAVAILABLE',is_ood:null,details:[],imputed_features:[],out_of_range_count:null,reason});
export function checkRange(row,bundle){
 if(bundle===null||bundle===undefined)return unavailableRange();if(!verified.has(bundle)||!bundle.data.ranges)fail();
 const hour=sourceCalendar(row.date).Hour;if(Object.hasOwn(row,'Hour')&&row.Hour!==hour)throw new ContractError('TIME_FEATURE_MISMATCH','Hour');
 const details=bundle.data.ranges.map(r=>{const v=r.feature==='Hour'?hour:row[r.feature];if(v===null)return {...r,value:null,status:'IMPUTED'};if(typeof v!=='number'||!Number.isFinite(v))throw new ContractError('NONFINITE_OR_TYPE',r.feature);return {...r,value:v,status:v<r.min?'BELOW_MIN':v>r.max?'ABOVE_MAX':'IN_RANGE'};});const bad=details.filter(r=>r.status==='BELOW_MIN'||r.status==='ABOVE_MAX'),imputed=details.filter(r=>r.status==='IMPUTED').map(r=>r.feature);
 return {status:bad.length?'OOD_WARNING':imputed.length?'IMPUTED':'IN_RANGE',is_ood:bad.length>0,details,imputed_features:imputed,out_of_range_count:bad.length,source_sha256:bundle.data.provenance.source_sha256};
}
