import {BASE,VERSION,FEATURES,DEMO_MANIFEST_HASH} from './config.js';
import {sha256} from './loader.js';import {assertSchema,ContractError} from '../inference/schema.js';
const verifiedBundles=new WeakSet();
const fail=()=>{throw new ContractError('DEMO_INTEGRITY_MISMATCH');};
const freeze=x=>{if(x&&typeof x==='object'){Object.values(x).forEach(freeze);Object.freeze(x);}return x;};
export async function loadDemo({fetcher=globalThis.fetch}={}){
 async function read(name,expected){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);try{const response=await fetcher(new URL('public/test-demo-v1/'+name,BASE),{signal:controller.signal,credentials:'omit',cache:'no-cache'});if(!response.ok)fail();const bytes=await response.arrayBuffer();if(expected.bytes!==undefined&&bytes.byteLength!==expected.bytes||await sha256(bytes)!==expected.sha256)fail();return {bytes,text:new TextDecoder('utf-8',{fatal:true}).decode(bytes)};}finally{clearTimeout(timer);}}
 const manifest=JSON.parse((await read('manifest.json',{sha256:DEMO_MANIFEST_HASH})).text);
 if(manifest.model_version!==VERSION||manifest.sample_count!==12||manifest.split!=='TEST'||Object.keys(manifest.files).toSorted().join('|')!=='demo.csv|samples.json|samples.schema.json')fail();
 const items=await Promise.all(['samples.json','samples.schema.json','demo.csv'].map(n=>read(n,manifest.files[n])));
 const data=JSON.parse(items[0].text),schema=JSON.parse(items[1].text);assertSchema(data,schema);
 if(JSON.stringify(data.feature_order)!==JSON.stringify(FEATURES)||data.rows.length!==12||data.population_rows!==2962||JSON.stringify(data.sources)!==JSON.stringify(manifest.sources))fail();
 for(const [i,r]of data.rows.entries()){if(r.test_row_index_0based!==Math.floor(i*2961/11)||r.dataset_row_index_0based!==16773+r.test_row_index_0based||r.source_csv_line_1based!==r.dataset_row_index_0based+2)fail();}
 for(const [p,h]of Object.entries(data.sources))if(p.startsWith('/')||p.includes('..')||!/^[a-f0-9]{64}$/.test(h))fail();
 const bundle=freeze({manifest,data,csv:items[2].text,manifest_sha256:DEMO_MANIFEST_HASH});verifiedBundles.add(bundle);return bundle;
}
export function predictDemo(engine,bundle,index){
 if(!verifiedBundles.has(bundle))fail();
 if(!Number.isInteger(index)||index<0||index>=12)throw new ContractError('INVALID_DEMO_SAMPLE');
 if(engine.model.model_version!==VERSION||engine.model_sha256!==bundle.manifest.model_sha256||engine.model.dataset_version!==bundle.data.sources['data/raw/energydata_complete.csv'])fail();
 const row=bundle.data.rows[index],prediction=engine.predictOne(row.input);
 const errors=Object.fromEntries(['predicted','lower','upper'].map(k=>[k,Math.abs(prediction[k]-row.saved[k])]));
 if(Object.values(errors).some(e=>!Number.isFinite(e)||e>1e-6)||prediction.interval_valid!==row.saved.interval_valid||JSON.stringify(prediction.warnings)!==JSON.stringify(row.saved.warnings))fail();
 return {row,prediction,absolute_errors:errors,residual:row.actual-prediction.predicted,absolute_error:Math.abs(row.actual-prediction.predicted)};
}
