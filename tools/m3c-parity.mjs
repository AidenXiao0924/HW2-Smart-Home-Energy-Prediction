import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {localLoader} from './local-loader.mjs';
import {loadEngine,loadExamples,interval} from '../web/js/inference/engine.js';
import {hash} from './web-assets.mjs';
const root=new URL('../',import.meta.url),loader=localLoader(),engine=await loadEngine(loader),fixtures=await loadExamples(loader,engine);
const supplemental=JSON.parse(await readFile(new URL('reports/tests/m3c/supplemental-python-reference.json',root),'utf8'));
const rows=[];
function compare(name,origin,input,expected,actual){const errors=Object.fromEntries(['predicted','lower','upper'].map(k=>[k,Math.abs(actual[k]-expected[k])]));const ok=Object.values(errors).every(v=>Number.isFinite(v)&&v<=1e-6)&&actual.interval_valid===expected.interval_valid&&JSON.stringify(actual.warnings)===JSON.stringify(expected.warnings);rows.push({name,origin,input,expected,actual,absolute_errors:errors,status:ok?'PASS':'FAIL'});}
fixtures.fixtures.forEach((f,i)=>compare('C1-'+(i+1),f.kind,f.input,f.expected,engine.predictOne(f.input)));
supplemental.cases.forEach(f=>compare(f.name,f.source,f.input,f.expected,engine.predictOne(f.input)));
fixtures.interval_boundary_cases.forEach((f,i)=>compare('interval-function-'+(i+1),f.kind,{prediction:f.prediction,q:f.q},f.expected,interval(f.prediction,f.q)));
const rejected=fixtures.invalid_input_cases.map((f,i)=>{let actual_code='ACCEPTED';try{engine.predictOne(f.input);}catch(e){actual_code=e.code;}return {name:'C3-invalid-'+(i+1),expected_code:f.expected_code,actual_code,status:actual_code===f.expected_code?'PASS':'FAIL'};});
const result={test_version:'m3c-parity-1.0.0',model_version:engine.model.model_version,tolerance_absolute_wh:1e-6,original_python_fixtures:fixtures.fixtures.length,supplemental_python_cases:supplemental.cases.length,interval_function_cases:fixtures.interval_boundary_cases.length,original_invalid_cases:rejected.length,max_absolute_errors:Object.fromEntries(['predicted','lower','upper'].map(k=>[k,Math.max(...rows.map(r=>r.absolute_errors[k]))])),rows,rejected,inputs:{original_fixture_sha256:hash(await readFile(new URL('artifacts/deployment/mlr-1.0.0/parity_fixtures.json',root))),public_model_sha256:engine.model_sha256,supplemental_sha256:hash(await readFile(new URL('reports/tests/m3c/supplemental-python-reference.json',root)))}};
const failed=[...rows,...rejected].filter(r=>r.status!=='PASS').length;result.status=failed?'FAIL':'PASS';
const output=(process.env.PARITY_OUTPUT||'reports/tests/m3c')+'/';
await mkdir(new URL(output,root),{recursive:true});await writeFile(new URL(output+'parity.json',root),JSON.stringify(result,null,2)+'\n');
await writeFile(new URL(output+'parity.xml',root),`<testsuite name="m3c-parity" tests="${rows.length+rejected.length}" failures="${failed}">${[...rows,...rejected].map(r=>`<testcase name="${r.name}">${r.status==='FAIL'?'<failure message="Parity mismatch"/>':''}</testcase>`).join('')}</testsuite>`);
console.log(JSON.stringify({status:result.status,original:result.original_python_fixtures,supplemental:result.supplemental_python_cases,boundary:result.interval_function_cases,invalid:rejected.length,max_absolute_errors:result.max_absolute_errors}));if(failed)process.exitCode=1;
