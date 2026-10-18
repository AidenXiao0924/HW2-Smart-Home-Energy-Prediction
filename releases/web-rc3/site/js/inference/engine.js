import {createLoader,validate,verifiedArtifact,verifiedHash} from '../core/loader.js';
import {FEATURES,VERSION} from '../core/config.js';
import {assertSchema,ContractError} from './schema.js';
import {sourceCalendar} from './time.js';
export {ContractError};
export function interval(predicted,q){
 if(!Number.isFinite(predicted)||!Number.isFinite(q)||q<0)throw new ContractError('NONFINITE_RESULT');
 const lower=Math.max(0,predicted-q),upper=predicted+q;
 if(!Number.isFinite(lower)||!Number.isFinite(upper))throw new ContractError('NONFINITE_RESULT');
 const warnings=predicted<0?['NEGATIVE_ESTIMATE_WARNING']:[],interval_valid=upper>=lower;
 if(!interval_valid)warnings.push('INVALID_INTERVAL');
 return {predicted,lower,upper,interval_valid,warnings};
}
export function validateModel(model,schema){
 assertSchema(model,schema);validate('model.json',model);
 const p=model.prediction_interval,t=model.preprocessing;
 if(p.method!=='split_conformal_absolute_residual'||t.imputer.strategy!=='median'||t.time_features.format!=='YYYY-MM-DD HH:mm:ss'||p.negative_warning!=='NEGATIVE_ESTIMATE_WARNING'||p.invalid_warning!=='INVALID_INTERVAL'||model.target.name!=='Appliances'||model.target.unit!=='Wh')throw new ContractError('MODEL_CONTRACT_MISMATCH');
 return model;
}
export function orderedInput(row,rowSchema){
 if(!row||typeof row!=='object'||Array.isArray(row)||![Object.prototype,null].includes(Object.getPrototypeOf(row)))throw new ContractError('INVALID_INPUT_OBJECT');
 if(!Object.hasOwn(row,'date'))throw new ContractError('MISSING_COLUMN','date');
 const cal=sourceCalendar(row.date);
 if(Object.hasOwn(row,'Hour')&&(typeof row.Hour!=='number'||!Number.isFinite(row.Hour)||row.Hour!==cal.Hour))throw new ContractError('TIME_FEATURE_MISMATCH','Hour');
 const values=FEATURES.map(feature=>{
  if(feature==='Hour')return cal.Hour;
  if(!Object.hasOwn(row,feature))throw new ContractError('MISSING_COLUMN',feature);
  const v=row[feature];if(v!==null&&(typeof v!=='number'||!Number.isFinite(v)))throw new ContractError('NONFINITE_OR_TYPE',feature);return v;
 });
 if(Object.hasOwn(row,'Appliances')&&(typeof row.Appliances!=='number'||!Number.isFinite(row.Appliances)||row.Appliances<0))throw new ContractError('INVALID_TARGET','Appliances');
 assertSchema(row,rowSchema);return values;
}
export function createPredictor(model,modelSchema,rowSchema){
 for(const [v,n]of [[model,'model.json'],[modelSchema,'schemas/model.schema.json'],[rowSchema,'schemas/external-row.schema.json']])if(!verifiedArtifact(v,n))throw new ContractError('UNVERIFIED_ARTIFACT');
 validateModel(model,modelSchema);
 function predictOne(row){
  const values=orderedInput(row,rowSchema),prep=model.preprocessing,effective=[],scaled=[],imputed=[];
  let predicted=model.intercept;
  for(let i=0;i<FEATURES.length;i++){
   const value=values[i]===null?prep.imputer.statistics[i]:values[i];if(values[i]===null)imputed.push(FEATURES[i]);effective.push(value);
   const z=(value-prep.scaler.mean[i])/prep.scaler.scale[i];scaled.push(z);
   if(!Number.isFinite(z))throw new ContractError('NONFINITE_RESULT',FEATURES[i]);
   predicted+=z*model.coefficients[i];if(!Number.isFinite(predicted))throw new ContractError('NONFINITE_RESULT',FEATURES[i]);
  }
  return {schema_version:'1.0.0',model_version:VERSION,unit:'Wh',date:row.date,...interval(predicted,model.prediction_interval.q),imputed_features:imputed,ordered_inputs:values,effective_inputs:effective,scaled_inputs:scaled};
 }
 return Object.freeze({predictOne,model,model_sha256:verifiedHash(model)});
}
export async function loadEngine(loader=createLoader()){
 const [model,modelSchema,rowSchema]=await Promise.all(['model.json','schemas/model.schema.json','schemas/external-row.schema.json'].map(loader.load));
 return createPredictor(model,modelSchema,rowSchema);
}
export async function loadExamples(loader,engine){
 const [fixtures,schema]=await Promise.all(['parity_fixtures.json','schemas/parity-fixtures.schema.json'].map(loader.load));assertSchema(fixtures,schema);
 if(fixtures.model_version!==engine.model.model_version||fixtures.public_model_sha256!==engine.model_sha256||fixtures.source_model_sha256!==engine.model.source_artifact_sha256||!fixtures.fixtures.length)throw new ContractError('FIXTURE_VERSION_MISMATCH');
 return fixtures;
}
export function comparison(baseline,current){
 const delta=current.predicted-baseline.predicted;
 if(!Number.isFinite(delta))throw new ContractError('NONFINITE_RESULT');
 const percent=baseline.predicted>0?delta/baseline.predicted*100:null;
 return {delta,percent:percent!==null&&Number.isFinite(percent)?percent:null};
}
