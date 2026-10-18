// Validator for the keywords used by the pinned C1/C3 schemas, not a general Draft engine.
export class ContractError extends Error {
 constructor(code,field=null){super(code);this.name='ContractError';this.code=code;this.field=field;}
}
const equal=(a,b)=>a===b || !!a&&!!b&&typeof a==='object'&&typeof b==='object'&&Array.isArray(a)===Array.isArray(b)&&Object.keys(a).length===Object.keys(b).length&&Object.keys(a).every(k=>Object.hasOwn(b,k)&&equal(a[k],b[k]));
const typed=(v,t)=>t==='null'?v===null:t==='object'?v!==null&&typeof v==='object'&&!Array.isArray(v):t==='array'?Array.isArray(v):t==='integer'?Number.isInteger(v):t==='number'?typeof v==='number'&&Number.isFinite(v):typeof v===t;
export function assertSchema(value,schema,{root=schema,path=''}={}){
 const fail=()=>{throw new ContractError('SCHEMA_MISMATCH',path||null);};
 if(schema.$ref){if(!/^#\/(?:\$defs|definitions)\/[A-Za-z_-]+$/.test(schema.$ref))fail();const [,group,key]=schema.$ref.split('/');if(!root[group]?.[key])fail();return assertSchema(value,root[group][key],{root,path});}
 if(Object.hasOwn(schema,'const')&&!equal(value,schema.const))fail();
 if(schema.enum&&!schema.enum.some(v=>equal(v,value)))fail();
 if(schema.type&&!(Array.isArray(schema.type)?schema.type:[schema.type]).some(t=>typed(value,t)))fail();
 if(typeof value==='number'){if(!Number.isFinite(value))fail();if(schema.minimum!==undefined&&value<schema.minimum)fail();if(schema.maximum!==undefined&&value>schema.maximum)fail();if(schema.exclusiveMinimum!==undefined&&value<=schema.exclusiveMinimum)fail();}
 if(typeof value==='string'){if(schema.pattern&&!new RegExp(schema.pattern,'u').test(value))fail();if(schema.minLength!==undefined&&value.length<schema.minLength)fail();}
 if(Array.isArray(value)){if(schema.minItems!==undefined&&value.length<schema.minItems)fail();if(schema.maxItems!==undefined&&value.length>schema.maxItems)fail();if(schema.items)value.forEach((v,i)=>assertSchema(v,schema.items,{root,path:path+'/'+i}));}
 else if(value!==null&&typeof value==='object'){
  for(const key of schema.required||[])if(!Object.hasOwn(value,key))fail();
  for(const [key,v]of Object.entries(value)){if(schema.properties&&Object.hasOwn(schema.properties,key))assertSchema(v,schema.properties[key],{root,path:path+'/'+key});else if(schema.additionalProperties===false)fail();}
 }
 return value;
}
