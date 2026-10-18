import {CsvError} from './contracts.js';
const decimal=/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
function numeric(value,code,row,field){if(!decimal.test(value))throw new CsvError(code,row,field);const n=Number(value);if(!Number.isFinite(n))throw new CsvError(code,row,field);return n;}
export function validateHeaders(headers,model){for(const f of model.input_contract.required_columns)if(!headers.includes(f))throw new CsvError('MISSING_COLUMN',null,f);}
export function adaptRow(record,headers,model){
 const n=record.row_number;if(record.cells.length!==headers.length)throw new CsvError('CSV_PARSE_ERROR',n);
 const cells=Object.fromEntries(headers.map((f,i)=>[f,record.cells[i]])),row={date:cells.date.trim()},extra=Object.create(null);
 for(const f of model.feature_order.slice(0,-1)){const text=cells[f].trim();row[f]=text===''||text==='null'?null:numeric(text,'NONFINITE_OR_TYPE',n,f);}
 if(Object.hasOwn(cells,'Hour'))row.Hour=numeric(cells.Hour.trim(),'TIME_FEATURE_MISMATCH',n,'Hour');
 if(Object.hasOwn(cells,'Appliances')&&cells.Appliances.trim()!==''){row.Appliances=numeric(cells.Appliances.trim(),'INVALID_TARGET',n,'Appliances');if(row.Appliances<0)throw new CsvError('INVALID_TARGET',n,'Appliances');}
 const known=new Set([...model.feature_order,'date','Appliances']);for(const [f,v]of Object.entries(cells))if(!known.has(f))extra[f]=v;
 return {row,extra};
}
