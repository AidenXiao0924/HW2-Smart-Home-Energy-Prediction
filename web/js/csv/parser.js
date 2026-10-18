import {CsvError,LIMITS,checkAbort,yieldUI} from './contracts.js';
export const PARSER_VERSION='project-csv-1.0.0';
// Strict comma-delimited RFC-style state machine. Records preserve logical and physical row IDs.
export async function parseCSV(input,{signal,onProgress=()=>{},yieldFn=yieldUI}={}){
 const text=input.replace(/^\uFEFF/,'');let cell='',cells=[],state='start',line=1,start=1,ended=false;const records=[];
 function pushCell(){if(cells.length>=LIMITS.columns)throw new CsvError('TOO_MANY_COLUMNS');cells.push(cell);cell='';state='start';}
 function pushRow(){pushCell();records.push({cells,line_number:start});cells=[];if(records.length>LIMITS.rows+1)throw new CsvError('TOO_MANY_ROWS');}
 for(let i=0;i<text.length;i++){
  if(i%32768===0){checkAbort(signal);onProgress({phase:'PARSING',current:i,total:text.length});await yieldFn();}
  const c=text[i];ended=false;
  if(state==='quoted'){
   if(c==='"'){if(text[i+1]==='"'){cell+='"';i++;}else state='after';}
   else {cell+=c;if(c==='\n'||c==='\r'&&text[i+1]!=='\n')line++;}
  }else if(c===','){pushCell();}
  else if(c==='\r'||c==='\n'){pushRow();if(c==='\r'&&text[i+1]==='\n')i++;line++;start=line;ended=true;}
  else if(c==='"'){if(state!=='start'||cell!=='')throw new CsvError('CSV_PARSE_ERROR');state='quoted';}
  else {if(state==='after')throw new CsvError('CSV_PARSE_ERROR');cell+=c;state='bare';}
  if(cell.length>LIMITS.cell)throw new CsvError('CELL_TOO_LONG');
 }
 checkAbort(signal);if(state==='quoted')throw new CsvError('CSV_PARSE_ERROR');if(!ended&&(cell!==''||cells.length||state!=='start'))pushRow();
 if(!records.length)throw new CsvError('EMPTY_CSV');
 const headers=records.shift().cells.map(v=>v.trim());if(headers.some(v=>!v))throw new CsvError('EMPTY_HEADER');if(new Set(headers).size!==headers.length)throw new CsvError('DUPLICATE_HEADER');
 return {headers,rows:records.map((r,i)=>({...r,row_number:i+1})),row_count:records.length};
}
