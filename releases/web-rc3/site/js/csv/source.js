import {CsvError,LIMITS,checkAbort} from './contracts.js';
import {sha256} from '../core/loader.js';
export function publicURL(value){let url;try{url=new URL(value);}catch{throw new CsvError('UNSUPPORTED_URL');}
 if(url.protocol!=='https:'||url.username||url.password||url.hash||url.port||!/^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}$/i.test(url.hostname)||/\.(?:local|internal|invalid)$/i.test(url.hostname))throw new CsvError('UNSUPPORTED_URL');return url;
}
async function consume(reader,{signal,onProgress=()=>{},total=0}){
 const chunks=[];let length=0;const cancel=()=>{reader.cancel().catch(()=>{});};signal?.addEventListener('abort',cancel,{once:true});
 try{for(;;){checkAbort(signal);const {done,value}=await reader.read();checkAbort(signal);if(done)break;length+=value.byteLength;if(length>LIMITS.bytes)throw new CsvError('FILE_TOO_LARGE');chunks.push(value);onProgress({phase:'READING',current:length,total});}
  const bytes=new Uint8Array(length);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.byteLength;}
  let text;try{text=new TextDecoder('utf-8',{fatal:true}).decode(bytes);}catch{throw new CsvError('ENCODING_ERROR');}if(text.includes('\0'))throw new CsvError('ENCODING_ERROR');
  const digest=await sha256(bytes);checkAbort(signal);return {text,bytes:length,sha256:digest};
 }finally{signal?.removeEventListener('abort',cancel);await reader.cancel().catch(()=>{});}
}
export async function readLocal(file,options={}){
 checkAbort(options.signal);if(!file||typeof file.size!=='number')throw new CsvError('CSV_PARSE_ERROR');if(file.size>LIMITS.bytes)throw new CsvError('FILE_TOO_LARGE');
 return {...await consume(file.stream().getReader(),{...options,total:file.size}),source:{kind:'EXTERNAL_LOCAL_FILE',label:'使用者本機CSV（未上傳）'}};
}
export async function fetchCSV(value,{signal,onProgress,timeout=LIMITS.timeout,fetcher=globalThis.fetch}={}){
 const url=publicURL(value);checkAbort(signal);const controller=new AbortController();let expired=false;const stop=()=>controller.abort();signal?.addEventListener('abort',stop,{once:true});
 const timer=setTimeout(()=>{expired=true;controller.abort();},timeout);
 try{
  const response=await fetcher(url,{method:'GET',mode:'cors',credentials:'omit',redirect:'manual',cache:'no-store',referrerPolicy:'no-referrer',signal:controller.signal});
  if(response.type==='opaqueredirect'||response.status>=300&&response.status<400)throw new CsvError('REDIRECT_BLOCKED');
  if(response.type==='opaque')throw new CsvError('CORS_OR_NETWORK');if(!response.ok)throw new CsvError('HTTP_ERROR');
  if(response.url)publicURL(response.url);
  const total=Number(response.headers.get('content-length'))||0;if(total>LIMITS.bytes)throw new CsvError('FILE_TOO_LARGE');if(!response.body)throw new CsvError('CORS_OR_NETWORK');
  const data=await consume(response.body.getReader(),{signal:controller.signal,onProgress,total});
  return {...data,source:{kind:'EXTERNAL_HTTPS_USER_SPECIFIED',label:'使用者指定外部CSV · '+url.origin,redirected:false,response_type:response.type}};
 }catch(e){if(expired)throw new CsvError('TIMEOUT');if(signal?.aborted)throw new CsvError('CANCELLED');if(e instanceof CsvError)throw e;throw new CsvError('CORS_OR_NETWORK');}
 finally{clearTimeout(timer);signal?.removeEventListener('abort',stop);}
}
