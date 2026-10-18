// Browser-only QA, no research model execution. PLAYWRIGHT_MODULE may point to a bundled package.
import {createRequire} from 'node:module';
import {createServer} from 'node:http';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=process.env.WEB_ROOT||fileURLToPath(new URL('../../web/',import.meta.url)),out=process.env.E2E_OUTPUT||fileURLToPath(new URL('../../reports/tests/m3a/',import.meta.url));
await mkdir(out,{recursive:true});const results=[];
const server=createServer(async(req,res)=>{try{let name=decodeURIComponent(new URL(req.url,'http://preview.invalid').pathname);if(name.startsWith('/HW2_Smart-Home-Energy/'))name=name.slice('/HW2_Smart-Home-Energy'.length);if(name==='/')name='/index.html';const file=path.resolve(root,'.'+name);if(!file.startsWith(root))throw Error();const data=await readFile(file);res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json'})[path.extname(file)]||'application/octet-stream');res.end(data);}catch{res.statusCode=404;res.end('Not found');}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin=`http://127.0.0.1:${server.address().port}`;
const browser=await chromium.launch({channel:process.env.PLAYWRIGHT_CHANNEL||'msedge',headless:true});
async function check(name,fn){const start=Date.now();try{await fn();results.push({name,status:'PASS',ms:Date.now()-start});}catch(e){results.push({name,status:'FAIL',ms:Date.now()-start,error:String(e)});}}
async function ready(page){await page.locator('#page-content[data-state="SUCCESS"]').waitFor();}
const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();
const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
try{
 for(const prefix of ['/','/HW2_Smart-Home-Energy/'])await check('five-page navigation under '+prefix,async()=>{
  await page.goto(origin+prefix);await ready(page);
  for(const [id,title] of [['overview','Overview'],['data','Data & Features'],['evaluation','Model Evaluation'],['simulator','What-If Simulator'],['csv','External CSV Prediction']]){
   await page.locator(`nav a[href="#${id}"]`).click();await ready(page);assert.equal(await page.locator('h1').textContent(),title);assert.equal(await page.locator('nav a[aria-current="page"]').count(),1);
   if(id==='csv'){await page.locator('[data-csv="READY"]').waitFor();assert(await page.locator('#csv-file').isEnabled());}if(id==='simulator'){await page.locator('[data-simulator="READY"]').waitFor();assert(await page.locator('#sim-T1').isEnabled());}
  }
 });
 await check('homepage JSON budget and attribution',async()=>{await page.goto(origin+'/HW2_Smart-Home-Energy/#overview');await ready(page);requests.length=0;await page.reload();await ready(page);await page.locator('#attribution a').first().waitFor();const initial=requests.filter(u=>u.endsWith('.json'));assert(!initial.some(u=>/charts\/|model\.json|parity|schema\.json/.test(u)));assert.equal(initial.length,3);assert((await page.locator('#attribution').textContent()).includes('CC BY 4.0'));});
 await check('lazy chart JSON loading, no evaluation',async()=>{await page.locator('nav a[href="#evaluation"]').click();await ready(page);await page.getByRole('button',{name:'驗證圖表資料連線'}).click();await page.getByText('三份圖表資料已通過', {exact:false}).waitFor();});
 for(const [name,width,height] of [['desktop',1440,1000],['tablet',820,1180],['mobile',390,844],['small-mobile',320,740]])await check('responsive '+name,async()=>{
  await page.setViewportSize({width,height});await page.goto(origin+'/HW2_Smart-Home-Energy/#overview');await ready(page);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  if(width<761){assert(await page.locator('#menu-toggle').isVisible());await page.locator('#menu-toggle').click();await page.locator('nav a[href="#data"]').click();await ready(page);assert.equal(await page.locator('#menu-toggle').getAttribute('aria-expanded'),'false');await page.locator('#menu-toggle').click();await page.keyboard.press('Escape');assert.equal(await page.locator('#menu-toggle').getAttribute('aria-expanded'),'false');assert(await page.locator('#sidebar').evaluate(e=>e.inert));}
  await page.goto(origin+'/HW2_Smart-Home-Energy/#overview');await ready(page);
  await page.locator('h1').click();await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(out,name+'.png'),fullPage:true,animations:'disabled'});
 });
 await check('keyboard drawer focus trap and current link',async()=>{await page.locator('#menu-toggle').focus();await page.keyboard.press('Enter');await page.keyboard.press('Tab');assert.equal(await page.evaluate(()=>document.activeElement.className),'brand');await page.keyboard.press('Shift+Tab');assert.equal(await page.evaluate(()=>document.activeElement.id),'menu-toggle');await page.keyboard.press('Shift+Tab');assert.equal(await page.evaluate(()=>document.activeElement.getAttribute('href')),'#csv');await page.keyboard.press('Escape');});
 await check('semantic accessibility and labels',async()=>{assert.equal(await page.locator('html').getAttribute('lang'),'zh-Hant');assert.equal(await page.locator('main').count(),1);assert.equal(await page.locator('h1').count(),1);assert.equal(await page.locator('nav[aria-label]').count(),1);await page.goto(origin+'/#simulator');await ready(page);assert.equal(await page.locator('label[for="sim-T1"]').count(),1);assert.equal(await page.locator('#sim-T1').getAttribute('aria-describedby'),'sim-T1-help');});
 await check('skip link keeps current route and moves focus',async()=>{await page.locator('.skip').focus();await page.keyboard.press('Enter');assert.equal(await page.evaluate(()=>document.activeElement.id),'main');assert.equal(new URL(page.url()).hash,'#simulator');await ready(page);});
 await check('text contrast tokens >=4.5',async()=>{const pairs=await page.evaluate(()=>{const c=getComputedStyle(document.documentElement);return ['--text','--muted','--actual','--estimated','--warning','--error'].flatMap(f=>['--bg','--surface','--raised'].map(b=>[f,b,c.getPropertyValue(f).trim(),c.getPropertyValue(b).trim()]));});function lum(s){const c=s.slice(1).match(/../g).map(x=>parseInt(x,16)/255).map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4);return c[0]*.2126+c[1]*.7152+c[2]*.0722;}for(const [f,b,fg,bg]of pairs){const l=[lum(fg),lum(bg)].sort((a,b)=>b-a);assert((l[0]+.05)/(l[1]+.05)>=4.5,`${f}/${b}`);}});
 await check('reduced motion respected',async()=>{await page.emulateMedia({reducedMotion:'reduce'});assert.equal(await page.locator('#sidebar').evaluate(e=>getComputedStyle(e).transitionDuration),'0s');});
 await check('missing JSON error no fallback',async()=>{await page.route('**/research_summary.json',r=>r.fulfill({status:404,body:'missing'}));await page.reload();await page.locator('#page-content[data-state="ERROR"]').waitFor();assert.equal(await page.locator('.metric').count(),0);await page.screenshot({path:path.join(out,'error.png'),fullPage:true});await page.unroute('**/research_summary.json');});
 await check('corrupt model blocks simulator',async()=>{await page.route('**/model.json',r=>r.fulfill({status:200,contentType:'application/json',body:'{}'}));await page.goto(origin+'/#simulator');await page.locator('#page-content[data-state="ERROR"]').waitFor();assert.equal(await page.locator('input').count(),0);await page.unroute('**/model.json');});
 await check('unknown page explicit unavailable',async()=>{await page.goto(origin+'/#missing');await page.locator('#page-content[data-state="UNAVAILABLE"]').waitFor();});
 await check('no browser uncaught errors',async()=>assert.deepEqual(errors,[]));
}finally{
 await browser.close();await new Promise(r=>server.close(r));
 const report={framework:'Playwright',version:require((process.env.PLAYWRIGHT_MODULE||'playwright')+'/package.json').version,browser_channel:process.env.PLAYWRIGHT_CHANNEL||'msedge',results,passed:results.filter(r=>r.status==='PASS').length,failed:results.filter(r=>r.status==='FAIL').length,limitations:'Basic semantic/contrast/keyboard checks; not a complete WCAG audit. No inference/parity/CSV/production deployment tests in M3-A.'};
 await writeFile(path.join(out,'browser.json'),JSON.stringify(report,null,2));
 const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
 await writeFile(path.join(out,'browser.xml'),`<testsuite name="m3a-browser" tests="${results.length}" failures="${report.failed}">${results.map(r=>`<testcase name="${esc(r.name)}" time="${r.ms/1000}">${r.status==='FAIL'?`<failure message="${esc(r.error)}"/>`:''}</testcase>`).join('')}</testsuite>`);
 console.log(JSON.stringify(report,null,2));if(report.failed)process.exitCode=1;
}
