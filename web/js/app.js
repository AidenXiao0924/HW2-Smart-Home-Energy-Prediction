import {loadTraining} from './core/training-data.js';
import {PAGES} from './core/config.js';
import {createLoader} from './core/loader.js';
import {el,state} from './core/ui.js';
import {renderOverview,renderData,renderEvaluation} from './pages/research.js';
import {renderFuture} from './pages/future.js';
import {disposeCharts} from './charts/svg.js';
import {loadEngine,loadExamples} from './inference/engine.js';
import {renderSimulator} from './pages/simulator.js';
import {renderCSV} from './pages/csv.js';
const loader=createLoader(), content=document.querySelector('#page-content');
const nav=document.querySelector('#navigation'), toggle=document.querySelector('#menu-toggle'), sidebar=document.querySelector('#sidebar'),scrim=document.querySelector('#scrim');
let generation=0,cleanupPage=()=>{};
document.querySelector('.skip').addEventListener('click',e=>{e.preventDefault();document.querySelector('#main').focus();});
for(const [id,label,sub,num] of PAGES){const a=el('a');a.href=`#${id}`;a.dataset.page=id;a.append(el('span',num,'nav-number'),el('span',label));nav.append(a);}
function menu(open){toggle.setAttribute('aria-expanded',String(open));toggle.textContent=open?'關閉導覽':'開啟導覽';sidebar.classList.toggle('is-open',open);scrim.hidden=!open;sidebar.inert=!open&&matchMedia('(max-width: 760px)').matches;}
toggle.addEventListener('click',()=>menu(toggle.getAttribute('aria-expanded')!=='true'));
scrim.addEventListener('click',()=>{menu(false);toggle.focus();});
document.addEventListener('keydown',e=>{
  if(toggle.getAttribute('aria-expanded')!=='true')return;
  if(e.key==='Escape'){menu(false);toggle.focus();}
  if(e.key==='Tab'){const list=[toggle,...sidebar.querySelectorAll('a')];const i=list.indexOf(document.activeElement);if(e.shiftKey&&i===0){e.preventDefault();list.at(-1).focus();}else if(!e.shiftKey&&i===list.length-1){e.preventDefault();toggle.focus();}}
});
const media=matchMedia('(max-width: 760px)');
function syncMenu(){menu(false);sidebar.inert=media.matches;}
media.addEventListener('change',syncMenu);syncMenu();
toggle.addEventListener('click',()=>{sidebar.inert=toggle.getAttribute('aria-expanded')!=='true';});
function closeMenu(){menu(false);sidebar.inert=media.matches;}
async function attribution(){
  const target=document.querySelector('#attribution');
  try{const a=await loader.load('attribution.json');target.replaceChildren(el('p',`${a.creator} · ${a.dataset_title}`));
    for(const [text,url] of [['UCI Dataset',a.source_url],[a.license,a.license_url]]){const link=el('a',text);link.href=url;link.rel='noopener noreferrer';target.append(link,document.createTextNode(' · '));}target.append(el('p',a.modifications,'attribution-detail'));
  }catch{state(target,'ERROR','資料來源無法驗證，請檢查公開資料包。');}
}
async function route(focus=false){
  const token=++generation, id=location.hash.slice(1)||'overview';cleanupPage();cleanupPage=()=>{};closeMenu();disposeCharts();
  const page=PAGES.find(p=>p[0]===id);
  nav.querySelectorAll('a').forEach(a=>{if(a.dataset.page===id)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  document.querySelector('#page-title').textContent=page?.[1]||'找不到頁面';document.querySelector('#page-subtitle').textContent=page?.[2]||'請使用導覽回到研究總覽';document.title=`${page?.[1]||'Unavailable'} | Home Energy Lab`;
  if(focus)document.querySelector('#main').focus();
  content.setAttribute('aria-busy','true');state(content,'LOADING','正在驗證正式研究資料…');
  try{
    if(!page){state(content,'UNAVAILABLE','此頁面不存在。請選擇左側導覽。');return;}
    const summary=await loader.load('research_summary.json');
    let engine,examples,rangeBundle=null;
    if(id==='simulator'){engine=await loadEngine(loader);examples=await loadExamples(loader,engine);}
    if(id==='csv')engine=await loadEngine(loader);
    if(id==='simulator'||id==='csv')rangeBundle=await loadTraining('ranges').catch(()=>null);
    if(token!==generation)return;
    content.replaceChildren();content.dataset.state='SUCCESS';
    if(id==='overview')renderOverview(content,summary);
    else if(id==='data')renderData(content,summary,loader);
    else if(id==='evaluation')renderEvaluation(content,summary,loader);
    else if(id==='simulator')renderSimulator(content,engine,examples,rangeBundle);
    else if(id==='csv')cleanupPage=renderCSV(content,engine,rangeBundle);
    else renderFuture(content,id);
  }catch(e){if(token===generation){state(content,e.code==='EMPTY'?'EMPTY':'ERROR',`資料載入或契約驗證失敗（${e.code||'UNAVAILABLE'}）。請確認正式公開資料包；不提供替代數值。`);const retry=el('button','重新載入','secondary');retry.addEventListener('click',()=>route());content.append(retry);}}
  finally{if(token===generation)content.setAttribute('aria-busy','false');}
}
window.addEventListener('hashchange',()=>route(true));attribution();route();
