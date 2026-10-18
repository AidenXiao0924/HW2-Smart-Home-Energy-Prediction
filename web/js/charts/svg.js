import {el,state} from '../core/ui.js';
const NS='http://www.w3.org/2000/svg',observers=new Set(),owners=new WeakMap();
const colors=['#83b9ff','#5ce1bf','#f3c66b','#d3a5ff','#ffb4b4'];
export const activeCharts=()=>observers.size;
export function disposeCharts(){for(const o of observers)o.disconnect();observers.clear();}
function svgEl(tag,attrs={},text){const n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
export function dataTable(title,heads,rows,id){
 const wrap=el('div',undefined,'table-wrap');wrap.tabIndex=0;wrap.setAttribute('role','region');wrap.setAttribute('aria-label',title+'，可橫向捲動');
 const table=el('table');if(id)table.dataset.table=id;table.append(el('caption',title));const thead=el('thead'),tr=el('tr');for(const h of heads){const th=el('th',h);th.scope='col';tr.append(th);}thead.append(tr);table.append(thead);const body=el('tbody');
 for(const [i,row]of rows.entries()){const tr=el('tr');tr.dataset.row=String(i);for(const v of row)tr.append(el('td',v));body.append(tr);}table.append(body);wrap.append(table);return wrap;
}
export function plot(host,{id,title,split,source,description,xLabel,yLabel,series,kind='line',xTick,zero=false}){
 if(!host.isConnected)return;
 for(const o of [...observers])if(!o.host.isConnected){o.disconnect();observers.delete(o);}
 const previous=owners.get(host);if(previous){previous.disconnect();observers.delete(previous);}
 if(!series.some(s=>s.points.length)){state(host,'EMPTY','沒有符合篩選條件的圖表資料。');return;}
 host.replaceChildren();host.dataset.state='SUCCESS';host.dataset.chart=id;host.dataset.split=split;
 const figure=el('figure',undefined,'research-chart');const heading=el('figcaption',`${split} · ${title}`);figure.append(heading,el('p',description,'chart-description'));
 const legend=el('div',undefined,'chart-legend');series.forEach((s,i)=>{const item=el('span',s.name);item.dataset.color=String(i%5);legend.append(item);});figure.append(legend);
 const box=el('div',undefined,'svg-box');figure.append(box);host.append(figure);
 function draw(){
  const width=Math.max(260,Math.min(1000,box.clientWidth)),height=320,left=62,right=20,top=20,bottom=65;
  const points=series.flatMap(s=>s.points);let xmin=Math.min(...points.map(p=>p.x)),xmax=Math.max(...points.map(p=>p.x)),ymin=Math.min(...points.map(p=>p.y)),ymax=Math.max(...points.map(p=>p.y));
  if(zero)ymin=Math.min(0,ymin);if(xmin===xmax){xmin-=.5;xmax+=.5;}if(ymin===ymax){ymin-=1;ymax+=1;}const pad=(ymax-ymin)*.05;ymin-=pad;ymax+=pad;
  const x=v=>left+(v-xmin)/(xmax-xmin)*(width-left-right),y=v=>height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom);
  const svg=svgEl('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':`${title}。${description}`});svg.append(svgEl('title',{},`${split} · ${title}`));
  const ticks=width<500?3:5;
  for(let i=0;i<=ticks;i++){const a=xmin+(xmax-xmin)*i/ticks,b=ymin+(ymax-ymin)*i/ticks;svg.append(svgEl('line',{x1:left,x2:width-right,y1:y(b),y2:y(b),stroke:'#314359'}),svgEl('text',{x:left-8,y:y(b)+4,'text-anchor':'end'},Number(b.toPrecision(3)).toString()),svgEl('text',{x:x(a),y:height-bottom+24,class:'x-tick','text-anchor':i===0?'start':i===ticks?'end':'middle'},xTick?(width<500?xTick(a).slice(0,5):xTick(a)):Number(a.toPrecision(3)).toString()));}
  svg.append(svgEl('text',{x:(width+left-right)/2,y:height-8,'text-anchor':'middle'},xLabel),svgEl('text',{transform:`translate(15 ${(height-bottom)/2}) rotate(-90)`,'text-anchor':'middle'},yLabel));
  series.forEach((s,i)=>{const color=colors[i%colors.length],g=svgEl('g',{'data-series':s.name});
   if(kind==='line')g.append(svgEl('polyline',{points:s.points.map(p=>`${x(p.x)},${y(p.y)}`).join(' '),fill:'none',stroke:color,'stroke-width':1.7}));
   for(const p of s.points){let mark;
    if(kind==='bar'){const bw=Math.min(22,(width-left-right)/(s.points.length+1)/series.length*.7);mark=svgEl('rect',{x:x(p.x)-bw/2+i*bw,y:Math.min(y(p.y),y(0)),width:bw,height:Math.abs(y(p.y)-y(0)),fill:color,opacity:.85});}
    else {mark=svgEl('circle',{cx:x(p.x),cy:y(p.y),r:kind==='scatter'?2.7:1.8,fill:color,opacity:kind==='scatter'?.65:1});}
    mark.append(svgEl('title',{},`${s.name} · ${p.label??p.x}: ${p.y}`));g.append(mark);
   }svg.append(g);
  });box.replaceChildren(svg);
 }
 const details=el('details');details.append(el('summary','檢視圖表資料表（文字替代）'));const rows=series.flatMap(s=>s.points.map(p=>[s.name,p.label??p.x,String(p.y)]));details.append(dataTable(`${split} · ${title} · 原值`,['系列',xLabel,yLabel],rows));figure.append(details,el('p',`來源：${source}`,'source-note'));
 const observer=new ResizeObserver(draw);observer.host=host;owners.set(host,observer);observers.add(observer);observer.observe(box);draw();
}
