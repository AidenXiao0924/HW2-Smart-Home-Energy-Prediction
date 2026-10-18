import {checkRange} from '../core/training-data.js';import {renderOOD} from './training.js';
import {el,card,state} from '../core/ui.js';
import {FEATURE_MEANINGS,fmt} from '../core/research-data.js';
import {dataTable} from '../charts/svg.js';
import {sourceCalendar,fromDateTimeControl} from '../inference/time.js';
import {comparison,ContractError} from '../inference/engine.js';
const messages={MISSING_COLUMN:'必要欄位缺失。',NONFINITE_OR_TYPE:'請輸入有限數值；空白須明確勾選缺值。',INVALID_DATE:'請選擇合法的來源當地日期與時間。',TIME_FEATURE_MISMATCH:'Hour 必須為0–23整數，且與來源時間一致。',NONFINITE_RESULT:'輸入造成數值溢位，無法產生有限結果。',INVALID_INPUT_OBJECT:'必須提供具名欄位物件。',INVALID_DELTA:'敏感度變化量必須是大於零的有限數值。'};
const describe=e=>`${e.code||'INVALID_INPUT'}${e.field?' · '+e.field:''}：${messages[e.code]||'輸入或模型契約無法驗證。'}`;
const display=v=>v===null?'null（明確缺值）':fmt(v,6);
const hint=f=>f==='Hour'?'合法值：0–23整數，與date一致。':f.startsWith('RH_')?'接受有限數值／明確null；物理參考0–100%，不是訓練範圍。':['Windspeed','Visibility'].includes(f)?'接受有限數值／明確null；物理參考非負，不是訓練範圍。':'接受有限數值／明確null；模型未提供可核實的訓練Min/Max。';
export function renderSimulator(host,engine,fixtures,rangeBundle=null){
 const baseline=structuredClone(fixtures.fixtures[0].input),model=engine.model,fields=new Map();let revision=0;
 const intro=card('What-If Simulator · Frozen MLR','同時段能源估計，不是未來 Forecast，也不是因果推論或真實節能效果。推論只在本機瀏覽器執行。');
 intro.append(el('p',`模型 ${model.model_version} · ${model.feature_order.length} 特徵 · q=${model.prediction_interval.q} Wh`,'mono'),el('p',`初始方案：第一筆真實 Calibration parity fixture（${baseline.date}）。這是歷史示範輸入，不是使用者即時感測值，也不是重新校準。`),el('p',`來源：C3 parity_fixtures.json → ${fixtures.source.path} · SHA-256 ${fixtures.source.sha256}`,'source-note'));host.append(intro);
 if(!rangeBundle)intro.append(el('p','Training OOD檢查不可用：未驗證範圍來源；合法推論仍可執行，不能宣稱無OOD。','warning-text'));
 const form=el('form',undefined,'simulator-form');form.noValidate=true;form.setAttribute('aria-label','正式15特徵輸入');host.append(form);
 const result=el('section');result.id='prediction-result';result.setAttribute('aria-label','估計與What-If結果');result.setAttribute('aria-live','polite');result.setAttribute('aria-atomic','false');
 const sensitivity=el('section');sensitivity.setAttribute('aria-label','敏感度結果');sensitivity.setAttribute('aria-live','polite');
 function invalidate(){revision++;delete result.dataset.predicted;state(result,'UNAVAILABLE','輸入已變更，請重新執行 Predict；先前結果已清除。');state(sensitivity,'UNAVAILABLE','敏感度結果尚未計算。');}
 const groups=[['廚房感測',['T1','RH_1']],['室內各區域感測',['T3','RH_3','T4','T5','RH_5','T7','RH_8','T9','RH_9']],['室外氣象',['Windspeed','Visibility','Tdewpoint']],['來源日曆時間',['Hour']]];
 const date=el('input');date.id='sim-date';date.type='datetime-local';date.step='1';date.required=true;date.setAttribute('aria-describedby','sim-date-help');
 for(const [name,names]of groups){const group=el('fieldset');group.append(el('legend',name));const grid=el('div',undefined,'sim-input-grid');group.append(grid);form.append(group);
  if(names.includes('Hour')){const w=el('div',undefined,'sim-field');const l=el('label','date · 來源當地日期時間');l.htmlFor=date.id;const help=el('p','固定來源日曆時間（Belgium dataset），不轉UTC、不套用瀏覽器時區。C1契約要求date；Hour可編輯並同步此時間。','muted');help.id='sim-date-help';w.append(l,date,help);grid.append(w);}
  for(const f of names){const metadata=model.features.find(v=>v.name===f),wrap=el('div',undefined,'sim-field'),label=el('label',`${f} · ${FEATURE_MEANINGS[f]} (${metadata.unit})`),input=el('input'),help=el('p',(rangeBundle?`${f} Training Min/Max：${rangeBundle.data.ranges.find(r=>r.feature===f).min} — ${rangeBundle.data.ranges.find(r=>r.feature===f).max} ${model.features.find(r=>r.name===f).unit}；超界只提醒，不裁切。`:hint(f)),'muted');
   input.id='sim-'+f;input.name=f;input.type='number';input.step=f==='Hour'?'1':'any';input.required=true;label.htmlFor=input.id;help.id=input.id+'-help';input.setAttribute('aria-describedby',help.id);if(f==='Hour'){input.min='0';input.max='23';}
   wrap.append(label,input,help);let missing=null;
   if(f!=='Hour'){missing=el('input');missing.type='checkbox';missing.id='missing-'+f;const ml=el('label','明確缺值（null，使用凍結Median）','null-label');ml.htmlFor=missing.id;ml.prepend(missing);wrap.append(ml);missing.addEventListener('change',()=>{input.disabled=missing.checked;invalidate();});}
   fields.set(f,{input,missing});grid.append(wrap);
  }
 }
 date.addEventListener('input',()=>{try{fields.get('Hour').input.value=String(sourceCalendar(fromDateTimeControl(date.value)).Hour);}catch{};invalidate();});
 fields.get('Hour').input.addEventListener('input',()=>{const hour=fields.get('Hour').input.valueAsNumber;if(Number.isInteger(hour)&&hour>=0&&hour<=23&&/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(date.value))date.value=date.value.slice(0,11)+String(hour).padStart(2,'0')+date.value.slice(13);});
 for(const {input}of fields.values())input.addEventListener('input',invalidate);
 const actions=el('div',undefined,'sim-actions'),predict=el('button','Predict · 計算估計','primary'),reset=el('button','Reset · 還原真實範例','secondary');predict.type='submit';reset.type='button';actions.append(predict,reset);form.append(actions);host.append(result);
 function readInput(){const row={date:fromDateTimeControl(date.value)};date.removeAttribute('aria-invalid');for(const [f,{input,missing}]of fields){input.removeAttribute('aria-invalid');if(missing?.checked)row[f]=null;else {const n=input.valueAsNumber;if(!Number.isFinite(n)){input.setAttribute('aria-invalid','true');throw new ContractError('NONFINITE_OR_TYPE',f);}row[f]=n;}}return row;}
 function showResult(row,current,base){
  result.replaceChildren();result.dataset.state='SUCCESS';result.dataset.predicted=String(current.predicted);const c=card('Frozen Model Result',`${current.model_version} · Wh · 當前來源時間 ${row.date}`);
  const point=el('p',fmt(current.predicted,8)+' Wh','metric');point.dataset.result='predicted';c.append(point);
  if(current.interval_valid){const p=el('p',`名目95% Prediction Interval：${fmt(current.lower,8)} — ${fmt(current.upper,8)} Wh`);p.dataset.result='interval';c.append(p);}else{const box=el('div');state(box,'ERROR','INVALID_INTERVAL · 無有效Prediction Interval；不得交換端點或顯示為有效95%區間。');c.append(box);const detail=el('details');detail.append(el('summary','無效端點診斷（非有效區間）'),el('p',`Lower=${current.lower}，Upper=${current.upper}`));c.append(detail);}
  const warn=el('div');warn.dataset.result='warnings';if(current.warnings.length){warn.setAttribute('role','alert');for(const code of current.warnings)warn.append(el('p',code,'warning-text'));}else warn.append(el('p','未觸發模型警告。','muted'));c.append(warn);
  c.append(el('p',current.imputed_features.length?'Median填補：'+current.imputed_features.join(', '):'沒有使用Median填補。','muted'));
  const diff=comparison(base,current);const compare=el('p',`真實範例基準 ${fmt(base.predicted,8)} Wh；當前−基準 ${fmt(diff.delta,8)} Wh；變化百分比 ${diff.percent===null?'不適用（基準≤0或非有限比例）':fmt(diff.percent,4)+'%'}`);compare.dataset.result='comparison';c.append(compare);
  const cautions=[];for(const [f,v]of Object.entries(row))if(typeof v==='number'&&((f.startsWith('RH_')&&(v<0||v>100))||(['Windspeed','Visibility'].includes(f)&&v<0)))cautions.push(f);
  if(cautions.length)c.append(el('p','輸入參考範圍提醒（非模型Warning／非訓練OOD判定）：'+cautions.join(', '),'warning-text'));
  c.append(dataTable('當前輸入與真實範例比較（非因果效果）',['欄位','單位','範例輸入','當前輸入','有效值（含填補）','有效值變化'],model.feature_order.map((f,i)=>[f,model.features[i].unit,display(baseline[f]),display(row[f]),display(current.effective_inputs[i]),display(current.effective_inputs[i]-base.effective_inputs[i])]),'sim-comparison'));
  renderOOD(c,checkRange(row,rangeBundle),'sim-ood');result.append(c);
 }
 function report(error,where=result){delete where.dataset.predicted;state(where,'INVALID_INPUT',describe(error));if(error.field==='date')date.setAttribute('aria-invalid','true');else if(fields.has(error.field))fields.get(error.field).input.setAttribute('aria-invalid','true');}
 form.addEventListener('submit',e=>{e.preventDefault();const token=++revision;try{const row=readInput(),base=engine.predictOne(baseline),current=engine.predictOne(row);if(token===revision&&result.isConnected)showResult(row,current,base);}catch(error){report(error);}});
 const controls=card('單一特徵敏感度 · Frozen Engine','對當前輸入選一特徵加／減Δ，各自重新呼叫同一Engine。主表可同時變更多欄位與範例比較；這不是因果／節能／Forecast結論。');
 const layout=el('div',undefined,'sim-actions'),selector=el('select'),sl=el('label','變動特徵'),delta=el('input'),dl=el('label','正變化量 Δ（使用所選特徵原始單位）'),run=el('button','比較 −Δ／當前／＋Δ','secondary');selector.id='sensitivity-feature';sl.htmlFor=selector.id;
 for(const metadata of model.features.filter(f=>f.name!=='Hour')){const option=el('option',`${metadata.name} · ${FEATURE_MEANINGS[metadata.name]} (${metadata.unit})`);option.value=metadata.name;selector.append(option);}delta.id='sensitivity-delta';delta.type='number';delta.step='any';delta.value='1';dl.htmlFor=delta.id;run.type='button';run.id='sensitivity-run';layout.append(sl,selector,dl,delta,run);controls.append(layout,sensitivity);host.append(controls);
 run.addEventListener('click',()=>{try{const row=readInput(),f=selector.value,d=delta.valueAsNumber;if(!Number.isFinite(d)||d<=0)throw new ContractError('INVALID_DELTA');const current=engine.predictOne(row),i=model.feature_order.indexOf(f),effective=current.effective_inputs[i];const scenarios=[['−Δ',{...row,[f]:effective-d}],['當前',row],['＋Δ',{...row,[f]:effective+d}]];const outputs=scenarios.map(([name,input])=>({name,input,result:engine.predictOne(input)}));sensitivity.replaceChildren();sensitivity.dataset.state='SUCCESS';sensitivity.append(dataTable('What-If 敏感度（每列均重新推論）',['情境',f+' ('+model.features[i].unit+')','Point Wh','相對當前 ΔWh','區間狀態','Warning'],outputs.map(o=>[o.name,display(o.input[f]),fmt(o.result.predicted,8),fmt(comparison(current,o.result).delta,8),o.result.interval_valid?'有效':'INVALID_INTERVAL',o.result.warnings.join(', ')||'—']),'sim-sensitivity'));}catch(error){report(error,sensitivity);}});
 for(const input of [selector,delta])input.addEventListener('input',()=>state(sensitivity,'UNAVAILABLE','敏感度設定已變更，請重新比較。'));
 const limits=card('Prediction Interval ≠ Confidence Interval','本區間用於單筆能耗估計的不確定性，不是平均反應的Confidence Interval。95%為名目水準；時間序列不保證交換性，不能保證未來嚴格95% Coverage。Test Observed Coverage 不代表每筆未來預測的保證。');host.append(limits);
 function restore(){date.value=baseline.date.replace(' ','T');for(const [f,{input,missing}]of fields){input.value=String(baseline[f]);input.disabled=false;input.removeAttribute('aria-invalid');if(missing)missing.checked=false;}date.removeAttribute('aria-invalid');selector.value=model.feature_order[0];delta.value='1';invalidate();}
 reset.addEventListener('click',restore);restore();host.dataset.simulator='READY';
}
