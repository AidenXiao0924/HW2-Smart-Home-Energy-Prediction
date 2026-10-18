import {renderTrainingAnalytics} from './training.js';
import {renderDemo} from './demo.js';
import {el,card,state} from '../core/ui.js';
import {METHODS,PURPOSES,fmt,restriction,kpis,splitRows,cvRows,validationRows,featureRows,vifRows,seriesRows} from '../core/research-data.js';
import {plot,dataTable} from '../charts/svg.js';
const provenance=section=>`${section.source.path} · SHA-256 ${section.source.sha256}`;
function section(host,title,description,source){const c=card(title,description);if(source)c.append(el('p',provenance(source),'source-note'));host.append(c);return c;}
function table(host,title,heads,rows,id){host.append(dataTable(title,heads,rows,id));}
function metric(host,key,label,value,unit,split='TEST'){const c=card(label,`${split} · ${unit}`);const n=el('p',fmt(value, key==='r2'?6:4),'metric');n.dataset.metric=key;n.dataset.value=String(value);c.insertBefore(n,c.lastChild);host.append(c);}
function splits(host,s){const c=section(host,'Chronological split','依時間排序，禁止 Shuffle；比例為實際整數筆數占比。',s.splits);table(c,'ALL_RAW · 資料切分',['區段','筆數','實際 %','開始','結束','用途'],splitRows(s).map(r=>[r[0],String(r[1]),fmt(r[2],3),...r.slice(3)]),'splits');}
function select(host,id,label,options){const wrap=el('div');const l=el('label',label);l.htmlFor=id;const input=el('select');input.id=id;for(const [value,text] of options){const op=el('option',text);op.value=value;input.append(op);}wrap.append(l,input);host.append(wrap);return input;}
function chartSlot(host){const slot=el('div');host.append(slot);return slot;}
function chart(host,config){plot(host,config);}
export function renderOverview(container,s){
 const d=s.dataset.values,f=s.final_model.values;
 const hero=el('article',undefined,'hero');hero.append(el('span','SMART HOME ENERGY CONSUMPTION PREDICTION & ANALYTICS','eyebrow'),el('h2','理解居家能源，從可驗證的研究開始。'),el('p','研究目標：以同一觀測時間的感測器與環境資料估計 Appliances（Wh）。這是同時段能源消耗估計，不是未來 Forecast。'));container.append(hero);
 const info=section(container,`${f.model_version} · ${f.estimator}`,'凍結的正式 MLR；相關性與特徵選擇結果不代表因果。',s.final_model);info.append(el('p',`${METHODS[f.selection_method]} · ${f.feature_order.length} features · Final Fit ${f.training_rows.toLocaleString()} TRAIN_VALIDATION rows`));
 const grid=el('div',undefined,'grid metrics kpi-grid');for(const [key,label,v,u]of kpis(s))metric(grid,key,label,v,u);container.append(grid);container.append(el('p',`KPI 來源：${provenance(s.test)}；Observed Coverage 是完整 Test 的實際觀察值。`,'source-note'));
 const ds=section(container,'Dataset · ALL_RAW','Appliances Energy Prediction · 原始取樣間隔約 10 分鐘。',s.dataset);ds.append(el('p',`${d.rows.toLocaleString()} 筆 · ${d.columns} 原始欄位 · ${d.time_start} — ${d.time_end}`));
 splits(container,s);
 const process=section(container,'CRISP-DM · 六階段研究流程','模型研究已完成；網站展示分階段建置，公開部署尚未完成。');const ol=el('ol',undefined,'process');for(const t of ['Business Understanding — 問題與成功標準','Data Understanding — 真實來源與品質','Data Preparation — 時間切分與防洩漏','Modeling — 五種選擇器與 MLR','Evaluation — 凍結 Test 與區間診斷','Deployment — 靜態研究展示（建置中）'])ol.append(el('li',t));process.append(ol);
 const limits=section(container,'適用範圍與限制','單一住宅、有限季節與同時段感測資料；不能推論未來或其他住宅的保證效果。');for(const t of s.limitations)limits.append(el('p',t,'muted'));limits.append(el('p','高耗電峰值容易低估；時間相依、共線性與線性形式限制解讀。Test R² 有限，不宣稱高準確度。95% 是名目區間水準，不保證未來嚴格 95% 覆蓋。'));
}
export function renderData(container,s,loader){
 const d=s.dataset.values;
 const quality=section(container,'Dataset Summary & Quality · ALL_RAW','原始品質檢查保留異常觀測，不以刪除尖峰美化研究結果。',s.dataset);
 table(quality,'ALL_RAW · 品質結果',['項目','真實檢查值'],[['筆數',d.rows],['原始欄位',d.columns],['缺失值',Object.values(d.missing).reduce((a,b)=>a+b,0)],['重複列',d.duplicate_rows],['重複時間',d.duplicate_timestamps],['原始時間已排序',String(d.original_time_sorted)],['物理範圍違反總數',Object.values(d.physical_range_violations).reduce((a,b)=>a+b,0)],['開始',d.time_start],['結束',d.time_end]],'quality');
 const features=section(container,'候選特徵與排除規則',`${d.candidate_count} 個研究候選、${d.deployable_count} 個部署候選。rv1／rv2 皆排除；28 集合包含 lights，僅研究對照；K10 為 RESEARCH_ONLY。`,s.dataset);
 table(features,'候選特徵分類（依 PDD §3）',['分類','欄位','數量'],[['溫度','T1–T9',9],['相對濕度','RH_1–RH_9',9],['室外天氣','T_out, Press_mm_hg, RH_out, Windspeed, Visibility, Tdewpoint',6],['日曆衍生','Hour, DayOfWeek, IsWeekend',3],['照明（研究專用）','lights',1]]);
 features.append(el('p','原始欄位角色：Appliances（Wh）為 Target，date 為時間來源；rv1／rv2 為排除的隨機欄位。Hour=0–23，DayOfWeek=Monday 0 至 Sunday 6，IsWeekend 為週六／日。衍生採來源當地日曆時間，不套用瀏覽器時區。','muted'));
 const frozen=section(container,'正式 15 特徵 · TRAIN_VALIDATION','順序與單位直接讀正式 Model JSON；文字意義依已驗證報告及來源 Dataset。',s.final_model);const fslot=chartSlot(frozen);state(fslot,'LOADING','讀取正式特徵清單…');
 loader.load('model.json').then(m=>{if(!fslot.isConnected)return;fslot.replaceChildren();table(fslot,'TRAIN_VALIDATION · 正式特徵順序與單位',['順序','欄位','意義','單位'],featureRows(m),'features');fslot.dataset.state='SUCCESS';}).catch(()=>{if(fslot.isConnected)state(fslot,'ERROR','正式特徵資料無法驗證。');});
 splits(container,s);
 const methods=section(container,'五種 Feature Selection','選擇器及前處理均只 Fit 當折 Training；外折驗證 Target 不參與選擇。',s.feature_selection);
 table(methods,'TRAIN_CV · 方法介紹',['方法','方式','內部邊界'],[['SelectKBest','單變量 F 統計排名','只用 Training 折'],['RFE','迴歸係數遞迴消除','只用 Training 折'],['Forward SFS','前向逐步加欄位','Training 折內 TimeSeriesSplit'],['Lasso-based','標準化後依正則化係數排名','alpha 固定於研究設定'],['Permutation','擾動欄位後計算驗證誤差變化','僅 Training 折內時間驗證區間']]);
 const cv=section(container,'Training CV · K 與候選集合比較','50 配置／250 外折；五折 SD 為 population ddof=0。ALL 等價重用保留原研究來源，不冒稱新 Fit。',s.feature_selection);const slot=chartSlot(cv);state(slot,'LOADING','載入50組正式配置…');
 loader.load('charts/training_cv.json').then(data=>{if(slot.isConnected)renderCV(slot,data,'data');}).catch(e=>{if(slot.isConnected)state(slot,e.code==='EMPTY'?'EMPTY':'ERROR','Feature Selection 資料無法載入或沒有資料。');});
 renderTrainingAnalytics(container);
}
export function renderCV(slot,data,prefix){
 slot.replaceChildren();slot.dataset.state='SUCCESS';
 const controls=el('div',undefined,'filters');slot.append(controls);
 const set=select(controls,prefix+'-set','候選集合',[['all','全部（27／28）'],['deployable27','27 · deployable27'],['research28','28 · research28（含 lights，研究對照）']]);
 const method=select(controls,prefix+'-method','Feature Selection',[['all','全部方法'],...Object.entries(METHODS)]);
 const k=select(controls,prefix+'-k','K 配置',[['all','全部 K'],...['10','15','20','25','ALL'].map(v=>[v,v==='10'?'10 · RESEARCH_ONLY':v])]);
 const sort=select(controls,prefix+'-sort','表格排序',[['cv_rmse_mean','CV RMSE（低至高）'],['feature_jaccard_mean','Jaccard（低至高）'],['compute_seconds','耗時（低至高）']]);
 const out=chartSlot(slot);
 const update=()=>{
  out.replaceChildren();const rows=cvRows(data,set.value,method.value,k.value,sort.value);if(!rows.length){state(out,'EMPTY','沒有符合篩選的研究配置。');return;}
  const groups=[];for(const candidate of ['deployable27','research28'])for(const m of Object.keys(METHODS)){const subset=rows.filter(r=>r.candidate_set===candidate&&r.method===m).toSorted((a,b)=>a.effective_k-b.effective_k);if(subset.length)groups.push({name:`${candidate==='deployable27'?'27':'28'} · ${METHODS[m]}`,points:subset.map(r=>({x:r.effective_k,y:r.cv_rmse_mean,label:`K=${r.k}`}))});}
  // Separate candidate panels keep method colors consistent and avoid overlapping 10-series legends.
  for(const candidate of ['27','28']){const ss=groups.filter(g=>g.name.startsWith(candidate));if(ss.length)chart(chartSlot(out),{id:prefix+'-k-'+candidate,title:`${candidate} 候選 · K vs CV RMSE`,split:'TRAIN_CV',source:provenance(data),description:'每點為五折平均 RMSE；ALL 的實際 K 為候選數。不是 Test 成績。K10 僅研究對照；28 包含 lights。',xLabel:'選取特徵數 K',yLabel:'CV RMSE (Wh)',series:ss});}
  chart(chartSlot(out),{id:prefix+'-stability',title:'特徵穩定性與 CV RMSE',split:'TRAIN_CV',source:provenance(data),description:'Jaccard 為外折選取集合平均相似度（0–1）；穩定性不代表因果或泛化保證。',xLabel:'Jaccard (0–1)',yLabel:'CV RMSE (Wh)',kind:'scatter',series:[{name:'篩選後配置',points:rows.map(r=>({x:r.feature_jaccard_mean,y:r.cv_rmse_mean,label:`${r.candidate_set}/${r.method}/K${r.k}`}))}]});
  table(out,'TRAIN_CV · 正式配置與成本',['候選集合','方法','K','實際 K','RMSE Wh','SD Wh','Jaccard','秒','來源分類','使用限制'],rows.map(r=>[r.candidate_set,METHODS[r.method],r.k,r.effective_k,fmt(r.cv_rmse_mean),fmt(r.cv_rmse_std),fmt(r.feature_jaccard_mean),fmt(r.compute_seconds,3),r.classification,restriction(r)]),prefix+'-cv');
 };
 for(const control of [set,method,k,sort])control.addEventListener('change',update);update();
}
export function renderEvaluation(container,s,loader){
 renderDemo(container,loader);
 const roles=section(container,'Training CV ≠ Validation ≠ Test','正式 Test 僅一次評估，未據此重新選模。Calibration 不 Fit；Ridge／Lasso alpha 僅 Training 內 TimeSeriesSplit 調整。');for(const [k,v]of Object.entries(PURPOSES))roles.append(el('p',`${k.toUpperCase()} · ${v}`));
 const base=section(container,'Training CV · Mean／Full-Feature MLR','這是 Training CV 基準，不是正式 Test 分數。Full MLR 等價重用不代表重跑。',s.training_cv);table(base,'TRAIN_CV · 基準',['基準','候選集合','RMSE Wh','SD Wh','重用外折'],s.training_cv.values.baselines.map(r=>[r.kind,r.candidate_set,fmt(r.cv_rmse_mean),fmt(r.cv_rmse_std),r.reused_folds]),'baselines');
 const locked=section(container,'四組預鎖 MLR Validation','候選清單在首次讀取 Validation Target 前已鎖定；正式模型選 Forward SFS K15。Ridge／Lasso 僅研究比較，不取代 MLR。',s.validation);
 const mlr=validationRows(s,'mlr');table(locked,'VALIDATION · 四組 MLR',['方法','K','特徵數','MAE Wh','MSE Wh²','RMSE Wh','R²','Adjusted R²'],mlr.map(r=>[METHODS[r.method],r.k,r.p,fmt(r.mae),fmt(r.mse),fmt(r.rmse),fmt(r.r2,6),fmt(r.adjusted_r2,6)]),'mlr-validation');
 chart(chartSlot(locked),{id:'validation-mlr',title:'四組預鎖 MLR · Validation RMSE',split:'VALIDATION',source:provenance(s.validation),description:'依已鎖定方案比較，未因 Validation 或 Test 新增候選。方案名稱與完整數字見表格。',xLabel:'方案序號（依上表）',yLabel:'RMSE (Wh)',kind:'bar',zero:true,series:[{name:'LinearRegression',points:mlr.map((r,i)=>({x:i+1,y:r.rmse,label:`${METHODS[r.method]} K${r.k}`}))}]});
 const comparisons=section(container,'MLR／Ridge／Lasso · 12 組 Validation','alpha 僅 Training CV 調整。Ridge／Lasso 的 Adjusted R² 為傳統特徵數公式，不代表有效自由度。',s.validation);
 const modelSelect=select(comparisons,'validation-model','模型篩選',[['all','全部12組'],['mlr','MLR（4組）'],['ridge','Ridge（4組）'],['lasso','Lasso（4組）']]);const vslot=chartSlot(comparisons);
 const update=()=>{vslot.replaceChildren();table(vslot,'VALIDATION · 研究模型完整比較',['方法','K','模型','alpha','MAE Wh','MSE Wh²','RMSE Wh','R²','Adjusted R²'],validationRows(s,modelSelect.value).map(r=>[METHODS[r.method],r.k,r.model,r.alpha??'—',fmt(r.mae),fmt(r.mse),fmt(r.rmse),fmt(r.r2,6),fmt(r.adjusted_r2,6)]),'validation');};modelSelect.addEventListener('change',update);update();
 const test=section(container,'正式 Test · 凍結模型一次性評估','模型 mlr-1.0.0；所有數值直接取自保存的正式結果，不由抽樣圖表重算。',s.test);const m=s.test.values.metrics;
 table(test,'TEST · 完整資料指標',['n','p','MAE Wh','MSE Wh²','RMSE Wh','R²','Adjusted R²'],[[m.n,m.p,fmt(m.mae),fmt(m.mse),fmt(m.rmse),fmt(m.r2,6),fmt(m.adjusted_r2,6)]],'test-metrics');
 const pi=section(container,'95% Split Conformal · 校準與實際涵蓋','Lower=max(0,Prediction−q)，Upper=Prediction+q；點估計保留原值。時間相依不保證交換性，Observed Test Coverage 不保證未來95%。',s.calibration);const c=s.calibration.values,p=s.test.values.prediction_interval;
 table(pi,'CALIBRATION · 凍結參數',['n','k=ceil((n+1)×0.95)','q Wh','alpha'],[[c.n,c.k,String(c.q),c.alpha]],'calibration');
 table(pi,'TEST · PI Audit（完整資料）',['項目','值'],[['全部筆數',p.total_rows],['有效區間',p.valid_intervals],['無效區間',p.invalid_intervals],['涵蓋筆數',p.covered_count],['全體 Coverage %',fmt(p.coverage_all*100)],['有效 Coverage %',fmt(p.coverage_valid*100)],['平均寬度 Wh',fmt(p.average_width_valid)],['中位寬度 Wh',fmt(p.median_width_valid)],['低於下界',p.below_lower_count_valid],['高於上界',p.above_upper_count_valid],['負值點估計／Warning',p.negative_warning_count]],'pi');pi.append(el('p',`TEST 來源：${provenance(s.test)}`,'source-note'));
 chart(chartSlot(pi),{id:'coverage-audit',title:'Prediction Interval Coverage Audit',split:'TEST',source:provenance(s.test),description:'完整Test分母；有效區間外的誤差與無效區間分開呈現。精確Coverage與分母見上表。',xLabel:'1 涵蓋 / 2 低於 / 3 高於 / 4 無效',yLabel:'Count (筆)',kind:'bar',zero:true,series:[{name:'Saved Test audit',points:[p.covered_count,p.below_lower_count_valid,p.above_upper_count_valid,p.invalid_intervals].map((v,i)=>({x:i+1,y:v}))}]});
 const diag=section(container,'DW／BP · Test 殘差診斷','BP p-value 假設獨立觀測；序列相依限制其名目解讀。Q-Q 與 ACF 為描述性診斷，不保證常態或獨立。',s.diagnostics);const dg=s.diagnostics.values;
 table(diag,'TEST · 診斷',['DW','BP LM','BP LM p-value','BP F','BP F p-value'],[[fmt(dg.durbin_watson,6),fmt(dg.breusch_pagan.lm_statistic),String(dg.breusch_pagan.lm_pvalue),fmt(dg.breusch_pagan.f_statistic),String(dg.breusch_pagan.f_pvalue)]],'diagnostics');
 const vf=section(container,'VIF · 兩個資料區段分開比較','同一正式15特徵設計矩陣、凍結前處理並含截距。VIF 診斷不觸發重新選擇特徵；係數非因果效果。',s.training_vif);const vr=vifRows(s);table(vf,'TRAIN_VALIDATION 與 TEST · VIF',['特徵','TRAIN_VALIDATION VIF','TEST VIF'],vr.map(r=>[r[0],fmt(r[1]),fmt(r[2])]),'vif');vf.append(el('p',`TEST 來源：${provenance(s.diagnostics)}`,'source-note'));
 chart(chartSlot(vf),{id:'vif',title:'VIF 共線性比較',split:'TRAIN_VALIDATION / TEST',source:provenance(s.training_vif)+'；'+provenance(s.diagnostics),description:'依正式特徵順序（見表格）；共線性在兩區段可不同，不以Test診斷改模型。',xLabel:'正式特徵順序',yLabel:'VIF (無單位)',series:[{name:'TRAIN_VALIDATION',points:vr.map((r,i)=>({x:i+1,y:r[1],label:r[0]}))},{name:'TEST',points:vr.map((r,i)=>({x:i+1,y:r[2],label:r[0]}))}]});
 const cv=section(container,'Training CV · 全50配置','依候選集合、方法、K與成本篩選；K10／research28 為研究對照。');const cvslot=chartSlot(cv);state(cvslot,'LOADING','載入Training CV…');loader.load('charts/training_cv.json').then(d=>{if(cvslot.isConnected)renderCV(cvslot,d,'evaluation');}).catch(e=>{if(cvslot.isConnected)state(cvslot,e.code==='EMPTY'?'EMPTY':'ERROR','Training CV 無法載入。');});
 const lazy=section(container,'Test 圖表 · 按需載入','散點為非代表性抽樣；時間線為每小時彙整；直方圖涵蓋全部 Test。圖表不重新推論或計算正式指標。');
 const button=el('button','驗證圖表資料連線','secondary'),slot=chartSlot(lazy);lazy.insertBefore(button,slot);state(slot,'UNAVAILABLE','按下按鈕載入正式 Test 圖表，避免初始下載大型資料。');
 button.addEventListener('click',async()=>{button.disabled=true;state(slot,'LOADING','載入既有圖表資料…');try{const [,series,residuals]=await Promise.all(['charts/training_cv.json','charts/test_series.json','charts/test_residuals.json'].map(loader.load));if(!slot.isConnected)return;slot.replaceChildren();slot.append(el('p','三份圖表資料已通過版本、區段及 Hash 驗證；僅呈現既有研究產物。','muted'));renderTestCharts(slot,series,residuals);}catch(e){if(slot.isConnected)state(slot,e.code==='EMPTY'?'EMPTY':'ERROR','圖表載入失敗或沒有資料；不顯示替代圖表。');}finally{button.disabled=false;}});
}
export function renderTestCharts(host,series,residuals){
 const source=provenance(residuals),sample=`${residuals.scatter.length} 筆非代表性抽樣（均勻時間索引聯集全部負值估計與最大誤差）；不能重算官方指標。`;
 const rs=residuals.scatter;
 chart(chartSlot(host),{id:'actual-predicted',title:'Actual vs Predicted',split:'TEST',source,description:sample,xLabel:'Actual (Wh)',yLabel:'Predicted (Wh)',kind:'scatter',series:[{name:'真實 Test 樣本',points:rs.map(r=>({x:r.actual,y:r.prediction,label:r.timestamp}))}]});
 chart(chartSlot(host),{id:'residual-fitted',title:'Residual vs Predicted',split:'TEST',source,description:sample+' Residual=Actual−Prediction。',xLabel:'Predicted (Wh)',yLabel:'Residual (Wh)',kind:'scatter',series:[{name:'殘差樣本',points:rs.map(r=>({x:r.prediction,y:r.residual,label:r.timestamp}))}]});
 const h=residuals.histogram;
 chart(chartSlot(host),{id:'residual-distribution',title:'Residual Distribution',split:'TEST',source,description:`完整 ${residuals.source_rows} 筆 Test 直方圖，非散點抽樣；橫軸為每個bin中心。`,xLabel:'Residual bin center (Wh)',yLabel:'Count (筆)',kind:'bar',zero:true,series:[{name:'完整 Test 頻數',points:h.counts.map((n,i)=>({x:(h.edges[i]+h.edges[i+1])/2,y:n,label:`[${h.edges[i]},${h.edges[i+1]}]`}))}]});
 chart(chartSlot(host),{id:'qq',title:'Residual Q-Q',split:'TEST',source,description:'既有殘差分位數的101個展示點；右尾偏離可見，不宣稱常態性。',xLabel:'Normal theoretical quantile',yLabel:'Ordered residual (Wh)',kind:'scatter',series:[{name:'Q-Q 展示分位數',points:residuals.qq.theoretical.map((v,i)=>({x:v,y:residuals.qq.ordered_residual[i]}))}]});
 chart(chartSlot(host),{id:'acf',title:'Residual ACF',split:'TEST',source,description:'完整序列殘差的既有ACF；lag單位為10分鐘觀測步長。描述時間相依，不加入未驗證信賴帶。',xLabel:'Lag (10-min steps)',yLabel:'ACF (無單位)',kind:'bar',zero:true,series:[{name:'Test ACF',points:residuals.residual_acf.map((v,i)=>({x:i,y:v}))}]});
 const time=card('TEST · 每小時時間線與區間摘要',`${series.rows.length}組每小時彙整，首末組可能不足6筆。Actual mean/max、Prediction mean來自既有保存結果。上下端點均值不是小時平均值的新校準Prediction Interval。`);host.append(time);
 const filters=el('div',undefined,'filters');time.append(filters);const from=el('input'),to=el('input');from.type=to.type='date';from.id='series-from';to.id='series-to';from.min=to.min=series.rows[0].timestamp.slice(0,10);from.max=to.max=series.rows.at(-1).timestamp.slice(0,10);from.value=from.min;to.value=to.max;
 for(const [id,label,input]of [['series-from','開始日期',from],['series-to','結束日期',to]]){const w=el('div'),l=el('label',label);l.htmlFor=id;w.append(l,input);filters.append(w);}const area=chartSlot(time);
 const update=()=>{area.replaceChildren();if(from.value>to.value){state(area,'INVALID_INPUT','開始日期不可晚於結束日期。');return;}const rows=seriesRows(series,from.value,to.value);if(!rows.length){state(area,'EMPTY','指定日期沒有資料。');return;}
  const point=key=>rows.map((r,i)=>({x:i,y:r[key],label:r.timestamp})),tick=v=>rows[Math.max(0,Math.min(rows.length-1,Math.round(v)))].timestamp.slice(5,16);
  chart(chartSlot(area),{id:'time-series',title:'Actual vs Predicted · hourly',split:'TEST',source:provenance(series),description:'每小時平均與實際峰值；單位Wh代表觀測能耗值的平均／最大，並非重新加總的小時耗電。',xLabel:'Source-local time (MM-DD HH:mm)',yLabel:'Energy (Wh)',xTick:tick,series:[{name:'Actual mean',points:point('actual_mean')},{name:'Prediction mean',points:point('prediction_mean')},{name:'Actual max',points:point('actual_max')}]});
  chart(chartSlot(area),{id:'interval-summary',title:'Pointwise interval endpoint means',split:'TEST',source:provenance(series),description:'每小時組內逐筆區間端點的均值，僅供展示，不是小時平均值的新95%區間；q未改動。',xLabel:'Source-local time (MM-DD HH:mm)',yLabel:'Endpoint mean (Wh)',xTick:tick,series:[{name:'Lower endpoint mean',points:point('lower_mean')},{name:'Upper endpoint mean',points:point('upper_mean')}]});
 };from.addEventListener('change',update);to.addEventListener('change',update);update();
}
