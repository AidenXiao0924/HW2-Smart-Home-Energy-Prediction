// Presentation transforms only. Official scores are never recomputed from chart samples.
export const METHODS={select_k_best:'SelectKBest',rfe:'RFE',forward_sfs:'Forward SFS',lasso:'Lasso-based',permutation:'Permutation'};
export const FEATURE_MEANINGS={T1:'廚房溫度',RH_1:'廚房相對濕度',T3:'洗衣間溫度',RH_3:'洗衣間相對濕度',T4:'辦公室溫度',T5:'浴室溫度',RH_5:'浴室相對濕度',T7:'熨衣室溫度',RH_8:'青少年房相對濕度',T9:'父母房溫度',RH_9:'父母房相對濕度',Windspeed:'風速',Visibility:'能見度',Tdewpoint:'露點溫度',Hour:'來源當地時間之小時（0–23）'};
export const PURPOSES={train:'Fit／Feature Selection／5-fold TimeSeriesSplit',validation:'已預鎖候選之有限次比較',calibration:'僅校準 Conformal q，不 Fit',test:'凍結模型一次性正式評估'};
export const fmt=(v,d=4)=>typeof v==='number'&&Number.isFinite(v)?v.toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d}):String(v??'—');
export const restriction=r=>String(r.k)==='10'?'RESEARCH_ONLY':r.candidate_set==='research28'?'RESEARCH_CONTROL_LIGHTS':'DEPLOYMENT_ELIGIBLE_NOT_SELECTED';
export function kpis(s){const m=s.test.values.metrics,p=s.test.values.prediction_interval;return [['rmse','Test RMSE',m.rmse,'Wh'],['mae','Test MAE',m.mae,'Wh'],['r2','Test R²',m.r2,'無單位'],['coverage_all','Observed PI Coverage',p.coverage_all*100,'%']];}
export function splitRows(s){return Object.entries(s.splits.values.partitions).map(([name,p])=>[name.toUpperCase(),p.rows,100*p.rows/s.dataset.values.rows,p.start,p.end,PURPOSES[name]]);}
export function cvRows(data,set='all',method='all',k='all',sort='cv_rmse_mean'){
 return data.configurations.filter(r=>(set==='all'||r.candidate_set===set)&&(method==='all'||r.method===method)&&(k==='all'||String(r.k)===k)).toSorted((a,b)=>a[sort]-b[sort]||a.experiment_id.localeCompare(b.experiment_id));
}
export function validationRows(s,model='all'){return s.validation.values.comparisons.filter(r=>model==='all'||r.model===model);}
export function featureRows(model){return model.features.map((f,i)=>[i+1,f.name,FEATURE_MEANINGS[f.name],f.unit]);}
export function vifRows(s){return s.final_model.values.feature_order.map(f=>[f,s.training_vif.values.vif.find(v=>v.feature===f).vif,s.diagnostics.values.vif.find(v=>v.feature===f).vif]);}
export function seriesRows(data,from='',to=''){return data.rows.filter(r=>(!from||r.timestamp.slice(0,10)>=from)&&(!to||r.timestamp.slice(0,10)<=to));}
