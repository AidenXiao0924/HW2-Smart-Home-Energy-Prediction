export const VERSION = 'mlr-1.0.0';
export const FEATURES = Object.freeze(['T1','RH_1','T3','RH_3','T4','T5','RH_5','T7','RH_8','T9','RH_9','Windspeed','Visibility','Tdewpoint','Hour']);
export const Q = 185.3571214361994;
export const BASE = new URL('../../', import.meta.url);
export const DATA = new URL('public/mlr-1.0.0/', BASE);
// Pin the existing C3 manifest; publication copies its allowlist byte for byte.
export const MANIFEST_HASH = '73e53a369d0d00ac6deb01a94d92a50cd1a08bbd48b9ec7c0e67dcfc105a86ba';
export const PAGES = Object.freeze([
  ['overview','Overview','研究總覽','01'],
  ['data','Data & Features','資料與特徵','02'],
  ['evaluation','Model Evaluation','模型評估','03'],
  ['simulator','What-If Simulator','情境模擬','04'],
  ['csv','External CSV Prediction','外部資料估計','05'],
]);

export const DEMO_MANIFEST_HASH='eb66c1fa65f709711292439b2f518810aeadb7f7af6a453d5a9f2a0ec7ed3fd3';

export const TRAINING_MANIFEST_HASH='533074b373bb73541096e7ae51a6f724c268285f1245d6142fd4bdd3733e3035';
