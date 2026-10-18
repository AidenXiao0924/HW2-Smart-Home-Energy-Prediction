import {DATA, VERSION, FEATURES, Q, MANIFEST_HASH} from './config.js';
const verified = new WeakMap();
export const verifiedArtifact = (value,name) => verified.get(value)?.name === name;
export const verifiedHash = value => verified.get(value)?.sha256;
function freezeTree(value) { if(value && typeof value==='object'){Object.values(value).forEach(freezeTree);Object.freeze(value);}return value; }
export class DataError extends Error { constructor(code) { super(code); this.name = 'DataError'; this.code = code; } }
const assert = (ok, code='CONTRACT_MISMATCH') => { if (!ok) throw new DataError(code); };
export async function sha256(bytes) {
  return Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2,'0')).join('');
}
export function validate(name, value) {
  assert(value && typeof value === 'object' && !Array.isArray(value));
  if(name.startsWith('schemas/')){assert(value.$schema==='https://json-schema.org/draft/2020-12/schema' && typeof value.$id==='string');return value;}
  if (name === 'attribution.json') {
    for (const key of ['dataset_title','creator','citation','source_url','license','license_url','modifications']) assert(typeof value[key] === 'string' && value[key].length > 0);
    for (const key of ['source_url','license_url']) assert(new URL(value[key]).protocol === 'https:');
    return value;
  }
  assert(value.model_version === VERSION && value.schema_version === '1.0.0', 'VERSION_MISMATCH');
  if (name === 'manifest.json') {
    assert(value.files && value.inference?.model === 'model.json');
    for (const [path, entry] of Object.entries(value.files)) {
      assert(/^(?:schemas\/|charts\/)?[a-z_-]+(?:\.schema)?\.json$/.test(path), 'UNSAFE_PATH');
      assert(/^[a-f0-9]{64}$/.test(entry.sha256) && Number.isInteger(entry.bytes) && entry.bytes > 0);
    }
  } else if (name === 'model.json') {
    assert(value.model_type === 'LinearRegression' && value.prediction_task === 'contemporaneous_estimation');
    assert(JSON.stringify(value.feature_order) === JSON.stringify(FEATURES), 'FEATURE_ORDER_MISMATCH');
    assert(value.features?.length === 15 && value.features.every((f,i) => f.name === FEATURES[i] && f.type === 'number' && typeof f.unit === 'string'));
    for (const vector of [value.coefficients, value.preprocessing?.imputer?.statistics, value.preprocessing?.scaler?.mean, value.preprocessing?.scaler?.scale]) assert(Array.isArray(vector) && vector.length === 15 && vector.every(Number.isFinite));
    assert(Number.isFinite(value.intercept) && value.preprocessing.scaler.scale.every(n => n > 0));
    assert(value.prediction_interval?.q === Q && value.prediction_interval.alpha === 0.05, 'QUANTILE_MISMATCH');
    assert(value.prediction_interval.lower_rule === 'max(0,predicted-q)' && value.prediction_interval.upper_rule === 'predicted+q');
  } else if (name === 'research_summary.json') {
    const splits = {dataset:'ALL_RAW',splits:'ALL_RAW',training_cv:'TRAIN_CV',validation:'VALIDATION',final_model:'TRAIN_VALIDATION',calibration:'CALIBRATION',test:'TEST',training_vif:'TRAIN_VALIDATION'};
    for (const [key,split] of Object.entries(splits)) assert(value[key]?.split === split && value[key].values && /^[a-f0-9]{64}$/.test(value[key].source?.sha256));
    assert(Number.isInteger(value.dataset.values.rows) && value.dataset.values.rows > 0);
  } else if (name.startsWith('charts/')) {
    assert(value.split === (name.includes('training_cv') ? 'TRAIN_CV' : 'TEST'));
    const key = name.includes('training_cv') ? 'configurations' : name.includes('test_series') ? 'rows' : 'scatter';
    assert(Array.isArray(value[key]));
    if (!value[key].length) throw new DataError('EMPTY');
  }
  return value;
}
export function createLoader({base=DATA, fetcher=globalThis.fetch, manifestHash=MANIFEST_HASH}={}) {
  const cache = new Map();
  async function bytes(name) {
    const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetcher(new URL(name,base), {signal:controller.signal, cache:'no-cache',credentials:'omit'});
      assert(response.ok, 'HTTP_ERROR');
      return await response.arrayBuffer();
    } catch (e) { throw e instanceof DataError ? e : new DataError('NETWORK_ERROR'); }
    finally { clearTimeout(timer); }
  }
  function load(name) {
    if (!cache.has(name)) {
      const task = (async () => {
        assert(name === 'manifest.json' || /^(?:(?:charts|schemas)\/)?[a-z_-]+(?:\.schema)?\.json$/.test(name), 'UNSAFE_PATH');
        const manifest = name === 'manifest.json' ? null : await load('manifest.json');
        const expected = name === 'manifest.json' ? {sha256:manifestHash} : manifest.files[name];
        assert(expected, 'UNAVAILABLE');
        const raw = await bytes(name);
        assert((expected.bytes === undefined || raw.byteLength === expected.bytes) && await sha256(raw) === expected.sha256, 'HASH_MISMATCH');
        let value; try { value = JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(raw)); } catch { throw new DataError('INVALID_JSON'); }
        validate(name,value);freezeTree(value);verified.set(value,{name,sha256:expected.sha256});return value;
      })();
      cache.set(name,task); task.catch(() => cache.delete(name));
    }
    return cache.get(name);
  }
  return {load};
}
