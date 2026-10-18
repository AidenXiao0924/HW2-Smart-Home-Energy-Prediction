import {el,card,state} from '../core/ui.js';
export function renderFuture(container, page) {
  const sim=page==='simulator';const c=card(sim?'探索情境，等待推論功能開放':'帶入外部資料，等待批次功能開放',sim?'未來可調整正式特徵，取得同時段能源估計及區間。':'未來將支援 HTTPS CSV URL、本機 CSV 與結果下載。');
  const slot=el('div');state(slot,'UNAVAILABLE','功能開發中 · 本階段沒有模型推論或預測結果。');c.append(slot);
  const fieldset=el('fieldset');fieldset.disabled=true;const legend=el('legend',sim?'情境輸入（尚未開放）':'資料輸入（尚未開放）');fieldset.append(legend);
  const label=el('label',sim?'T1 · 溫度（°C）':'公開 HTTPS CSV URL');label.htmlFor='future-input';
  const input=el('input');input.id='future-input';input.type=sim?'number':'url';input.setAttribute('aria-describedby','future-help');
  const help=el('p','輸入、估計與下載將於後續階段實作。','muted');help.id='future-help';fieldset.append(label,input,help);c.append(fieldset);container.append(c);
}
