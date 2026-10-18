export function el(tag, text, className) {
  const node = document.createElement(tag); if (text !== undefined) node.textContent = text;
  if (className) node.className = className; return node;
}
export function state(container, status, message) {
  container.replaceChildren(); container.dataset.state = status;
  const box = el('div',undefined,'state'); box.setAttribute('role',['ERROR','INVALID_INPUT'].includes(status) ? 'alert' : 'status');
  box.append(el('span',status,'state-label'),el('p',message)); container.append(box);
}
export function card(title, detail) {
  const c = el('article',undefined,'card'); c.append(el('h2',title),el('p',detail,'muted')); return c;
}
