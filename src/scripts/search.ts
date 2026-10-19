import { compareCategories } from '../utils/post-order';

interface SearchDocument {
  url: string;
  meta: { title?: string };
  filters: Record<string, string[]>;
  excerpt: string;
}
interface SearchHit { data(): Promise<SearchDocument> }
interface SearchIndex {
  options(options: { excerptLength: number }): Promise<void>;
  filters(): Promise<Record<string, Record<string, number>>>;
  search(term: string | null, options: { filters: Record<string, string> }): Promise<{ results: SearchHit[] }>;
  destroy(): Promise<void>;
}

const query = document.querySelector<HTMLInputElement>('#search-query')!;
const topic = document.querySelector<HTMLSelectElement>('#search-topic')!;
const type = document.querySelector<HTMLSelectElement>('#search-type')!;
const output = document.querySelector<HTMLElement>('#search-output')!;
const status = document.querySelector<HTMLElement>('#search-status')!;
const list = document.querySelector<HTMLOListElement>('#search-results')!;
const start = document.querySelector<HTMLElement>('#search-start')!;
const empty = document.querySelector<HTMLElement>('#search-empty')!;
const error = document.querySelector<HTMLElement>('#search-error')!;
const clear = document.querySelector<HTMLButtonElement>('#search-clear')!;
const reset = document.querySelector<HTMLButtonElement>('#search-reset')!;
const recover = document.querySelector<HTMLButtonElement>('#search-recover')!;
const pagination = document.querySelector<HTMLElement>('#search-pagination')!;
const progress = document.querySelector<HTMLElement>('#search-progress')!;
const more = document.querySelector<HTMLButtonElement>('#search-more')!;
const pageSize = 5;

let indexPromise: Promise<SearchIndex> | undefined;
let loadAttempt = 0;
let hits: SearchHit[] = [];
let shown = 0;
let total = 0;
let revision = 0;
let timer: number | undefined;
let composing = false;

function readUrl() {
  const params = new URLSearchParams(location.search);
  query.value = (params.get('q') ?? '').slice(0, 120);
  for (const [select, key] of [[topic, 'topic'], [type, 'type']] as const) {
    const rawValue = params.get(key) ?? '';
    // 原“未分类”的三篇 Ascend C 笔记已合并，保留旧筛选链接的含义。
    const value = key === 'topic' && rawValue === '未分类' ? 'CANN / Ascend C' : rawValue;
    select.value = [...select.options].some((option) => option.value === value) ? value : '';
  }
}

function updateControls() {
  clear.hidden = !query.value;
  reset.hidden = !topic.value && !type.value;
}

function saveUrl() {
  const url = new URL(location.href);
  for (const [key, value] of [['q', query.value.trim()], ['topic', topic.value], ['type', type.value]]) {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  }
  history.replaceState(null, '', url);
}

function getIndex() {
  if (!indexPromise) {
    indexPromise = (async () => {
      // 浏览器会缓存模块加载失败；重试时使用新地址，允许网络恢复后重新获取。
      const path = `/pagefind/pagefind.js${loadAttempt++ ? `?retry=${loadAttempt}` : ''}`;
      const index: SearchIndex = await import(/* @vite-ignore */ path);
      try {
        await index.options({ excerptLength: 32 });
        const filters = await index.filters();
        for (const [select, key] of [[topic, '专题'], [type, '类型']] as const) {
          const names = Object.keys(filters[key] ?? {});
          names.sort(key === '专题' ? compareCategories : (a, b) => a.localeCompare(b, 'zh-CN'));
          select.replaceChildren(new Option(key === '专题' ? '全部专题' : '全部类型', ''));
          names.forEach((name) => select.add(new Option(name, name)));
          select.disabled = false;
        }
        total = Object.values(filters['类型'] ?? {}).reduce((sum, count) => sum + count, 0);
        // 输入可在索引加载期间编辑；只恢复筛选，不覆盖正在输入的关键词。
        const value = query.value;
        readUrl();
        query.value = value;
        return index;
      } catch (cause) {
        await index.destroy();
        throw cause;
      }
    })().catch((cause) => {
      indexPromise = undefined;
      throw cause;
    });
  }
  return indexPromise;
}

function createResult(data: SearchDocument) {
  const url = new URL(data.url, location.origin);
  if (url.origin !== location.origin || !/^\/(notes|research)\//.test(url.pathname)) {
    throw new Error('Unexpected search result URL');
  }
  const item = document.createElement('li');
  const article = document.createElement('article');
  const heading = document.createElement('h2');
  const link = document.createElement('a');
  link.href = url.pathname + url.search + url.hash;
  link.textContent = data.meta.title ?? '未命名文章';
  heading.append(link);
  const meta = document.createElement('p');
  meta.className = 'search-result-meta';
  meta.textContent = [...(data.filters['类型'] ?? []), ...(data.filters['专题'] ?? [])].join(' · ');
  const excerpt = document.createElement('p');
  excerpt.className = 'search-result-excerpt';
  // 摘要只保留索引提供的文字和匹配高亮，不插入其他 HTML。
  const template = document.createElement('template');
  template.innerHTML = data.excerpt;
  for (const node of template.content.childNodes) {
    if (node.nodeName === 'MARK') {
      const mark = document.createElement('mark');
      mark.textContent = node.textContent;
      excerpt.append(mark);
    } else {
      excerpt.append(document.createTextNode(node.textContent ?? ''));
    }
  }
  article.append(heading, meta, excerpt);
  item.append(article);
  return item;
}

function updatePagination() {
  pagination.hidden = hits.length === 0;
  progress.textContent = `已显示 ${shown} / ${hits.length} 篇`;
  more.hidden = shown >= hits.length;
  more.disabled = false;
  more.textContent = '加载更多结果';
  delete more.dataset.retry;
}

async function runSearch(ticket: number) {
  try {
    const index = await getIndex();
    if (ticket !== revision) return;
    const term = query.value.trim();
    const filters: Record<string, string> = {};
    if (topic.value) filters['专题'] = topic.value;
    if (type.value) filters['类型'] = type.value;
    updateControls();
    saveUrl();
    if (!term && !Object.keys(filters).length) {
      hits = [];
      shown = 0;
      list.replaceChildren();
      start.hidden = false;
      status.textContent = `可搜索 ${total} 篇内容`;
      updatePagination();
      return;
    }
    const result = await index.search(term || null, { filters });
    const documents = await Promise.all(result.results.slice(0, pageSize).map((hit) => hit.data()));
    if (ticket !== revision) return;
    const items = documents.map(createResult);
    hits = result.results;
    shown = items.length;
    list.replaceChildren(...items);
    status.textContent = term ? `「${term}」：${hits.length} 篇结果` : `找到 ${hits.length} 篇内容`;
    empty.hidden = hits.length > 0;
    if (!hits.length) {
      document.querySelector('#search-empty-message')!.textContent = Object.keys(filters).length
        ? '当前筛选范围内没有匹配内容，可以先清除筛选条件。'
        : `没有找到与「${term}」相关的内容。`;
      recover.textContent = Object.keys(filters).length ? '清除筛选，重新搜索' : '更换关键词';
    }
    updatePagination();
  } catch {
    if (ticket !== revision) return;
    hits = [];
    list.replaceChildren();
    pagination.hidden = true;
    start.hidden = empty.hidden = true;
    error.hidden = false;
    status.textContent = '搜索加载失败';
  } finally {
    if (ticket === revision) output.setAttribute('aria-busy', 'false');
  }
}

function requestSearch(delay = 0) {
  window.clearTimeout(timer);
  const ticket = ++revision;
  updateControls();
  start.hidden = empty.hidden = error.hidden = true;
  pagination.hidden = true;
  output.setAttribute('aria-busy', 'true');
  status.textContent = '正在搜索…';
  timer = window.setTimeout(() => { void runSearch(ticket); }, delay);
}

document.querySelector('#search-form')!.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!composing) requestSearch();
});
query.addEventListener('compositionstart', () => {
  composing = true;
  window.clearTimeout(timer);
  revision++;
});
query.addEventListener('compositionend', () => {
  composing = false;
  requestSearch(200);
});
query.addEventListener('input', () => {
  updateControls();
  if (!composing) requestSearch(200);
});
[topic, type].forEach((select) => select.addEventListener('change', () => requestSearch()));
clear.addEventListener('click', () => {
  query.value = '';
  query.focus();
  requestSearch();
});
function resetFilters() {
  topic.value = type.value = '';
  query.focus();
  requestSearch();
}
reset.addEventListener('click', resetFilters);
recover.addEventListener('click', () => {
  if (topic.value || type.value) resetFilters();
  else { query.focus(); query.select(); }
});
async function retrySearch() {
  window.clearTimeout(timer);
  const ticket = ++revision;
  const previous = indexPromise;
  indexPromise = undefined;
  output.setAttribute('aria-busy', 'true');
  status.textContent = '正在重新加载…';
  try { await (await previous)?.destroy(); } catch { /* 加载失败时也允许重新获取索引。 */ }
  if (ticket === revision) {
    query.focus();
    requestSearch();
  }
}
document.querySelector('#search-retry')!.addEventListener('click', () => { void retrySearch(); });
document.querySelectorAll<HTMLButtonElement>('[data-search-example]').forEach((button) => {
  button.addEventListener('click', () => {
    query.value = button.dataset.searchExample!;
    requestSearch();
  });
});
more.addEventListener('click', async () => {
  if (more.dataset.retry) { await retrySearch(); return; }
  const ticket = revision;
  const focusNext = document.activeElement === more;
  more.disabled = true;
  more.textContent = '正在加载…';
  try {
    const documents = await Promise.all(hits.slice(shown, shown + pageSize).map((hit) => hit.data()));
    if (ticket !== revision) return;
    const items = documents.map(createResult);
    list.append(...items);
    shown += items.length;
    updatePagination();
    if (focusNext) items[0]?.querySelector('a')?.focus({ preventScroll: true });
  } catch {
    if (ticket !== revision) return;
    status.textContent = '其余结果暂时无法加载，已保留当前结果。可以重新加载搜索。';
    more.disabled = false;
    more.textContent = '重新加载搜索';
    more.dataset.retry = 'true';
  }
});
window.addEventListener('popstate', () => { readUrl(); requestSearch(); });
readUrl();
requestSearch();
