/* Offline replay only. No provider client, credentials, telemetry or remote API. */
"use strict";
const $ = selector => document.querySelector(selector);
const MODAL_NOTICE = "此网页仅回放已有问题。真实问答请使用本地部署版；这里不会调用模型 API。";
let catalog, busy = false, storageAvailable = true, currentView = "chat";
let state = { conversations: [{ id: "initial", records: [] }], active: "initial" };
const ICONS = /* @shared-icons */ {};
function icon(name) { const node = document.createElementNS("http://www.w3.org/2000/svg", "svg"); node.setAttribute("viewBox", "0 0 24 24"); node.setAttribute("class", "ui-icon"); node.setAttribute("aria-hidden", "true"); node.innerHTML = ICONS[name]; return node; }
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const STORAGE = "jev-offline-demo-v1";

function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function message(text) { $("#notice").textContent = text; }
function markdown(container, source) {
  // Extract TeX before Markdown consumes underscores/backslashes. Render math
  // with trust=false and strict limits; no remote images or embedded HTML apps.
  const formulas = [];
  const prepared = source.replace(/(```[\s\S]*?```|`[^`\n]*`)|(\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+\$)/g, (whole, code, math) => {
    if (code) return code;
    const index = formulas.push({ text: math.startsWith("$$") ? math.slice(2, -2) : math.slice(1, -1), display: math.startsWith("$$") }) - 1;
    return `JEVMATHPLACEHOLDER${index}END`;
  });
  container.innerHTML = DOMPurify.sanitize(marked.parse(prepared, { breaks: false }), {
    FORBID_TAGS: ["img", "iframe", "script", "style", "form", "input", "video", "audio", "object", "embed"],
    FORBID_ATTR: ["style", "srcset", "id", "name"]
  });
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    if (node.parentElement.closest("code,pre")) continue;
    const matches = [...node.textContent.matchAll(/JEVMATHPLACEHOLDER(\d+)END/g)];
    if (!matches.length) continue;
    const fragment = document.createDocumentFragment(); let last = 0;
    for (const match of matches) {
      fragment.append(document.createTextNode(node.textContent.slice(last, match.index)));
      const span = el("span"), formula = formulas[Number(match[1])];
      if (formula) katex.render(formula.text, span, { displayMode: formula.display, throwOnError: false, trust: false, maxExpand: 1000, maxSize: 20 });
      fragment.append(span); last = match.index + match[0].length;
    }
    fragment.append(document.createTextNode(node.textContent.slice(last))); node.replaceWith(fragment);
  }
  container.querySelectorAll("a").forEach(link => { link.rel = "noopener noreferrer"; link.target = "_blank"; });
}
function persist() {
  try { localStorage.setItem(STORAGE, JSON.stringify(state)); }
  catch { storageAvailable = false; }
}
function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE) || "null");
    const valid = new Set(catalog.recordings.map(r => r.id));
    if (saved && Array.isArray(saved.conversations) && saved.conversations.length <= 30) {
      state.conversations = saved.conversations.filter(c => typeof c.id === "string" && Array.isArray(c.records))
        .map(c => ({ id: c.id.slice(0, 80), records: c.records.filter(id => valid.has(id)).slice(-50) }));
      if (!state.conversations.length) state.conversations = [{ id: "initial", records: [] }];
      state.active = state.conversations.some(c => c.id === saved.active) ? saved.active : state.conversations[0].id;
    }
  } catch { storageAvailable = false; }
}
function active() { return state.conversations.find(c => c.id === state.active); }
function selectedRecord() {
  return catalog.recordings.find(r => r.question === $("#question").value && r.mode === $("#mode").value);
}
function setModes() {
  const previous = $("#mode").value, available = Object.keys(catalog.modes).filter(m => catalog.recordings.some(r => r.question === $("#question").value && r.mode === m));
  $("#mode").replaceChildren(...available.map(mode => new Option(catalog.modes[mode], mode)));
  $("#mode").value = available.includes(previous) ? previous : available[available.length - 1];
  const chosen = selectedRecord();
  $("#current-example-title").textContent = chosen?.title || "示例解答";
  document.querySelectorAll(".example-card").forEach(button => {
    const selected = button.dataset.question === $("#question").value;
    button.classList.toggle("selected", selected);

  });
}
function renderExamples() {
  const seen = new Set(), host = $("#example-cards"); host.replaceChildren();
  for (const record of catalog.recordings) {
    if (seen.has(record.question)) continue; seen.add(record.question);
    const button = el("button", "example-card"); button.dataset.question = record.question;
    button.append(el("b", "", record.title), el("small", "", record.category), el("p", "example-question", record.question), el("span", "choice-mark", "查看解答与相关 Wiki"));
    button.querySelector(".choice-mark").append(icon("arrow"));
    button.addEventListener("click", () => { if (busy) return; $("#question").value = record.question; setModes(); send(selectedRecord()); }); host.append(button);
  }
}
function switchView(view) {
  if (!["chat", "examples", "answer", "wiki", "library", "archive"].includes(view)) view = "chat";
  currentView = view;
  document.querySelectorAll(".view").forEach(section => section.hidden = section.id !== view + "-view");
  document.querySelectorAll("nav button").forEach(button => button.classList.toggle("active", button.dataset.view === (view === "answer" ? "examples" : view)));
  $("#config").hidden = view !== "answer";
  $("#page-title").textContent = { chat: "知识概览", examples: "示例问题", answer: "示例解答", wiki: "知识探索", library: "资料库", archive: "实验档案" }[view];
  document.querySelector("nav button.active")?.setAttribute("aria-current", "page");
  document.querySelectorAll("nav button:not(.active)").forEach(b => b.removeAttribute("aria-current"));
  history.replaceState(null, "", "#" + view);
  window.scrollTo({ top: 0, behavior: "instant" });
  closeSidebar();
  if (!reducedMotion()) document.querySelector("#" + view + "-view").animate([{opacity:.55},{opacity:1}], {duration:180,easing:"ease-out"});
  message(storageAvailable ? "" : "浏览器未允许保存记录；当前会话仍可正常使用。");
}
function closeSidebar() {
 const wasOpen = document.body.classList.contains("sidebar-open");
 $("main").inert = false; document.body.classList.remove("sidebar-open"); $("#menu").setAttribute("aria-expanded", "false"); $("#menu").setAttribute("aria-label", "展开侧栏"); $("#menu").replaceChildren(icon("menu"));
 if (wasOpen) $("#menu").focus();
}
function panel(label) {
  const details = el("details", "record-panel"), summary = el("summary", "", label), body = el("div", "panel-body");
  details.append(summary, body); return { details, body };
}
function renderEvidence(record, host) {
  if (!record.evidence.length) return;
  const { details, body } = panel(`引用与${record.scope === "完整词条" ? "完整词条" : "原文"} · ${record.evidence.length} 条`);
  const select = el("select"), label = el("label", "", "选择参考资料"), location = el("p", "source-location"), text = el("div", "source-text");
  select.id = "evidence-" + crypto.randomUUID(); label.htmlFor = select.id;
  select.append(...record.evidence.map((item, i) => new Option(`${i + 1}. ${item.title}`, String(i))));
  function show() {
    const source = record.evidence[Number(select.value)];
    const locator = source.locator;
    location.textContent = typeof locator === "string" ? locator : source.page != null ? `第 ${source.page} 页 · ${source.section}` : source.section;
    // Source text stays literal, matching the local demo's faithful evidence view.
    text.textContent = source.text;
  }
  select.addEventListener("change", show); body.append(label, select, location, text); show(); host.append(details);
}
function renderRun(record, host) {
  const { details, body } = panel("运行详情与引用核对"), grid = el("div", "details-grid");
  for (const [title, value] of [["本次 API 请求", "0"], ["历史请求", record.historical_calls ?? "未记录"], ["历史状态", record.status === "answered" ? "已回答" : "生成失败"]]) {
    const metric = el("div", "", title); metric.append(el("b", "", String(value))); grid.append(metric);
  }
  body.append(grid, el("p", "hint", "当前为历史回放；没有执行新检索、生成或核验。历史调用与模型判断不代表本次运行或人工验收。"));
  for (const event of record.timeline) body.append(el("div", "trace", [event.node, event.selected_action, event.error].filter(Boolean).join(" · ")));
  body.append(el("p", "provenance", "历史记录 SHA-256：" + (record.source_sha256 || "未记录")));
  host.append(details);
  if (record.navigation.length) {
    const graph = panel("本次 Wiki 与图谱检索路径");
    for (const step of record.navigation) {
      graph.body.append(el("p", "", "Wiki：" + (step.wiki_pages.length ? step.wiki_pages : step.seeds).join("、")));
      graph.body.append(el("p", "", "关联主题：" + (step.graph_neighbors.join("、") || "无额外关联")));
    }
    host.append(graph.details);
  }
}
function exportRecord(record) {
  const sources = record.evidence.map((e, i) => `### ${i + 1}. ${e.title}\n\n${e.text}`).join("\n\n");
  const text = `# ${record.question}\n\n${record.answer}\n\n## 参考资料\n\n${sources}\n\n---\n离线历史回放，本次API 0次。\n历史记录SHA-256: ${record.source_sha256}\n`;
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" })), link = el("a");
  link.href = url; link.download = "jev-research.md"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function renderMessages() {
  const host = $("#messages"); host.replaceChildren();

  for (const id of active().records) {
    const record = catalog.recordings.find(r => r.id === id); if (!record) continue;
    const item = el("article", "message"), question = el("div", "question-bubble", record.question), answer = el("div", "answer-bubble markdown");
    markdown(answer, record.answer); item.append(question, answer);
    if (record.status !== "answered") item.append(el("p", "diagnostic", "本条历史运行未成功交付答案。原因和资料可在下方详情中查阅。"));
    const actions = el("div", "actions"), copy = el("button", "", "复制回复"), download = el("button", "", "导出研究记录");
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(answer.innerText); copy.textContent = "已复制"; }
      catch { message("当前浏览器不允许自动复制，请选择正文手动复制。"); }
    });
    download.addEventListener("click", () => exportRecord(record)); actions.append(copy, download); item.append(actions);
    renderEvidence(record, item); renderRun(record, item);
    const related = el("section", "related-wiki"); related.append(el("h3", "", "继续探索相关 Wiki"));
    for (const id of record.wiki_ids) {
      const article = catalog.articles.find(a => a.id === id); if (!article) continue;
      const link = el("button", "wiki-chip", article.title);
      link.append(icon("arrow")); link.addEventListener("click", () => openWiki(id)); related.append(link);
    }
    item.append(related); host.append(item);
  }
  renderHistory();
}
function renderHistory() {
  const host = $("#history"); host.replaceChildren();
  for (const conversation of [...state.conversations].reverse()) {
    if (!conversation.records.length) continue;
    const first = catalog.recordings.find(r => r.id === conversation.records[0]), button = el("button", conversation.id === state.active ? "selected" : "", first?.question || "研究对话");
    button.title = first?.question || "研究对话";
    button.addEventListener("click", () => { if (busy) return; state.active = conversation.id; persist(); renderMessages(); switchView("answer"); }); host.append(button);
  }
}
function setBusy(value) {
  busy = value;
  for (const selector of ["#send-example", "#browse-examples", "#new-chat", "#clear-history", "#question", "#mode"]) $(selector).disabled = value;
}
async function send(record) {
  if (busy) return;
  if (!record) { message("此问题暂无对应的历史记录。请选择示例或输入问题完整原文；未调用 API。"); return; }
  switchView("answer"); setBusy(true);
  const identifier = crypto.randomUUID(); state.conversations.push({ id: identifier, records: [] });
  state.conversations = state.conversations.slice(-30); state.active = identifier; $("#messages").replaceChildren();
  const waiting = el("div", "processing"), dots = el("span", "dots"); dots.append(el("i"), el("i"), el("i"));
  waiting.setAttribute("role", "status"); waiting.append(dots, el("span", "", "正在载入历史记录…")); $("#messages").append(waiting);
  await new Promise(resolve => requestAnimationFrame(resolve));
  active().records.push(record.id); active().records = active().records.slice(-50); persist(); renderMessages(); setBusy(false);
  $("#messages").lastElementChild?.scrollIntoView({ behavior: reducedMotion() ? "instant" : "smooth", block: "start" });
}
function showArticle(article) {
  $("#article-list").querySelectorAll("button").forEach(b => b.classList.toggle("selected", b.dataset.article === article.id));
  markdown($("#article-body"), article.text);
  const graph = $("#wiki-graph"); graph.replaceChildren(el("h2", "", "关联图谱"), el("p", "hint", "连线表示词条间的阅读关联；点击节点继续探索。"));
  const map = el("div", "graph-map"), hub = el("div", "graph-hub", article.title), nodes = el("div", "graph-neighbors");
  map.append(hub, nodes);
  for (const id of article.related) {
    const neighbor = catalog.articles.find(a => a.id === id); if (!neighbor) continue;
    const button = el("button", "graph-node", neighbor.title);
    button.append(icon("arrow")); button.addEventListener("click", () => openWiki(id)); nodes.append(button);
  }
  if (!nodes.childElementCount) nodes.append(el("p", "hint", "暂无关联主题"));
  graph.append(map);
}
function openWiki(id) {
  $("#wiki-search").value = ""; renderWiki(id); switchView("wiki");
  $("#wiki-graph").scrollIntoView({ behavior: reducedMotion() ? "instant" : "smooth", block: "start" });
}
function renderWiki(preferredId) {
  const query = $("#wiki-search").value.trim().toLowerCase(), list = $("#article-list"); list.replaceChildren();
  const terms = query.split(/\s+/).filter(Boolean);
  const matches = catalog.articles.map(article => {
    const title = [article.id, article.title, ...article.aliases].join(" ").toLowerCase(), body = article.text.toLowerCase();
    return { article, score: terms.every(t => title.includes(t) || body.includes(t)) ? terms.reduce((score, t) => score + (title.includes(t) ? 10 : 1), 0) : -1 };
  }).filter(row => row.score >= 0).sort((a, b) => b.score - a.score).map(row => row.article);
  for (const article of matches) {
    const button = el("button", "", article.title); button.dataset.article = article.id;
    button.addEventListener("click", () => showArticle(article)); list.append(button);
  }
  const chosen = matches.find(a => a.id === preferredId) || matches[0];
  if (chosen) showArticle(chosen);
  else { $("#article-body").textContent = "暂无匹配词条，请尝试其他关键词。"; $("#wiki-graph").replaceChildren(); }
}

function renderLibrary() {
  const sources = new Map();
  for (const record of catalog.recordings) for (const source of record.evidence) {
    if (source.kind === "wiki_article") continue;
    if (!sources.has(source.title)) sources.set(source.title, new Set());
    sources.get(source.title).add(source.section);
  }
  for (const [title, sections] of sources) {
    const card = el("article", "source-card"); card.append(el("h2", "", title), el("p", "", [...sections].filter(Boolean).join(" · "))); $("#source-list").append(card);
  }
}
function renderArchive() {
  for (const record of catalog.recordings) {
    const card = el("article", "archive-card"), content = el("div"), button = el("button", "", "查看回放");
    content.append(el("h2", "", record.question), el("p", "", `${catalog.modes[record.mode]} · ${record.status === "answered" ? "已回答" : "生成失败"} · 历史记录`));
    button.addEventListener("click", () => {
      if (busy) return; $("#question").value = record.question; setModes(); $("#mode").value = record.mode; send(record);
    }); card.append(content, button); $("#archive-list").append(card);
  }
}
async function init() {
  try {
    const response = await fetch("./catalog.json", { cache: "no-cache", credentials: "same-origin" });
    if (!response.ok) throw new Error("catalog unavailable");
    catalog = await response.json();
    if (catalog.schema !== 1 || !Array.isArray(catalog.recordings) || !catalog.recordings.length) throw new Error("invalid catalog");
    $("#knowledge-count").textContent = `${catalog.articles.length} 篇完整词条 · 关键词搜索 · 关联主题导航`;
    const questions = [...new Set(catalog.recordings.map(r => r.question))];
    $("#question").value = questions[0]; renderExamples(); setModes();
    restore(); renderMessages(); renderWiki(); renderLibrary(); renderArchive(); switchView(location.hash.slice(1) || "chat");
    $(".brand").addEventListener("click", event => { event.preventDefault(); switchView("chat"); });
    $("#send-example").addEventListener("click", () => send(selectedRecord()));
    $("#browse-examples").addEventListener("click", () => switchView("examples"));
    $("#back-examples").addEventListener("click", () => switchView("examples"));
    $("#knowledge-search-form").addEventListener("submit", event => {
      event.preventDefault(); $("#wiki-search").value = $("#knowledge-query").value;
      renderWiki(); switchView("wiki");
    });
    $("#new-chat").addEventListener("click", () => switchView("examples"));
    $("#clear-history").addEventListener("click", () => {
      if (!confirm("清空这个浏览器中的演示记录？不会影响本地项目的真实问答记录。")) return;
      state = { conversations: [{ id: crypto.randomUUID(), records: [] }], active: "" }; state.active = state.conversations[0].id; persist(); renderMessages();
    });
    document.querySelectorAll("nav button").forEach(button => button.addEventListener("click", () => switchView(button.dataset.view)));
    $("#wiki-search").addEventListener("input", renderWiki);
    document.querySelectorAll("[data-topic]").forEach(button => button.addEventListener("click", () => {
      const question = questions.find(q => q.includes(button.dataset.topic)); if (question) { $("#question").value = question; setModes(); send(selectedRecord()); }
    }));
    document.querySelectorAll("[data-wiki]").forEach(button => button.addEventListener("click", () => openWiki(button.dataset.wiki)));
    $("#menu").addEventListener("click", () => { if (document.body.classList.contains("sidebar-open")) closeSidebar(); else { document.body.classList.add("sidebar-open"); $("main").inert = true; $("#menu").setAttribute("aria-expanded", "true"); $("#menu").setAttribute("aria-label", "关闭侧栏"); $("#menu").replaceChildren(icon("close")); $("#sidebar nav button.active")?.focus(); } });
    $("#sidebar-dismiss").addEventListener("click", closeSidebar);
    matchMedia("(min-width: 761px)").addEventListener("change", event => { if (event.matches) closeSidebar(); });
    document.addEventListener("keydown", event => {
      if (!document.body.classList.contains("sidebar-open")) return;
      if (event.key === "Escape") { closeSidebar(); $("#menu").focus(); }
      if (event.key === "Tab") {
        const controls = [$("#menu"), ...$("#sidebar").querySelectorAll("a,button,select,input")].filter(n => !n.disabled && n.getClientRects().length);
        const first = controls[0], last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    });
  } catch {
    message("演示资料暂时无法载入，请刷新重试。若在本机预览，请通过HTTP服务打开，不能直接双击HTML文件。");
    $("#send-example").disabled = true;
  }
}
document.addEventListener("DOMContentLoaded", init);
