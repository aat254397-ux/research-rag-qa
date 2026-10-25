"use strict";

const ids = [
  "refresh-status", "document-count", "page-count", "chunk-count", "document-list",
  "storage-status", "document-paths", "load-documents", "build-embedding", "retriever",
  "rerank", "candidate-k", "top-k", "min-score", "index-badge", "retrieval-stage",
  "rerank-stage", "question-form", "question", "search-button", "ask-button",
  "operation-status", "error-message", "result-panel", "result-meta", "empty-state",
  "answer-section", "answered-question", "answer-kind", "answer-text",
  "references-section", "references-title", "references-count", "references-list",
];
const ui = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const state = { busy: false, index: null };

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function setMessage(message) {
  ui["operation-status"].textContent = message;
}

function showError(message = "") {
  ui["error-message"].textContent = message;
  ui["error-message"].hidden = !message;
}

function syncControls() {
  document.querySelectorAll("button, input, textarea, select").forEach((element) => {
    element.disabled = state.busy;
  });
  const semantic = ui.retriever.value === "embedding";
  const ready = state.index?.embedding?.status === "ready";
  ui.rerank.disabled = state.busy || !semantic;
  ui["candidate-k"].disabled = state.busy || !semantic || !ui.rerank.checked;
  ui["min-score"].min = semantic ? "-1" : "0";
  ui["search-button"].disabled = state.busy || !state.index || (semantic && !ready);
  ui["ask-button"].disabled = ui["search-button"].disabled;
  ui["build-embedding"].disabled = state.busy || !state.index || ready;
  ui["build-embedding"].textContent = ready ? "✓ E5 语义索引已就绪" : "建立 E5 语义索引";
  ui["retrieval-stage"].textContent = semantic ? "02 E5 召回" : "02 TF-IDF 检索";
  ui["rerank-stage"].textContent = semantic && ui.rerank.checked ? "03 重排筛选" : "03 跳过重排";
}

async function requestJSON(path, method = "GET", body) {
  let response;
  try {
    response = await fetch(path, {
      method,
      cache: "no-store",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error("无法连接本地服务，请确认 Uvicorn 仍在运行，然后点击“刷新”。");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    let detail = data?.detail;
    if (Array.isArray(detail)) {
      detail = detail.map((item) => `${(item.loc || []).join(".")}：${item.msg}`).join("\n");
    }
    throw new Error(typeof detail === "string" ? detail : `请求失败（HTTP ${response.status}）`);
  }
  if (data === null) throw new Error("接口没有返回 JSON，请确认当前服务已更新到 0.9.0。");
  return data;
}

async function runAction(message, action) {
  if (state.busy) return;
  state.busy = true;
  showError();
  setMessage(message);
  ui["operation-status"].dataset.busy = "true";
  ui["result-panel"].setAttribute("aria-busy", "true");
  syncControls();
  try {
    await action();
  } catch (error) {
    showError(error.message || "操作出现错误，请查看服务终端。");
    setMessage("请根据错误提示调整设置后重试。");
  } finally {
    state.busy = false;
    ui["operation-status"].dataset.busy = "false";
    ui["result-panel"].setAttribute("aria-busy", "false");
    syncControls();
  }
}

async function refreshIndex() {
  const data = await requestJSON("/index/status");
  state.index = data;
  ui["document-count"].textContent = data.document_count;
  ui["page-count"].textContent = data.indexed_pdf_pages;
  ui["chunk-count"].textContent = data.chunk_count;
  ui["document-list"].replaceChildren();
  for (const path of data.documents || []) {
    const item = node("li", "", path.split(/[\\/]/).pop() || path);
    item.title = path;
    ui["document-list"].append(item);
  }
  if (!data.documents?.length) ui["document-list"].append(node("li", "muted", "尚无文献"));
  const origin = { disk: "已从硬盘恢复", loaded_documents: "已加载文献", sample_documents: "当前为示例文档" };
  const storage = data.index_storage === "memory+disk" ? "内存检索 + 硬盘保存" : "仅内存索引";
  ui["storage-status"].textContent = `${origin[data.index_origin] || "索引已读取"} · ${storage}`;
  const labels = { ready: "语义索引就绪", not_built: "语义索引待建立", stale: "资料已变更，需建索引", invalid: "语义索引需重建" };
  const status = data.embedding?.status;
  ui["index-badge"].textContent = labels[status] || "本地服务已连接";
  ui["index-badge"].dataset.tone = status === "ready" ? "good" : "warning";
  syncControls();
  return data;
}

function integerValue(id, label, min, max) {
  const value = Number(ui[id].value);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${label}必须是 ${min}～${max} 之间的整数。`);
  }
  return value;
}

function questionPayload(generate) {
  const question = ui.question.value.trim();
  if (!question || question.length > 2000) throw new Error("请输入 1～2000 个字符的问题。");
  const semantic = ui.retriever.value === "embedding";
  const rerank = semantic && ui.rerank.checked;
  const payload = {
    question,
    retriever: ui.retriever.value,
    rerank,
    top_k: integerValue("top-k", "最终片段数", 1, 10),
  };
  if (rerank) {
    payload.candidate_k = integerValue("candidate-k", "候选数量", 1, 50);
    if (payload.candidate_k < payload.top_k) throw new Error("候选数量不能小于最终片段数。");
  }
  if (ui["min-score"].value.trim() !== "") {
    const score = Number(ui["min-score"].value);
    const min = semantic ? -1 : 0;
    if (!Number.isFinite(score) || score < min || score > 1) {
      throw new Error(`最低检索分数必须在 ${min}～1 之间，或留空。`);
    }
    payload.min_score = score;
  }
  if (generate) payload.mode = "deepseek";
  return payload;
}

function scoreText(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
}

function renderAnswer(text) {
  // 只渲染常见的 **加粗**；所有内容用文本节点插入，不把模型输出当 HTML 执行。
  ui["answer-text"].replaceChildren();
  for (const part of String(text).split(/(\*\*[^*\n]+\*\*)/g)) {
    ui["answer-text"].append(part.startsWith("**") && part.endsWith("**")
      ? node("strong", "", part.slice(2, -2)) : document.createTextNode(part));
  }
}

function renderReferences(references, retriever) {
  ui["references-list"].replaceChildren();
  ui["references-count"].textContent = `${references.length} 个片段`;
  const scoreLabel = retriever === "embedding" ? "E5" : "TF-IDF";
  references.forEach((reference, index) => {
    const details = node("details", "reference");
    const summary = node("summary");
    summary.append(node("span", "reference-number", String(reference.rank ?? index + 1)));
    const info = node("span", "reference-info");
    info.append(node("span", "reference-name", reference.source));
    const meta = [reference.page == null ? "文本文件" : `PDF 第 ${reference.page} 页`, `切片 ${reference.chunk_id}`];
    if (reference.retrieval_rank != null) meta.push(`原排名 ${reference.retrieval_rank}`);
    meta.push(`${scoreLabel} ${scoreText(reference.score)}`);
    if (reference.rerank_score != null) meta.push(`重排 ${scoreText(reference.rerank_score)}`);
    info.append(node("span", "reference-meta", meta.join(" · ")));
    summary.append(info, node("span", "reference-toggle", "展开原文 ↙"));
    const body = node("div", "reference-body");
    body.append(node("div", "reference-text", reference.text));
    body.append(node("p", "reference-path", `${reference.path} · 字符位置 ${reference.start}–${reference.end}`));
    details.append(summary, body);
    ui["references-list"].append(details);
  });
}

function clearResult() {
  ui["empty-state"].hidden = false;
  ui["answer-section"].hidden = true;
  ui["references-section"].hidden = true;
  ui["answer-text"].replaceChildren();
  ui["references-list"].replaceChildren();
  ui["result-meta"].textContent = "";
}

async function submitQuestion(generate) {
  await runAction(generate ? "正在检索、重排并生成答案；首次请求可能需要加载本地模型…" : "正在本地检索文献，按设置进行重排…", async () => {
    const payload = questionPayload(generate);
    clearResult();
    const started = performance.now();
    const data = await requestJSON(generate ? "/ask" : "/search", "POST", payload);
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    ui["empty-state"].hidden = true;
    ui["answer-section"].hidden = false;
    ui["answered-question"].textContent = `问题：${data.question}`;
    ui["answer-kind"].textContent = generate ? "DeepSeek 回答" : "仅检索 · 未调用生成模型";
    const references = data.references || [];
    renderAnswer(generate ? data.answer : (references.length ? "已找到以下参考片段，展开原文查看检索依据。" : "没有片段通过当前检索条件，可调整问题或最低分数后重试。"));
    ui["references-title"].textContent = generate ? "提供给模型的参考片段" : "检索到的参考片段";
    ui["references-section"].hidden = false;
    renderReferences(references, data.retriever);
    const method = data.retriever === "embedding" ? "E5" : "TF-IDF";
    ui["result-meta"].textContent = `${method}${data.rerank ? " + 重排" : ""} · 本次总耗时 ${elapsed}s`;
    setMessage(generate ? "回答完成。展开下方片段，即可核对来源、页码与原文。" : "检索完成，没有调用 DeepSeek。");
  });
}

ui["question-form"].addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(true);
});
ui["search-button"].addEventListener("click", () => submitQuestion(false));
ui.question.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !ui["ask-button"].disabled) {
    event.preventDefault();
    ui["question-form"].requestSubmit();
  }
});
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    ui.question.value = button.dataset.question;
    ui.question.focus();
  });
});
ui.retriever.addEventListener("change", () => {
  ui.rerank.checked = ui.retriever.value === "embedding";
  syncControls();
});
ui.rerank.addEventListener("change", syncControls);
ui["refresh-status"].addEventListener("click", () => runAction("正在刷新文献库状态…", async () => {
  const data = await refreshIndex();
  setMessage(data.embedding?.status === "ready" ? "文献库已连接，可以直接提问。" : "文献库已连接；使用语义检索前，请先建立 E5 索引。也可切换到 TF-IDF。");
}));
ui["load-documents"].addEventListener("click", () => runAction("正在读取文献、切片并保存索引…", async () => {
  const paths = ui["document-paths"].value.split(/\r?\n/).map((line) => {
    const path = line.trim();
    return ((path.startsWith('"') && path.endsWith('"')) || (path.startsWith("'") && path.endsWith("'"))) ? path.slice(1, -1) : path;
  }).filter(Boolean);
  if (!paths.length) throw new Error("请填写至少一个文件完整路径，一行一个。");
  await requestJSON("/documents/load", "POST", { paths });
  state.index = null;
  clearResult();
  const data = await refreshIndex();
  setMessage(data.embedding?.status === "ready" ? "文献已加载，现有 E5 索引匹配，可以直接提问。" : "文献已加载并保存。请点击“建立 E5 语义索引”，完成后即可语义问答。");
}));
ui["build-embedding"].addEventListener("click", () => runAction("正在本机建立 E5 索引；这一步不调用 DeepSeek，请等待完成…", async () => {
  await requestJSON("/index/embedding/build", "POST");
  state.index = null;
  await refreshIndex();
  setMessage("E5 语义索引已建立并保存，可以开始提问。");
}));

// 页面启动只读状态；不自动建索引、不自动运行问答或调用付费 API。
runAction("正在连接本地服务…", async () => {
  const data = await refreshIndex();
  setMessage(data.embedding?.status === "ready" ? "文献库与语义索引已就绪，输入问题即可开始。" : "已连接本地服务。请先建立 E5 索引，或切换到 TF-IDF 检索。");
});
