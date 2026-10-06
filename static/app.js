const STATUS_TEXT = {
  ready: "就绪",
  init: "初始化",
  running: "训练中",
  oom: "显存不足",
  done: "已完成",
  stopped: "已停止",
  error: "配置错误",
};

const PHASE_TEXT = {
  idle: "尚未开始",
  data: "取数",
  forward: "前向",
  backward: "反向",
  comm: "通信",
  optim: "优化器",
  checkpoint: "检查点",
  fail: "失败",
};

const $ = (id) => document.getElementById(id);

let catalog = null;
let plan = null;
let canStart = false;
let running = false;
let runId = 0;
let abortCtrl = null;
let losses = [];
let previewGen = 0;

document.addEventListener("DOMContentLoaded", async () => {
  $("configForm").addEventListener("submit", (event) => {
    event.preventDefault();
    start();
  });
  $("configForm").addEventListener("input", schedulePreview);
  $("configForm").addEventListener("change", schedulePreview);
  $("preset").addEventListener("change", () => {
    const preset = catalog.presets.find((item) => item.id === $("preset").value);
    if (preset) applyPreset(preset);
  });
  $("stopBtn").addEventListener("click", stop);
  installSplitters();
  window.addEventListener("resize", () => {
    clampPanes();
    drawChart(losses);
  });
  setPlaceholder();
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok) throw new Error("目录接口返回 " + response.status);
    catalog = await response.json();
    fillSelect($("gpu"), catalog.gpus);
    fillSelect($("interconnect"), catalog.interconnects);
    fillSelect($("framework"), catalog.frameworks);
    fillSelect($("model"), catalog.models);
    fillSelect($("precision"), catalog.precisions);
    fillSelect($("preset"), catalog.presets.map((item) => ({ id: item.id, name: item.name })));
    applyPreset(catalog.presets[0]);
  } catch (error) {
    $("topologyLine").textContent = "目录没有读到";
    showBanner("errors", "请先启动服务后再打开这个页面。\n" + error.message);
  }
});

function fillSelect(select, options) {
  select.replaceChildren();
  for (const option of options) {
    const node = document.createElement("option");
    node.value = option.id;
    node.textContent = option.name;
    select.append(node);
  }
}

function applyPreset(preset) {
  const cfg = preset.config;
  $("preset").value = preset.id;
  $("presetSummary").textContent = preset.summary;
  $("nodes").value = cfg.nodes;
  $("gpusPerNode").value = cfg.gpus_per_node;
  $("gpu").value = cfg.gpu;
  $("interconnect").value = cfg.interconnect;
  $("framework").value = cfg.framework;
  $("zeroStage").value = String(cfg.zero_stage);
  $("tp").value = cfg.tp;
  $("pp").value = cfg.pp;
  $("cpuOffload").checked = cfg.cpu_offload;
  $("model").value = cfg.model;
  $("precision").value = cfg.precision;
  $("microBatch").value = cfg.micro_batch;
  $("gradAccum").value = cfg.grad_accum;
  $("seqLen").value = cfg.seq_len;
  $("lr").value = cfg.lr;
  $("activationCheckpoint").checked = cfg.activation_checkpointing;
  $("steps").value = cfg.steps;
  $("speed").value = String(cfg.speed);
  syncFramework();
  preview();
}

function syncFramework() {
  const framework = $("framework").value;
  const found = catalog && catalog.frameworks.find((item) => item.id === framework);
  $("frameworkBlurb").textContent = found ? found.blurb : "";
  $("zeroRow").hidden = framework !== "deepspeed";
  $("offloadRow").hidden = framework !== "deepspeed";
  $("parallelRow").hidden = framework !== "megatron";
}

function schedulePreview() {
  syncFramework();
  window.clearTimeout(schedulePreview.timer);
  schedulePreview.timer = window.setTimeout(preview, 180);
}

function gather() {
  const framework = $("framework").value;
  return {
    nodes: intOf("nodes"),
    gpus_per_node: intOf("gpusPerNode"),
    gpu: $("gpu").value,
    interconnect: $("interconnect").value,
    framework,
    zero_stage: framework === "deepspeed" ? intOf("zeroStage") : 1,
    tp: framework === "megatron" ? intOf("tp") : 1,
    pp: framework === "megatron" ? intOf("pp") : 1,
    model: $("model").value,
    micro_batch: intOf("microBatch"),
    grad_accum: intOf("gradAccum"),
    seq_len: intOf("seqLen"),
    precision: $("precision").value,
    activation_checkpointing: $("activationCheckpoint").checked,
    cpu_offload: framework === "deepspeed" ? $("cpuOffload").checked : false,
    steps: intOf("steps"),
    lr: Number($("lr").value),
    speed: Number($("speed").value),
  };
}

function intOf(id) {
  return Number.parseInt($(id).value, 10);
}

async function preview() {
  if (running || !catalog) return;
  const cfg = gather();
  if (!configLooksComplete(cfg)) {
    canStart = false;
    lock(false);
    showBanner("errors", "有数值还没填完整，或落在允许范围外。");
    return;
  }
  const gen = ++previewGen;
  const response = await fetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  if (gen !== previewGen || running) return;
  if (!response.ok) {
    canStart = false;
    lock(false);
    showBanner("errors", await explainHttp(response));
    return;
  }
  renderPlan(await response.json(), false);
}

function configLooksComplete(cfg) {
  const ints = [cfg.nodes, cfg.gpus_per_node, cfg.zero_stage, cfg.tp, cfg.pp, cfg.micro_batch, cfg.grad_accum, cfg.seq_len, cfg.steps];
  if (ints.some((value) => !Number.isInteger(value))) return false;
  if (!Number.isFinite(cfg.lr) || !Number.isFinite(cfg.speed)) return false;
  return true;
}

async function explainHttp(response) {
  try {
    const body = await response.json();
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg).join("\n");
    }
    return JSON.stringify(body.detail || body);
  } catch (_error) {
    return "请求没有被接受（" + response.status + "）。";
  }
}

function renderPlan(next, fromRun) {
  plan = next;
  canStart = plan.errors.length === 0;
  showBanner("errors", plan.errors.join("\n"));
  showBanner("warnings", plan.warnings.join("\n"));
  const parallel = plan.parallel;
  const nodes = plan.cluster.nodes || [];
  if (plan.errors.length) {
    $("topologyLine").textContent = "这组并行度排不出进程";
    $("runTitle").textContent = "集群";
    $("cluster").replaceChildren();
  } else {
    const perNode = nodes[0] ? nodes[0].gpus.length : 0;
    $("topologyLine").textContent = nodes.length + " 节点 × " + perNode + " 卡 = " + parallel.world + " 进程";
    const labels = plan.labels;
    $("runTitle").textContent = labels.framework + " · " + labels.model;
    renderCluster(plan);
  }
  renderMemory(plan.memory);
  renderTime(plan.timing);
  applyTimingStats(plan.timing);
  renderNotes(plan.notes);
  renderFormulas(plan.formulas);
  $("speedupNote").textContent = speedupText(plan.timing);
  if (!fromRun) {
    losses = [];
    $("loss").textContent = "—";
    setPhase("idle", "");
    if (!running) setStatus(plan.errors.length ? "error" : "ready");
    requestAnimationFrame(() => drawChart(losses));
  }
  lock(running);
}

function speedupText(timing) {
  if (!timing) return "";
  let text = "加速比等于单卡完成同样多 token 的计算时间除以当前步时。";
  if (!timing.single_gpu_fits) {
    text += "单卡放不下这份模型时，这个比值只比较算力和通信。";
  }
  return text;
}

function renderCluster(next) {
  const root = $("cluster");
  root.replaceChildren();
  const megatron = next.parallel.tp > 1 || next.parallel.pp > 1;
  for (const node of next.cluster.nodes) {
    const card = document.createElement("article");
    card.className = "node";
    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent = "node" + node.id;
    const host = document.createElement("span");
    host.textContent = node.host;
    header.append(title, host);
    const gpus = document.createElement("div");
    gpus.className = "gpus";
    for (const gpu of node.gpus) {
      const chip = document.createElement("div");
      chip.className = "gpu";
      chip.dataset.rank = String(gpu.rank);
      chip.dataset.phase = "idle";
      chip.textContent = megatron
        ? "r" + gpu.rank + "\ntp" + gpu.tp + " pp" + gpu.pp
        : "r" + gpu.rank + "\ndp" + gpu.dp;
      gpus.append(chip);
    }
    card.append(header, gpus);
    root.append(card);
  }
}

function renderMemory(memory) {
  const bar = $("memBar");
  bar.replaceChildren();
  if (!memory || !memory.gpu_gb) {
    $("memText").textContent = "";
    return;
  }
  const parts = [
    ["param", memory.param_gb],
    ["grad", memory.grad_gb],
    ["optim", memory.optim_gb],
    ["act", memory.activation_gb],
    ["transient", memory.transient_gb],
    ["overhead", memory.overhead_gb],
  ];
  const scale = Math.max(memory.total_gb, memory.gpu_gb, 0.001);
  bar.classList.toggle("overflow", !memory.fits);
  for (const [name, value] of parts) {
    if (value <= 0) continue;
    const seg = document.createElement("div");
    seg.className = "seg " + name;
    seg.style.width = (value / scale) * 100 + "%";
    bar.append(seg);
  }
  if (memory.total_gb > memory.gpu_gb) {
    const cap = document.createElement("div");
    cap.className = "cap";
    cap.style.left = (memory.gpu_gb / scale) * 100 + "%";
    bar.append(cap);
  }
  const bits = [
    ["参数", memory.param_gb],
    ["梯度", memory.grad_gb],
    ["优化器", memory.optim_gb],
    ["激活", memory.activation_gb],
    ["临时拼参", memory.transient_gb],
    ["开销", memory.overhead_gb],
  ];
  $("memText").replaceChildren();
  for (const [name, value] of bits) {
    const span = document.createElement("span");
    span.innerHTML = "";
    span.append(document.createTextNode(name + " " + value.toFixed(2) + " GB"));
    $("memText").append(span);
  }
  const total = document.createElement("span");
  const strong = document.createElement("b");
  strong.textContent = "合计 " + memory.total_gb.toFixed(2) + " / " + memory.gpu_gb + " GB";
  total.append(strong);
  $("memText").append(total);
  if (memory.optim_offload_gb > 0) {
    const extra = document.createElement("span");
    extra.textContent = "CPU 上的优化器 " + memory.optim_offload_gb.toFixed(2) + " GB";
    $("memText").append(extra);
  }
}

function renderTime(timing) {
  const bar = $("timeBar");
  bar.replaceChildren();
  bar.classList.remove("overflow");
  if (!timing || !timing.step_ms) {
    $("timeText").textContent = "";
    return;
  }
  const parts = [
    ["data", timing.data_ms],
    ["compute", timing.compute_ms],
    ["exposed", timing.exposed_comm_ms],
    ["opt", timing.optim_ms],
  ];
  const scale = Math.max(timing.step_ms, 0.001);
  for (const [name, value] of parts) {
    if (value <= 0) continue;
    const seg = document.createElement("div");
    seg.className = "seg " + name;
    seg.style.width = (value / scale) * 100 + "%";
    bar.append(seg);
  }
  const bubble = timing.bubble_frac > 0 ? "，气泡 " + Math.round(timing.bubble_frac * 100) + "%" : "";
  $("timeText").textContent =
    "取数 " + formatMs(timing.data_ms) +
    " · 计算 " + formatMs(timing.compute_ms) +
    " · 暴露通信 " + formatMs(timing.exposed_comm_ms) +
    " · 优化器 " + formatMs(timing.optim_ms) +
    " · 通信总量 " + formatMs(timing.comm_ms) +
    bubble;
}

function applyTimingStats(timing) {
  if (!timing || !timing.step_ms) {
    $("tps").textContent = "—";
    $("mfu").textContent = "—";
    $("stepTime").textContent = "—";
    $("speedup").textContent = "—";
    return;
  }
  $("tps").textContent = formatTok(timing.tokens_per_s);
  $("mfu").textContent = (timing.mfu * 100).toFixed(1) + "%";
  $("stepTime").textContent = formatMs(timing.step_ms);
  $("speedup").textContent = timing.speedup.toFixed(2) + "×";
}

function renderNotes(notes) {
  const root = $("notes");
  root.replaceChildren();
  for (const note of notes || []) appendNote(note);
}

function appendNote(note) {
  if ($("notes").querySelector('[data-id="' + cssEscape(note.id) + '"]')) return;
  const card = document.createElement("article");
  card.className = "note";
  card.dataset.level = note.level;
  card.dataset.id = note.id;
  const title = document.createElement("h3");
  title.textContent = note.title;
  const body = document.createElement("p");
  body.textContent = note.body;
  card.append(title, body);
  $("notes").append(card);
}

function renderFormulas(formulas) {
  const root = $("formulas");
  root.replaceChildren();
  for (const text of formulas || []) {
    const p = document.createElement("p");
    p.textContent = text;
    root.append(p);
  }
}

function showBanner(id, text) {
  const node = $(id);
  node.hidden = !text;
  node.textContent = text || "";
}

function setStatus(status) {
  const pill = $("statusPill");
  pill.dataset.status = status;
  pill.textContent = STATUS_TEXT[status] || status;
}

function setPhase(phase, bus) {
  $("phaseLabel").textContent = PHASE_TEXT[phase] || phase;
  $("busLabel").textContent = bus || "";
  document.querySelectorAll(".gpu").forEach((chip) => {
    chip.dataset.phase = phase === "idle" ? "idle" : phase;
  });
}

function setPlaceholder() {
  const line = document.createElement("div");
  line.className = "placeholder";
  line.textContent = "开始模拟后，这里按启动、同步、训练步的顺序出现日志。";
  $("log").replaceChildren(line);
}

function clearLog() {
  $("log").replaceChildren();
}

function appendLog(level, text) {
  const log = $("log");
  const placeholder = log.querySelector(".placeholder");
  if (placeholder) placeholder.remove();
  const stick = log.scrollHeight - log.scrollTop - log.clientHeight < 48;
  const line = document.createElement("div");
  line.className = "line " + (level || "info");
  line.textContent = text;
  log.append(line);
  if (stick) log.scrollTop = log.scrollHeight;
}

async function start() {
  if (!canStart || running) return;
  const cfg = gather();
  const id = ++runId;
  abortCtrl?.abort();
  abortCtrl = new AbortController();
  running = true;
  losses = [];
  drawChart(losses);
  clearLog();
  setStatus("init");
  $("stepLabel").textContent = "启动中";
  lock(true);
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
      signal: abortCtrl.signal,
    });
    if (id !== runId) return;
    if (!response.ok) {
      showBanner("errors", await explainHttp(response));
      setStatus("error");
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (id === runId) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      let index = buffer.indexOf("\n");
      while (index >= 0) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (line) handle(JSON.parse(line));
        index = buffer.indexOf("\n");
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      if (id === runId) setStatus("stopped");
      return;
    }
    if (id === runId) {
      appendLog("error", "[模拟] 播放中断：" + error.message);
      setStatus("error");
    }
  } finally {
    if (id === runId) {
      running = false;
      lock(false);
    }
  }
}

function stop() {
  if (!running) return;
  abortCtrl?.abort();
}

function handle(event) {
  if (event.kind === "plan") {
    renderPlan(event.plan, true);
    return;
  }
  if (event.kind === "log") {
    appendLog(event.level, event.text);
    return;
  }
  if (event.kind === "phase") {
    setPhase(event.phase, event.bus || "");
    return;
  }
  if (event.kind === "metrics") {
    losses.push(event.loss);
    drawChart(losses);
    $("loss").textContent = event.loss.toFixed(3);
    $("tps").textContent = formatTok(event.tokens_per_s);
    $("mfu").textContent = (event.mfu * 100).toFixed(1) + "%";
    $("stepTime").textContent = formatMs(event.step_ms);
    $("stepLabel").textContent = "步 " + event.step + " / " + event.steps;
    renderTime({
      data_ms: event.data_ms,
      compute_ms: event.compute_ms,
      exposed_comm_ms: event.exposed_comm_ms,
      optim_ms: event.optim_ms,
      comm_ms: plan ? plan.timing.comm_ms : 0,
      step_ms: event.step_ms,
      bubble_frac: event.bubble_frac,
    });
    return;
  }
  if (event.kind === "note") {
    appendNote(event.note);
    return;
  }
  if (event.kind === "status") {
    setStatus(event.status);
    if (event.status === "oom") {
      setPhase("fail", "分配失败，训练步没有开始");
      $("stepLabel").textContent = "分配失败";
    }
    if (event.status === "done") $("stepLabel").textContent = "播放结束";
    if (event.status === "error") $("stepLabel").textContent = "未开始";
  }
}

function lock(on) {
  running = on;
  document.querySelectorAll("#configForm input, #configForm select").forEach((el) => {
    el.disabled = on;
  });
  $("startBtn").disabled = on || !canStart;
  $("stopBtn").disabled = !on;
}

function formatMs(ms) {
  if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
  return Math.round(ms) + " ms";
}

function formatTok(value) {
  if (value >= 1e6) return (value / 1e6).toFixed(2) + " M/s";
  if (value >= 1e3) return (value / 1e3).toFixed(1) + " k/s";
  return Math.round(value) + " /s";
}

function drawChart(points) {
  const canvas = $("lossChart");
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 280;
  const height = canvas.clientHeight || 120;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const left = 32;
  const right = 8;
  const top = 16;
  const bottom = 20;
  ctx.strokeStyle = "#3c4a38";
  ctx.fillStyle = "#93a08a";
  ctx.font = "11px IBM Plex Sans, Noto Sans SC, sans-serif";
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, height - bottom);
  ctx.lineTo(width - right, height - bottom);
  ctx.stroke();
  ctx.fillText("损失", 2, 12);
  ctx.fillText("步", width - 18, height - 4);
  if (!points.length) return;
  const maxY = Math.max(5, ...points) * 1.08;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  ctx.strokeStyle = "#d6f25c";
  ctx.fillStyle = "#d6f25c";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((value, index) => {
    const x = left + (points.length === 1 ? plotW / 2 : (index / (points.length - 1)) * plotW);
    const y = top + (1 - value / maxY) * plotH;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  if (points.length === 1) {
    const x = left + plotW / 2;
    const y = top + (1 - points[0] / maxY) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function cssEscape(value) {
  if (window.CSS && CSS.escape) return CSS.escape(value);
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "");
}

const PANE_MIN = { config: 200, side: 200, stage: 240, log: 96, workspace: 140 };
const STACK_MIN = 180;
const STACK_MAX = 1600;
const narrowQuery = window.matchMedia("(max-width: 1100px)");

function installSplitters() {
  for (const handle of document.querySelectorAll(".splitter")) {
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      beginPaneDrag(handle, event);
    });
    handle.addEventListener("keydown", (event) => {
      const step = event.shiftKey ? 48 : 16;
      const delta = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[event.key];
      if (!delta) return;
      event.preventDefault();
      nudgePane(handle.dataset.resize, delta[0], delta[1]);
    });
  }
  narrowQuery.addEventListener("change", syncSplitterAxes);
  syncSplitterAxes();
  clampPanes();
}

function syncSplitterAxes() {
  const stacked = narrowQuery.matches;
  for (const handle of document.querySelectorAll(".splitter-col")) {
    handle.setAttribute("aria-orientation", stacked ? "horizontal" : "vertical");
    handle.setAttribute("aria-label", stacked
      ? (handle.dataset.resize === "config" ? "调整配置区高度" : "调整集群区和指标区高度")
      : (handle.dataset.resize === "config" ? "调整配置区宽度" : "调整指标区宽度"));
  }
}

function beginPaneDrag(handle, event) {
  const kind = handle.dataset.resize;
  const originX = event.clientX;
  const originY = event.clientY;
  const start = {
    config: readPx("--config-w"),
    side: readPx("--side-w"),
    log: readPx("--log-h"),
    configH: readPx("--config-h"),
    stageH: readPx("--stage-h"),
    sideH: readPx("--side-h"),
  };
  handle.classList.add("is-dragging");
  document.body.classList.add("is-resizing");
  document.body.style.setProperty("--resize-cursor", narrowQuery.matches || kind === "log" ? "row-resize" : "col-resize");
  if (handle.setPointerCapture) {
    try { handle.setPointerCapture(event.pointerId); } catch (err) { /* 指针尚未按下时忽略 */ }
  }

  const move = (ev) => {
    applyPaneDelta(kind, ev.clientX - originX, ev.clientY - originY, start);
  };
  const end = () => {
    handle.classList.remove("is-dragging");
    document.body.classList.remove("is-resizing");
    document.body.style.removeProperty("--resize-cursor");
    handle.removeEventListener("pointermove", move);
    handle.removeEventListener("pointerup", end);
    handle.removeEventListener("pointercancel", end);
    drawChart(losses);
  };
  handle.addEventListener("pointermove", move);
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
}

function nudgePane(kind, dx, dy) {
  const start = {
    config: readPx("--config-w"),
    side: readPx("--side-w"),
    log: readPx("--log-h"),
    configH: readPx("--config-h"),
    stageH: readPx("--stage-h"),
    sideH: readPx("--side-h"),
  };
  applyPaneDelta(kind, dx, dy, start);
  drawChart(losses);
}

function applyPaneDelta(kind, dx, dy, start) {
  if (narrowQuery.matches) {
    if (kind === "config") setPx("--config-h", clamp(start.configH + dy, STACK_MIN, STACK_MAX));
    else if (kind === "side") tradeStackedHeight(start.stageH, start.sideH, dy);
    else setPx("--log-h", clamp(start.log - dy, PANE_MIN.log, STACK_MAX));
    return;
  }
  const room = paneRoom();
  if (kind === "config") setPx("--config-w", clamp(start.config + dx, PANE_MIN.config, room.maxConfig));
  else if (kind === "side") setPx("--side-w", clamp(start.side - dx, PANE_MIN.side, room.maxSide));
  else setPx("--log-h", clamp(start.log - dy, PANE_MIN.log, room.maxLog));
}

function tradeStackedHeight(stageStart, sideStart, dy) {
  let delta = dy;
  let side = sideStart - delta;
  if (side < STACK_MIN) {
    side = STACK_MIN;
    delta = sideStart - side;
  } else if (side > STACK_MAX) {
    side = STACK_MAX;
    delta = sideStart - side;
  }
  let stage = stageStart + delta;
  if (stage < STACK_MIN) {
    stage = STACK_MIN;
    side = clamp(sideStart - (stage - stageStart), STACK_MIN, STACK_MAX);
  } else if (stage > STACK_MAX) {
    stage = STACK_MAX;
    side = clamp(sideStart - (stage - stageStart), STACK_MIN, STACK_MAX);
  }
  setPx("--stage-h", stage);
  setPx("--side-h", side);
}

function clampPanes() {
  if (narrowQuery.matches) return;
  const width = document.querySelector(".workspace").clientWidth;
  const split = readPx("--split");
  const maxSide = Math.max(PANE_MIN.side, width - PANE_MIN.config - PANE_MIN.stage - split * 2);
  const side = clamp(readPx("--side-w"), PANE_MIN.side, maxSide);
  setPx("--side-w", side);
  const maxConfig = Math.max(PANE_MIN.config, width - side - PANE_MIN.stage - split * 2);
  setPx("--config-w", clamp(readPx("--config-w"), PANE_MIN.config, maxConfig));
  setPx("--log-h", clamp(readPx("--log-h"), PANE_MIN.log, paneRoom().maxLog));
}

function paneRoom() {
  const workspace = document.querySelector(".workspace");
  const split = readPx("--split");
  const width = workspace.clientWidth;
  const config = readPx("--config-w");
  const side = readPx("--side-w");
  const header = document.querySelector(".top").offsetHeight;
  return {
    maxConfig: Math.max(PANE_MIN.config, width - side - PANE_MIN.stage - split * 2),
    maxSide: Math.max(PANE_MIN.side, width - config - PANE_MIN.stage - split * 2),
    maxLog: Math.max(PANE_MIN.log, document.documentElement.clientHeight - header - split - PANE_MIN.workspace),
  };
}

function readPx(name) {
  const value = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
  return Number.isFinite(value) ? value : 0;
}

function setPx(name, value) {
  document.documentElement.style.setProperty(name, Math.round(value) + "px");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
