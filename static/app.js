const COPY = {
  zh: {
    eyebrow: "教学估算 · 不占用 GPU",
    title: "训练模拟台",
    lede: "配置集群和框架，看显存、通信和一步训练分别花在哪里。",
    langGroup: "语言",
    preset: "预设",
    hCluster: "集群",
    nodes: "节点数",
    gpusPerNode: "每节点 GPU",
    gpu: "GPU",
    interconnect: "节点间网络",
    hFramework: "框架",
    framework: "训练框架",
    zero: "ZeRO 阶段",
    zero1: "ZeRO-1 切分优化器状态",
    zero2: "ZeRO-2 再切分梯度",
    zero3: "ZeRO-3 再切分参数",
    tp: "张量并行",
    pp: "流水线并行",
    offload: "优化器状态放到 CPU",
    hModel: "模型与批次",
    model: "模型",
    precision: "精度",
    micro: "微批次",
    accum: "梯度累积",
    seq: "序列长度",
    lr: "学习率",
    activationCkpt: "激活重计算",
    hData: "数据",
    dataset: "语料",
    pack: "序列拼接",
    hRecipe: "配方",
    schedule: "学习率日程",
    schedConstant: "先升温再保持",
    schedCosine: "先升温再余弦衰减",
    gradClip: "梯度裁剪",
    weightDecay: "权重衰减",
    hJob: "作业",
    ckptEvery: "检查点间隔",
    resumeStep: "续训起始步",
    evalEvery: "验证间隔",
    fault: "故障",
    faultNone: "无",
    faultTimeout: "集合通信超时",
    hPlay: "播放",
    steps: "步数",
    speed: "速度",
    start: "开始模拟",
    stop: "停止",
    memTitle: "单卡显存",
    timeTitle: "一步时间",
    estimateTag: "教学估算，不是跑分",
    statLoss: "损失",
    statTps: "吞吐",
    statMfu: "FLOPs 利用率",
    statStep: "步时",
    statSpeedup: "加速比",
    chartCaption: "横轴是步，纵轴是损失。曲线用于观察下降和跳动。词表约 32000 时，真实交叉熵的起点约在 10 nat 附近；这条曲线是教学形状，不是这份语料的成绩。",
    chartAria: "教学损失曲线",
    formulasSummary: "估算说明",
    logTitle: "模拟日志",
    logHint: "每一行都来自估算模型",
    logRegion: "模拟日志",
    legendLabel: "阶段图例",
    statusReady: "就绪",
    statusInit: "初始化",
    statusRunning: "训练中",
    statusOom: "显存不足",
    statusDone: "已完成",
    statusStopped: "已停止",
    statusError: "配置错误",
    statusTimeout: "通信超时",
    phaseIdle: "尚未开始",
    phaseData: "取数",
    phaseForward: "前向",
    phaseBackward: "反向",
    phaseComm: "通信",
    phaseOptim: "优化器",
    phaseCheckpoint: "检查点",
    phaseFail: "失败",
    catalogFail: "目录接口返回 ",
    catalogMissing: "目录没有读到",
    catalogHint: "请先启动服务后再打开这个页面。",
    incomplete: "有数值还没填完整，或落在允许范围外。",
    httpRejected: "请求没有被接受（",
    topologyBad: "这组并行度排不出进程",
    runTitle: "集群",
    topologyLoading: "正在读取目录…",
    speedup: "加速比等于单卡完成同样多 token 的计算时间除以当前步时。",
    speedupSingle: "单卡放不下这份模型时，这个比值只比较算力和通信。",
    memParam: "参数",
    memGrad: "梯度",
    memOptim: "优化器",
    memAct: "激活",
    memTransient: "临时拼参",
    memOverhead: "开销",
    memTotal: "合计 ",
    memOffload: "CPU 上的优化器 ",
    timeData: "取数 ",
    timeCompute: "计算 ",
    timeExposed: "暴露通信 ",
    timeOptim: "优化器 ",
    timeComm: "通信总量 ",
    bubble: "，气泡 ",
    logPlaceholder: "开始模拟后，这里按启动、同步、训练步的顺序出现日志。",
    starting: "启动中",
    playbackBroken: "[模拟] 播放中断：",
    stepWord: "步 ",
    oomBus: "分配失败，训练步没有开始",
    oomStep: "分配失败",
    playbackDone: "播放结束",
    notStarted: "未开始",
    jobTitle: "整次作业",
    statTokens: "每步 token",
    statFullSteps: "整次步数",
    statWall: "墙钟时间",
    statGpuHours: "GPU 时",
    notExtrapolated: "未外推",
    ckptSize: "检查点约 ",
    ckptTime: "，按 2 GB/s 估算 ",
    ckptNotInStep: "。未计入预览步时。",
    ckptInPlayback: "。保存那一步的播放会加上这段时间。",
    timeoutStep: "通信超时",
    unitDays: " 天",
    unitHours: " 小时",
    unitMinutes: " 分钟",
    unitSeconds: " 秒",
    chartLoss: "损失",
    chartStep: "步",
    splitConfigW: "调整配置区宽度",
    splitSideW: "调整指标区宽度",
    splitLog: "调整日志区高度",
    splitConfigH: "调整配置区高度",
    splitStageH: "调整集群区和指标区高度",
  },
  en: {
    eyebrow: "Teaching estimate · no GPU used",
    title: "Training simulator",
    lede: "Set the cluster and framework, then see where memory, communication, and one training step go.",
    langGroup: "Language",
    preset: "Preset",
    hCluster: "Cluster",
    nodes: "Nodes",
    gpusPerNode: "GPUs per node",
    gpu: "GPU",
    interconnect: "Inter-node network",
    hFramework: "Framework",
    framework: "Training framework",
    zero: "ZeRO stage",
    zero1: "ZeRO-1 shards optimizer state",
    zero2: "ZeRO-2 also shards gradients",
    zero3: "ZeRO-3 also shards parameters",
    tp: "Tensor parallel",
    pp: "Pipeline parallel",
    offload: "Keep optimizer state on the CPU",
    hModel: "Model and batch",
    model: "Model",
    precision: "Precision",
    micro: "Micro-batch",
    accum: "Gradient accumulation",
    seq: "Sequence length",
    lr: "Learning rate",
    activationCkpt: "Activation recomputation",
    hData: "Data",
    dataset: "Corpus",
    pack: "Pack sequences",
    hRecipe: "Recipe",
    schedule: "Learning-rate schedule",
    schedConstant: "Warmup, then hold",
    schedCosine: "Warmup, then cosine",
    gradClip: "Gradient clipping",
    weightDecay: "Weight decay",
    hJob: "Job",
    ckptEvery: "Checkpoint interval",
    resumeStep: "Resume step",
    evalEvery: "Eval interval",
    fault: "Fault",
    faultNone: "None",
    faultTimeout: "Collective timeout",
    hPlay: "Playback",
    steps: "Steps",
    speed: "Speed",
    start: "Start",
    stop: "Stop",
    memTitle: "Memory on one GPU",
    timeTitle: "One step",
    estimateTag: "Teaching estimate, not a benchmark",
    statLoss: "Loss",
    statTps: "Throughput",
    statMfu: "FLOPs utilization",
    statStep: "Step time",
    statSpeedup: "Speedup",
    chartCaption: "The horizontal axis is the step and the vertical axis is loss. The curve shows the drop and the jumps. With a vocabulary of about 32000, real cross-entropy starts near 10 nats. This curve is a teaching shape, not a score on this corpus.",
    chartAria: "Teaching loss curve",
    formulasSummary: "How the estimate is made",
    logTitle: "Simulated log",
    logHint: "Every line comes from the estimate",
    logRegion: "Simulated log",
    legendLabel: "Phase legend",
    statusReady: "Ready",
    statusInit: "Starting",
    statusRunning: "Training",
    statusOom: "Out of memory",
    statusDone: "Finished",
    statusStopped: "Stopped",
    statusError: "Bad config",
    statusTimeout: "Timed out",
    phaseIdle: "Not started",
    phaseData: "Data",
    phaseForward: "Forward",
    phaseBackward: "Backward",
    phaseComm: "Comm",
    phaseOptim: "Optimizer",
    phaseCheckpoint: "Checkpoint",
    phaseFail: "Failed",
    catalogFail: "Catalog request returned ",
    catalogMissing: "Catalog was not loaded",
    catalogHint: "Start the server, then open this page.",
    incomplete: "A value is missing or outside the allowed range.",
    httpRejected: "The request was not accepted (",
    topologyBad: "This parallel layout does not fit on the processes",
    runTitle: "Cluster",
    topologyLoading: "Loading the catalog…",
    speedup: "Speedup is the single-GPU compute time for the same tokens, divided by the current step time.",
    speedupSingle: " When one GPU cannot hold the model, the ratio only compares compute and communication.",
    memParam: "Params",
    memGrad: "Grads",
    memOptim: "Optimizer",
    memAct: "Activation",
    memTransient: "Gather",
    memOverhead: "Overhead",
    memTotal: "Total ",
    memOffload: "Optimizer on CPU ",
    timeData: "Data ",
    timeCompute: "Compute ",
    timeExposed: "Exposed comm ",
    timeOptim: "Optimizer ",
    timeComm: "All comm ",
    bubble: ", bubble ",
    logPlaceholder: "After you start, lines appear in order: launch, sync, then training steps.",
    starting: "Starting",
    playbackBroken: "[sim] Playback interrupted: ",
    stepWord: "Step ",
    oomBus: "Allocation failed. Training steps did not start",
    oomStep: "Allocation failed",
    playbackDone: "Playback finished",
    notStarted: "Not started",
    jobTitle: "Full job",
    statTokens: "Tokens per step",
    statFullSteps: "Full-job steps",
    statWall: "Wall time",
    statGpuHours: "GPU-hours",
    notExtrapolated: "Not extrapolated",
    ckptSize: "Checkpoint about ",
    ckptTime: ", about ",
    ckptNotInStep: " at 2 GB/s. Not included in the preview step.",
    ckptInPlayback: " at 2 GB/s. A save step in playback adds this time.",
    timeoutStep: "Timed out",
    unitDays: " d",
    unitHours: " h",
    unitMinutes: " min",
    unitSeconds: " s",
    chartLoss: "Loss",
    chartStep: "Step",
    splitConfigW: "Resize the config pane",
    splitSideW: "Resize the metrics pane",
    splitLog: "Resize the log pane",
    splitConfigH: "Resize the config pane height",
    splitStageH: "Resize the cluster and metrics heights",
  },
};

const STATUS_KEY = {
  ready: "statusReady",
  init: "statusInit",
  running: "statusRunning",
  oom: "statusOom",
  done: "statusDone",
  stopped: "statusStopped",
  error: "statusError",
  timeout: "statusTimeout",
};

const PHASE_KEY = {
  idle: "phaseIdle",
  data: "phaseData",
  forward: "phaseForward",
  backward: "phaseBackward",
  comm: "phaseComm",
  optim: "phaseOptim",
  checkpoint: "phaseCheckpoint",
  fail: "phaseFail",
};

let lang = "zh";
let phaseId = "idle";
let phaseBus = "";
let stepKind = "idle";
let stepNums = { step: 0, steps: 0 };

function tr(key) {
  const pack = COPY[lang] || COPY.zh;
  return pack[key] !== undefined ? pack[key] : key;
}

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
  const stored = window.localStorage.getItem("llm-sim-lang");
  lang = stored === "en" ? "en" : "zh";
  applyStatic();
  document.querySelectorAll(".lang-btn").forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.lang));
  });
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
    const response = await fetch("/api/catalog?lang=" + lang);
    if (!response.ok) throw new Error(tr("catalogFail") + response.status);
    catalog = await response.json();
    fillSelect($("gpu"), catalog.gpus);
    fillSelect($("interconnect"), catalog.interconnects);
    fillSelect($("framework"), catalog.frameworks);
    fillSelect($("model"), catalog.models);
    fillSelect($("precision"), catalog.precisions);
    fillSelect($("dataset"), catalog.datasets || []);
    fillSelect($("preset"), catalog.presets.map((item) => ({ id: item.id, name: item.name })));
    applyPreset(catalog.presets[0]);
  } catch (error) {
    $("topologyLine").textContent = tr("catalogMissing");
    showBanner("errors", tr("catalogHint") + "\n" + error.message);
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
  $("dataset").value = cfg.dataset || "none";
  $("packSequences").checked = cfg.pack_sequences !== false;
  $("lrSchedule").value = cfg.lr_schedule || "warmup_constant";
  $("gradClip").value = cfg.grad_clip ?? 1;
  $("weightDecay").value = cfg.weight_decay ?? 0.1;
  $("ckptEvery").value = cfg.ckpt_every_steps || 0;
  $("resumeStep").value = cfg.resume_step || 0;
  $("evalEvery").value = cfg.eval_every_steps || 0;
  $("fault").value = cfg.fault || "none";
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
    dataset: $("dataset").value || "none",
    pack_sequences: $("packSequences").checked,
    lr_schedule: $("lrSchedule").value,
    weight_decay: Number($("weightDecay").value),
    grad_clip: Number($("gradClip").value),
    ckpt_every_steps: intOf("ckptEvery"),
    resume_step: intOf("resumeStep"),
    eval_every_steps: intOf("evalEvery"),
    fault: $("fault").value,
    steps: intOf("steps"),
    lr: Number($("lr").value),
    speed: Number($("speed").value),
    lang,
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
    showBanner("errors", tr("incomplete"));
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
  const ints = [
    cfg.nodes, cfg.gpus_per_node, cfg.zero_stage, cfg.tp, cfg.pp,
    cfg.micro_batch, cfg.grad_accum, cfg.seq_len, cfg.steps,
    cfg.ckpt_every_steps, cfg.resume_step, cfg.eval_every_steps,
  ];
  if (ints.some((value) => !Number.isInteger(value))) return false;
  if (!Number.isFinite(cfg.lr) || !Number.isFinite(cfg.speed)) return false;
  if (!Number.isFinite(cfg.weight_decay) || !Number.isFinite(cfg.grad_clip)) return false;
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
    return tr("httpRejected") + response.status + ").";
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
    $("topologyLine").textContent = tr("topologyBad");
    $("runTitle").textContent = tr("runTitle");
    $("cluster").replaceChildren();
  } else {
    const perNode = nodes[0] ? nodes[0].gpus.length : 0;
    $("topologyLine").textContent = lang === "en"
      ? nodes.length + (nodes.length === 1 ? " node" : " nodes")
        + " × " + perNode + (perNode === 1 ? " GPU" : " GPUs")
        + " = " + parallel.world + (parallel.world === 1 ? " process" : " processes")
      : nodes.length + " 节点 × " + perNode + " 卡 = " + parallel.world + " 进程";
    const labels = plan.labels;
    $("runTitle").textContent = labels.framework + " · " + labels.model;
    renderCluster(plan);
  }
  renderMemory(plan.memory);
  renderTime(plan.timing);
  applyTimingStats(plan.timing);
  renderJob(plan.job);
  renderNotes(plan.advice, plan.notes);
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
  let text = tr("speedup");
  if (!timing.single_gpu_fits) {
    text += tr("speedupSingle");
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
    [tr("memParam"), memory.param_gb],
    [tr("memGrad"), memory.grad_gb],
    [tr("memOptim"), memory.optim_gb],
    [tr("memAct"), memory.activation_gb],
    [tr("memTransient"), memory.transient_gb],
    [tr("memOverhead"), memory.overhead_gb],
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
  strong.textContent = tr("memTotal") + memory.total_gb.toFixed(2) + " / " + memory.gpu_gb + " GB";
  total.append(strong);
  $("memText").append(total);
  if (memory.optim_offload_gb > 0) {
    const extra = document.createElement("span");
    extra.textContent = tr("memOffload") + memory.optim_offload_gb.toFixed(2) + " GB";
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
  const bubble = timing.bubble_frac > 0 ? tr("bubble") + Math.round(timing.bubble_frac * 100) + "%" : "";
  $("timeText").textContent =
    tr("timeData") + formatMs(timing.data_ms) +
    " · " + tr("timeCompute") + formatMs(timing.compute_ms) +
    " · " + tr("timeExposed") + formatMs(timing.exposed_comm_ms) +
    " · " + tr("timeOptim") + formatMs(timing.optim_ms) +
    " · " + tr("timeComm") + formatMs(timing.comm_ms) +
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

function renderJob(job) {
  if (!job) {
    $("jobTokens").textContent = "—";
    $("jobSteps").textContent = "—";
    $("jobWall").textContent = "—";
    $("jobGpuHours").textContent = "—";
    $("ckptLine").textContent = "";
    return;
  }
  $("jobTokens").textContent = formatTokens(job.tokens_per_step);
  $("jobSteps").textContent = job.full_steps ? formatCount(job.full_steps) : tr("notExtrapolated");
  const wall = job.full_wall_s != null ? job.full_wall_s : job.playback_wall_s;
  $("jobWall").textContent = formatWall(wall);
  $("jobGpuHours").textContent = job.gpu_hours != null ? job.gpu_hours.toFixed(1) : tr("notExtrapolated");
  const ckpt = job.checkpoint || { ckpt_bytes: 0, ckpt_ms: 0, every_steps: 0 };
  const gb = ckpt.ckpt_bytes / (1024 ** 3);
  $("ckptLine").textContent =
    tr("ckptSize") + gb.toFixed(2) + " GB" +
    tr("ckptTime") + formatMs(ckpt.ckpt_ms) +
    (ckpt.every_steps > 0 ? tr("ckptInPlayback") : tr("ckptNotInStep"));
}

function renderNotes(advice, notes) {
  const root = $("notes");
  root.replaceChildren();
  for (const item of advice || []) appendNote(item);
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
  pill.textContent = tr(STATUS_KEY[status] || "") || status;
}

function setPhase(phase, bus) {
  phaseId = phase || "idle";
  phaseBus = bus || "";
  $("phaseLabel").textContent = tr(PHASE_KEY[phaseId] || "") || phaseId;
  $("busLabel").textContent = phaseBus;
  document.querySelectorAll(".gpu").forEach((chip) => {
    chip.dataset.phase = phase === "idle" ? "idle" : phase;
  });
}

function setPlaceholder() {
  const line = document.createElement("div");
  line.className = "placeholder";
  line.textContent = tr("logPlaceholder");
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
  stepKind = "starting";
  writeStepLabel();
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
      appendLog("error", tr("playbackBroken") + error.message);
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
    stepKind = "step";
    stepNums = { step: event.step, steps: event.steps };
    writeStepLabel();
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
      stepKind = "oom";
      setPhase("fail", tr("oomBus"));
      writeStepLabel();
    }
    if (event.status === "done") {
      stepKind = "done";
      writeStepLabel();
    }
    if (event.status === "error") {
      stepKind = "error";
      writeStepLabel();
    }
    if (event.status === "timeout") {
      stepKind = "timeout";
      setPhase("fail", tr("timeoutStep"));
      writeStepLabel();
    }
    if (event.status === "stopped") stepKind = "stopped";
  }
}

function writeStepLabel() {
  if (stepKind === "starting") $("stepLabel").textContent = tr("starting");
  else if (stepKind === "step") $("stepLabel").textContent = tr("stepWord") + stepNums.step + " / " + stepNums.steps;
  else if (stepKind === "oom") $("stepLabel").textContent = tr("oomStep");
  else if (stepKind === "timeout") $("stepLabel").textContent = tr("timeoutStep");
  else if (stepKind === "done") $("stepLabel").textContent = tr("playbackDone");
  else if (stepKind === "error") $("stepLabel").textContent = tr("notStarted");
  else $("stepLabel").textContent = tr("phaseIdle");
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

function formatCount(value) {
  return Math.round(value).toLocaleString(lang === "en" ? "en-US" : "zh-CN");
}

function formatTokens(value) {
  const n = Number(value) || 0;
  if (lang === "zh") {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + " 万亿";
    if (n >= 1e8) return (n / 1e8).toFixed(2) + " 亿";
    if (n >= 1e4) return (n / 1e4).toFixed(1) + " 万";
    return formatCount(n);
  }
  if (n >= 1e12) return (n / 1e12).toFixed(2) + " T";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + " M";
  return formatCount(n);
}

function formatWall(seconds) {
  const n = Number(seconds) || 0;
  if (n >= 86400) return (n / 86400).toFixed(1) + tr("unitDays");
  if (n >= 3600) return (n / 3600).toFixed(1) + tr("unitHours");
  if (n >= 60) return (n / 60).toFixed(1) + tr("unitMinutes");
  return Math.round(n) + tr("unitSeconds");
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
  ctx.fillText(tr("chartLoss"), 2, 12);
  ctx.fillText(tr("chartStep"), width - 18, height - 4);
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
      ? (handle.dataset.resize === "config" ? tr("splitConfigH") : tr("splitStageH"))
      : (handle.dataset.resize === "config" ? tr("splitConfigW") : tr("splitSideW")));
  }
  const logHandle = document.querySelector('.splitter[data-resize="log"]');
  if (logHandle) logHandle.setAttribute("aria-label", tr("splitLog"));
}

function applyStatic() {
  document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
  document.title = tr("title");
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = tr(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
    node.setAttribute("aria-label", tr(node.dataset.i18nAria));
  });
  document.querySelectorAll(".lang-btn").forEach((button) => {
    button.setAttribute("aria-pressed", button.dataset.lang === lang ? "true" : "false");
  });
  if ($("statusPill")) setStatus($("statusPill").dataset.status || "ready");
  writeStepLabel();
  if ($("phaseLabel")) setPhase(phaseId, phaseBus);
  if (!catalog && $("topologyLine")) $("topologyLine").textContent = tr("topologyLoading");
  if (typeof narrowQuery !== "undefined") syncSplitterAxes();
}

async function setLanguage(next) {
  if (next !== "zh" && next !== "en") return;
  if (next === lang) return;
  lang = next;
  window.localStorage.setItem("llm-sim-lang", lang);
  applyStatic();
  if (!catalog) return;
  const selected = ["preset", "gpu", "interconnect", "framework", "model", "precision", "dataset"]
    .reduce((acc, id) => { acc[id] = $(id).value; return acc; }, {});
  try {
    const response = await fetch("/api/catalog?lang=" + lang);
    if (!response.ok) throw new Error(tr("catalogFail") + response.status);
    catalog = await response.json();
    fillSelect($("gpu"), catalog.gpus);
    fillSelect($("interconnect"), catalog.interconnects);
    fillSelect($("framework"), catalog.frameworks);
    fillSelect($("model"), catalog.models);
    fillSelect($("precision"), catalog.precisions);
    fillSelect($("dataset"), catalog.datasets || []);
    fillSelect($("preset"), catalog.presets.map((item) => ({ id: item.id, name: item.name })));
    for (const [id, value] of Object.entries(selected)) $(id).value = value;
    const preset = catalog.presets.find((item) => item.id === $("preset").value);
    if (preset) $("presetSummary").textContent = preset.summary;
    syncFramework();
  } catch (error) {
    showBanner("errors", tr("catalogHint") + "\n" + error.message);
    return;
  }
  if ($("log").querySelector(".placeholder")) setPlaceholder();
  if (plan) {
    renderMemory(plan.memory);
    renderTime(plan.timing);
    renderJob(plan.job);
    $("speedupNote").textContent = speedupText(plan.timing);
    drawChart(losses);
  }
  if (!running) preview();
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
