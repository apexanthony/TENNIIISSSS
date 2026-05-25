const state = {
  config: null,
  selectedJobId: null,
  selectedResultPath: null,
  activeView: "infer",
  pendingPreviewJobId: null,
  pendingPreviewPath: null,
  previewFallbackTimer: null,
};

const $ = (id) => document.getElementById(id);
const VIDEO_EXTENSIONS = [".webm", ".mp4", ".avi", ".mov", ".mkv"];

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 2800);
}

function fmtSize(bytes) {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function shortPath(item) {
  return item.rel || item.path || item.name || "";
}

function fileUrl(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}

function previewUrl(path) {
  return `/api/preview?path=${encodeURIComponent(path)}&width=960&t=${Date.now()}`;
}

function isVideoName(name) {
  const lower = (name || "").toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function findVideoOutput(files) {
  return (files || []).find((file) => isVideoName(file.name));
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "请求失败");
  return data;
}

function setView(view) {
  state.activeView = view;
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `view-${view}`);
  });
}

function addOption(select, item, selectedPath) {
  const opt = document.createElement("option");
  opt.value = item.path;
  opt.textContent = shortPath(item);
  if (selectedPath && item.path === selectedPath) opt.selected = true;
  select.appendChild(opt);
}

function fillSelect(id, items, selectedPath, allowEmpty = false) {
  const select = $(id);
  select.innerHTML = "";
  if (allowEmpty) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "不使用";
    select.appendChild(opt);
  }
  items.forEach((item) => addOption(select, item, selectedPath));
}

function selectedValue(id, fallback) {
  const el = $(id);
  return el ? el.value : fallback;
}

async function loadConfig() {
  const previous = {
    inferVideo: selectedValue("inferVideo", null),
    ptModel: selectedValue("ptModel", null),
    onnxModel: selectedValue("onnxModel", null),
    evalModel: selectedValue("evalModel", null),
    trainResume: selectedValue("trainResume", null),
  };

  state.config = await api("/api/config");
  $("projectRoot").textContent = state.config.project_root;

  fillSelect("inferVideo", state.config.videos, previous.inferVideo || state.config.defaults.video);
  fillSelect("ptModel", state.config.pt_models, previous.ptModel || state.config.defaults.pt_model);
  fillSelect("onnxModel", state.config.onnx_models, previous.onnxModel || state.config.defaults.onnx_model);
  fillSelect("evalModel", state.config.pt_models, previous.evalModel || state.config.defaults.pt_model);
  fillSelect(
    "trainResume",
    state.config.pt_models,
    previous.trainResume !== null ? previous.trainResume : state.config.defaults.pt_model,
    true,
  );

  renderJobs(state.config.jobs || []);
  renderResults(state.config.outputs || []);
}

function renderJobs(jobs) {
  const list = $("jobsList");
  list.innerHTML = "";
  if (!jobs.length) {
    list.innerHTML = '<div class="status">暂无任务</div>';
    return;
  }

  jobs.forEach((job) => {
    const item = document.createElement("div");
    item.className = `job-item ${job.id === state.selectedJobId ? "active" : ""}`;
    item.onclick = () => selectJob(job.id);
    const status = job.status || "unknown";
    item.innerHTML = `
      <div class="job-main">
        <span>${job.kind}</span>
        <span class="status ${status}">${status}</span>
      </div>
      <div class="status">${job.id}</div>
      <div class="status">PID ${job.pid || "-"}</div>
    `;
    list.appendChild(item);
  });
}

function renderResults(files) {
  const list = $("resultList");
  list.innerHTML = "";
  const visible = files
    .filter((file) => file.name && (isVideoName(file.name) || file.name.endsWith(".csv")))
    .slice(0, 80);

  if (!visible.length) {
    list.innerHTML = '<div class="status">暂无输出</div>';
    return;
  }

  visible.forEach((file) => {
    const item = document.createElement("div");
    item.className = `result-item ${file.path === state.selectedResultPath ? "active" : ""}`;
    item.onclick = () => selectResult(file);
    item.innerHTML = `
      <strong>${file.name}</strong>
      <span class="status">${shortPath(file)}</span>
      <span class="status">${fmtSize(file.size)}</span>
    `;
    list.appendChild(item);
  });
}

function clearPreviewFallbackTimer() {
  if (state.previewFallbackTimer) {
    clearTimeout(state.previewFallbackTimer);
    state.previewFallbackTimer = null;
  }
}

function showStreamPreview(path, message) {
  clearPreviewFallbackTimer();
  const video = $("previewVideo");
  const stream = $("previewStream");

  video.onerror = null;
  video.oncanplay = null;
  video.pause();
  video.removeAttribute("src");
  video.load();
  video.classList.add("hidden");

  stream.src = previewUrl(path);
  stream.classList.remove("hidden");
  $("previewStatus").textContent = message;
}

function showVideoPreview(path) {
  clearPreviewFallbackTimer();
  const video = $("previewVideo");
  const stream = $("previewStream");

  stream.removeAttribute("src");
  stream.classList.add("hidden");
  video.classList.remove("hidden");

  video.onerror = () => {
    if (state.selectedResultPath === path) {
      showStreamPreview(path, "浏览器无法直接播放该视频编码，已切换为帧流预览。");
    }
  };
  video.oncanplay = () => {
    clearPreviewFallbackTimer();
    $("previewStatus").textContent = "正在预览：" + path.split(/[\\/]/).pop();
  };

  video.src = fileUrl(path);
  video.load();
  $("previewStatus").textContent = "正在加载视频预览...";

  state.previewFallbackTimer = setTimeout(() => {
    if (state.selectedResultPath === path && video.readyState === 0) {
      showStreamPreview(path, "视频预览加载较慢，已切换为帧流预览。");
    }
  }, 2500);
}

function selectResult(file, auto = false) {
  if (!file || !file.path) return;
  state.selectedResultPath = file.path;
  if (isVideoName(file.name)) {
    showVideoPreview(file.path);
  } else if (!auto) {
    window.open(fileUrl(file.path), "_blank");
  }
  renderResults(state.config.outputs || []);
}

async function previewFinishedInference(job) {
  if (!job || job.id !== state.pendingPreviewJobId || job.status === "running") return;

  if (job.status === "failed") {
    state.pendingPreviewJobId = null;
    state.pendingPreviewPath = null;
    toast("推理失败，请查看日志");
    return;
  }

  if (job.status !== "success") return;

  await loadConfig();
  const jobOutput = findVideoOutput(job.outputs);
  const targetPath = (jobOutput && jobOutput.path) || state.pendingPreviewPath;
  const output =
    (state.config.outputs || []).find((file) => file.path === targetPath) ||
    jobOutput ||
    (targetPath ? { path: targetPath, name: targetPath.split(/[\\/]/).pop() } : null);

  state.pendingPreviewJobId = null;
  state.pendingPreviewPath = null;

  if (output) {
    setView("results");
    selectResult(output, true);
    toast("推理完成，已加载预览");
  }
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  const data = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
  $("jobLog").textContent = data.job.log || "";
  const jobs = await api("/api/jobs");
  renderJobs(jobs.jobs || []);
}

async function refreshJobsOnly() {
  const data = await api("/api/jobs");
  renderJobs(data.jobs || []);

  if (!state.selectedJobId) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.selectedJobId)}`);
    $("jobLog").textContent = job.job.log || "";
    $("jobLog").scrollTop = $("jobLog").scrollHeight;
    await previewFinishedInference(job.job);
  } catch (_) {
    state.selectedJobId = null;
  }
}

function engineChanged() {
  const isPt = $("inferEngine").value === "pt";
  document.querySelectorAll(".pt-only").forEach((el) => el.classList.toggle("hidden", !isPt));
  document.querySelectorAll(".onnx-only").forEach((el) => el.classList.toggle("hidden", isPt));
}

async function uploadVideo(event) {
  event.preventDefault();
  const file = $("videoFile").files[0];
  if (!file) {
    toast("请选择视频文件");
    return;
  }

  const form = new FormData();
  form.append("video", file);
  const data = await api("/api/upload", { method: "POST", body: form });
  toast(`已上传 ${data.file.name}`);
  await loadConfig();
  $("inferVideo").value = data.file.path;
}

async function startInfer() {
  const engine = $("inferEngine").value;
  const payload = {
    engine,
    video: $("inferVideo").value,
    model: engine === "pt" ? $("ptModel").value : $("onnxModel").value,
    input_width: Number($("inferWidth").value),
    input_height: Number($("inferHeight").value),
    threshold: Number($("inferThreshold").value),
    peak_window: Number($("inferPeakWindow").value),
    trace: Number($("traceLength").value),
    batch_size: Number($("inferBatch").value),
    device: $("inferDevice").value,
    target: $("onnxTarget").value,
  };
  const data = await api("/api/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.selectedJobId = data.job.id;
  state.pendingPreviewJobId = data.job.id;
  const videoOutput = findVideoOutput(data.job.outputs);
  state.pendingPreviewPath = videoOutput ? videoOutput.path : null;
  toast("推理任务已启动");
  await refreshJobsOnly();
}

async function startEval() {
  const payload = {
    model: $("evalModel").value,
    input_width: Number($("evalWidth").value),
    input_height: Number($("evalHeight").value),
    heatmap_radius: Number($("evalRadius").value),
    heatmap_sigma: Number($("evalSigma").value),
    peak_window: Number($("evalPeakWindow").value),
    batch_size: Number($("evalBatch").value),
    thresholds: $("evalThresholds").value,
    device: $("evalDevice").value,
  };
  const data = await api("/api/eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.selectedJobId = data.job.id;
  toast("测试任务已启动");
  await refreshJobsOnly();
}

async function startTrain() {
  const trainHeight = Number($("trainHeight").value);
  const payload = {
    exp_id: $("trainExpId").value,
    resume: $("trainResume").value,
    start_epoch: Number($("trainStartEpoch").value),
    num_epochs: Number($("trainEpochs").value),
    lr: Number($("trainLr").value),
    batch_size: Number($("trainBatch").value),
    steps_per_epoch: Number($("trainSteps").value),
    val_intervals: Number($("trainValIntervals").value),
    input_width: Number($("trainWidth").value),
    input_height: trainHeight,
    heatmap_radius: Math.max(1, Math.round(trainHeight / 45)),
    heatmap_sigma: Math.max(0.5, trainHeight / 120),
    pos_weight: Number($("trainPosWeight").value),
    threshold: Number($("trainThreshold").value),
    device: $("trainDevice").value,
    amp: $("trainAmp").checked,
    augment: $("trainAugment").checked,
  };
  const data = await api("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.selectedJobId = data.job.id;
  toast("训练任务已启动");
  await refreshJobsOnly();
}

async function stopSelectedJob() {
  if (!state.selectedJobId) {
    toast("请选择任务");
    return;
  }
  await api(`/api/jobs/${encodeURIComponent(state.selectedJobId)}/stop`, { method: "POST" });
  toast("已发送停止命令");
  await refreshJobsOnly();
}

async function safeRun(fn) {
  try {
    await fn();
  } catch (err) {
    toast(err.message || String(err));
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  document.querySelectorAll(".tab").forEach((button) => {
    button.onclick = () => setView(button.dataset.view);
  });

  $("refreshBtn").onclick = () => safeRun(loadConfig);
  $("uploadForm").onsubmit = (event) => safeRun(() => uploadVideo(event));
  $("inferEngine").onchange = engineChanged;
  $("startInferBtn").onclick = () => safeRun(startInfer);
  $("startEvalBtn").onclick = () => safeRun(startEval);
  $("startTrainBtn").onclick = () => safeRun(startTrain);
  $("stopJobBtn").onclick = () => safeRun(stopSelectedJob);

  setView("infer");
  engineChanged();
  await safeRun(loadConfig);
  setInterval(() => safeRun(refreshJobsOnly), 2500);
  setInterval(() => safeRun(loadConfig), 15000);
});
