let cfg = null;
const policyRows = {};
const STAGE_ORDER = [
  "reception", "audio_decode", "scaler_load", "mlp_load", "metadata_load",
  "yamnet_load", "embedding", "scaling", "mlp_classification", "aggregation",
  "other_overhead",
];
const STAGE_LABELS = {
  reception: "Audio file reception",
  audio_decode: "Audio decoding and preprocessing",
  scaler_load: "Feature scaler (first use)",
  mlp_load: "MLP model (first use)",
  metadata_load: "Class metadata (first use)",
  yamnet_load: "Pretrained YAMNet (first use)",
  embedding: "YAMNet embedding extraction",
  scaling: "Feature scaling",
  mlp_classification: "MLP classification",
  aggregation: "Aggregation + supervisor",
  other_overhead: "Other request overhead",
};
const PHASE_ORDER = [
  "received", "audio_decode", "embedding", "scaling", "mlp_classification",
  "aggregation", "supervisor", "completed",
];
let modelTimer = null;
let historyTimer = null;
let latestRequestId = null;

function el(id) { return document.getElementById(id); }

async function apiGet(path) {
  const res = await fetch(path, { cache: "no-store" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function setStatus(node, text, isError) {
  if (!node) return;
  node.textContent = text || "";
  try { node.classList.toggle("error", Boolean(isError)); } catch (e) {}
}

function fmtMs(ms) {
  const n = Number(ms);
  if (!isFinite(n) || n < 0) return "—";
  if (n < 1000) return n.toFixed(n < 10 ? 2 : 1) + " ms";
  return (n / 1000).toFixed(2) + " s";
}

function fmtSeconds(seconds) {
  const n = Number(seconds);
  if (!isFinite(n) || n < 0) return "—";
  return n.toFixed(Math.abs(n) < 10 ? 2 : 1) + " s";
}

function newRequestId() {
  return "req-" + Date.now().toString(36) + "-" +
    Math.random().toString(36).slice(2, 9);
}

function modeNamed(sub) {
  const found = cfg.modes.find((m) => m.toLowerCase().includes(sub));
  return found || cfg.modes[0];
}

function buildModeSelect(select, defaultMode) {
  select.innerHTML = "";
  cfg.modes.forEach((mode) => {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = mode;
    select.appendChild(option);
  });
  if (defaultMode) select.value = defaultMode;
}

function numberCell(className, value, step, min, max) {
  const td = document.createElement("td");
  const input = document.createElement("input");
  input.type = "number";
  input.className = className;
  input.value = value;
  input.step = step;
  input.min = min;
  if (max !== undefined) input.max = max;
  td.appendChild(input);
  return { td, input };
}

function checkboxCell(className, checked) {
  const td = document.createElement("td");
  const input = document.createElement("input");
  input.type = "checkbox";
  input.className = className;
  input.checked = Boolean(checked);
  td.appendChild(input);
  return { td, input };
}

function buildPolicyTable() {
  const tbody = el("policy-table").querySelector("tbody");
  tbody.innerHTML = "";
  const defaults = cfg.policy_defaults;
  cfg.classes.forEach((name) => {
    const tr = document.createElement("tr");
    tr.dataset.className = name;
    const on = checkboxCell("p-enabled", false);
    tr.appendChild(on.td);
    const nameTd = document.createElement("td");
    nameTd.textContent = name;
    tr.appendChild(nameTd);
    const targetTd = document.createElement("td");
    const target = document.createElement("select");
    target.className = "p-target";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = cfg.no_action_mode_label;
    target.appendChild(none);
    cfg.modes.forEach((mode) => {
      const option = document.createElement("option");
      option.value = mode;
      option.textContent = mode;
      target.appendChild(option);
    });
    target.value = "";
    targetTd.appendChild(target);
    tr.appendChild(targetTd);
    const conf = numberCell("p-conf", defaults.confidence_threshold, 0.05, 0.01, 1);
    const confirm = numberCell("p-confirm", defaults.confirm_windows, 1, 1);
    const release = numberCell("p-release", defaults.release_seconds, 0.5, 0);
    const cool = numberCell("p-cool", defaults.cooldown_seconds, 0.5, 0);
    const priority = checkboxCell("p-priority", false);
    const critical = checkboxCell("p-critical", false);
    [conf, confirm, release, cool, priority, critical].forEach((cell) => tr.appendChild(cell.td));
    const labelTd = document.createElement("td");
    const qalabel = document.createElement("input");
    qalabel.type = "text";
    qalabel.className = "p-qalabel";
    qalabel.value = "";
    labelTd.appendChild(qalabel);
    tr.appendChild(labelTd);
    const notify = checkboxCell("p-notify", true);
    tr.appendChild(notify.td);
    policyRows[name] = {
      enabled: on.input, target, conf: conf.input, confirm: confirm.input,
      release: release.input, cool: cool.input, priority: priority.input,
      critical: critical.input, qalabel, notify: notify.input,
    };
    tbody.appendChild(tr);
  });
}

function buildClipSelect() {
  const select = el("clip-select");
  select.innerHTML = "";
  const groups = {};
  cfg.clips.forEach((clip) => {
    if (!groups[clip.label]) groups[clip.label] = [];
    groups[clip.label].push(clip);
  });
  cfg.classes.forEach((name) => {
    const clips = groups[name];
    if (!clips) return;
    const optgroup = document.createElement("optgroup");
    optgroup.label = name;
    clips.forEach((clip) => {
      const option = document.createElement("option");
      option.value = clip.clip_id;
      const dur = clip.duration_seconds != null
        ? " (" + clip.duration_seconds.toFixed(1) + "s)" : "";
      option.textContent = clip.source_dataset + " / " + clip.file_name + dur;
      optgroup.appendChild(option);
    });
    select.appendChild(optgroup);
  });
}

function setRow(name, spec) {
  const row = policyRows[name];
  if (!row) return;
  row.enabled.checked = true;
  if (spec.target !== undefined) row.target.value = spec.target || "";
  if (spec.conf !== undefined) row.conf.value = spec.conf;
  if (spec.confirm !== undefined) row.confirm.value = spec.confirm;
  if (spec.priority) row.priority.checked = true;
  if (spec.critical) {
    row.critical.checked = true;
    row.qalabel.value = spec.qalabel || "";
  }
}

function clearAll() {
  cfg.classes.forEach((name) => {
    const row = policyRows[name];
    row.enabled.checked = false;
    row.priority.checked = false;
    row.critical.checked = false;
    row.qalabel.value = "";
  });
}

function presetCommuting() {
  clearAll();
  el("preferred-mode").value = modeNamed("anc");
  el("profile-name").value = "Commuting (example)";
  setRow("Siren", { target: modeNamed("safety"), conf: 0.8, confirm: 1, priority: true });
  setRow("Vehicle_Horn", { target: modeNamed("transparency"), conf: 0.8, confirm: 2 });
  setRow("Car_Engine", { target: modeNamed("anc"), conf: 0.8, confirm: 3 });
  setRow("Motorcycle", { target: modeNamed("anc"), conf: 0.8, confirm: 3 });
  setRow("Train", { target: modeNamed("anc"), conf: 0.8, confirm: 3 });
}

function presetHome() {
  clearAll();
  el("preferred-mode").value = modeNamed("anc");
  el("profile-name").value = "Home and Family (example)";
  setRow("Baby_Crying", { target: modeNamed("transparency"), conf: 0.8, confirm: 2, priority: true });
  setRow("Doorbell", { target: modeNamed("transparency"), conf: 0.8, confirm: 1 });
  setRow("Knocking", { target: modeNamed("transparency"), conf: 0.8, confirm: 2 });
  setRow("Dog_Bark", { target: modeNamed("transparency"), conf: 0.8, confirm: 3 });
  setRow("Alarm", { target: modeNamed("safety"), conf: 0.8, confirm: 1, priority: true });
  setRow("Glass_Breaking", {
    target: modeNamed("safety"), conf: 0.8, confirm: 1, priority: true,
    critical: true, qalabel: "Call emergency contact",
  });
  setRow("Gunshot", {
    target: modeNamed("safety"), conf: 0.8, confirm: 1, priority: true,
    critical: true, qalabel: "Call emergency contact",
  });
}

function buildPreferencesPayload() {
  const policies = {};
  cfg.classes.forEach((name) => {
    const row = policyRows[name];
    if (!row.enabled.checked) {
      policies[name] = { enabled: false };
      return;
    }
    policies[name] = {
      enabled: true,
      target_mode: row.target.value || null,
      confidence_threshold: Number(row.conf.value),
      confirm_windows: parseInt(row.confirm.value, 10),
      release_seconds: Number(row.release.value),
      cooldown_seconds: Number(row.cool.value),
      high_priority: row.priority.checked,
      critical: row.critical.checked,
      quick_action_label: row.qalabel.value.trim() || null,
      notify: row.notify.checked,
    };
  });
  return {
    profile_name: el("profile-name").value.trim() || "Custom",
    preferred_mode: el("preferred-mode").value || null,
    policies,
  };
}

function renderList(node, items) {
  node.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.text;
    if (item.cls) li.className = item.cls;
    node.appendChild(li);
  });
}

// ---------------------------------------------------------- model status UI

function renderModel(status) {
  const pill = el("model-pill");
  const chip = el("model-status-chip");
  const state = status.status || "not_started";
  const pillState = state === "not_started" ? "initializing" : state;
  pill.dataset.state = pillState;
  pill.textContent = {
    not_started: "MODEL: PENDING",
    initializing: "MODEL: LOADING",
    ready: "MODEL READY",
    failed: "MODEL FAILED",
  }[state] || "MODEL: " + state.toUpperCase();
  chip.textContent = state.toUpperCase();
  setStatus(el("model-error"), state === "failed" ? (status.error || "Initialization failed") : "", state === "failed");
  el("model-retry").classList.toggle("hidden", state !== "failed");
  el("model-meta").textContent = (status.model || "YAMNet + MLP") +
    " · artifacts: " + (status.model_dir || "local project path");

  const rows = Array.isArray(status.stages) ? status.stages : [];
  const list = el("init-stages");
  list.innerHTML = "";
  rows.forEach((row) => {
    const li = document.createElement("li");
    li.dataset.state = row.state || "pending";
    const dot = document.createElement("span");
    dot.className = "stage-dot";
    const name = document.createElement("span");
    name.className = "stage-name";
    name.textContent = row.label || row.key;
    const time = document.createElement("span");
    time.className = "stage-time" + (row.state === "pending" ? " pending-time" : "");
    time.textContent = row.state === "pending" ? "—" : fmtMs(row.duration_ms);
    li.appendChild(dot); li.appendChild(name); li.appendChild(time);
    list.appendChild(li);
  });
  const done = rows.filter((r) => r.state === "done").length;
  const pct = state === "ready" ? 100 : (rows.length ? Math.round((done / rows.length) * 100) : 0);
  el("init-progress-fill").style.width = pct + "%";
  el("init-total").textContent = state === "ready" || state === "failed"
    ? fmtMs(status.total_ms) : "measuring…";
  el("stat-init").textContent = state === "ready" || state === "failed"
    ? fmtMs(status.total_ms) : "—";
}

async function refreshModel() {
  try {
    const status = await apiGet("/api/model");
    renderModel(status);
    if (status.status === "initializing") {
      if (!modelTimer) modelTimer = setInterval(refreshModel, 400);
    } else if (modelTimer) {
      clearInterval(modelTimer);
      modelTimer = null;
      refreshHistory();
    }
  } catch (err) {
    if (modelTimer) { clearInterval(modelTimer); modelTimer = null; }
    setStatus(el("model-error"), "Could not read model status: " + err.message, true);
  }
}

// ----------------------------------------------------------- activity UI

function renderActivity(track) {
  const dot = el("activity-dot");
  const timeline = el("activity-timeline");
  const running = track && track.status === "running";
  const failed = track && track.status === "failed";
  dot.classList.toggle("running", Boolean(running));
  dot.classList.toggle("failed", Boolean(failed));
  dot.innerHTML = "<i></i> " + (running ? "PROCESSING" : failed ? "FAILED" :
    track ? "IDLE" : "IDLE");

  if (!track) {
    el("activity-phase").textContent = "Waiting for audio input";
    el("activity-detail").textContent = "Submit a bundled clip or upload a WAV to start inference.";
    el("activity-timer").textContent = "0 ms";
    el("activity-fill").style.width = "0%";
    timeline.innerHTML = '<li class="empty">No processing events yet.</li>';
    return;
  }

  el("activity-phase").textContent = track.phase_label || PHASE_LABELS[track.phase] || track.phase;
  el("activity-detail").textContent =
    (track.file_name || "input") + " · " + (track.source || "audio");
  el("activity-timer").textContent = running ? fmtMs(track.elapsed_ms) :
    (track.status === "failed" ? "error" : "done");
  // Progress follows the real backend phase count, not elapsed wall time.
  const index = PHASE_ORDER.indexOf(track.phase);
  const pct = track.status === "completed" ? 100 :
    Math.round(((index + 1) / PHASE_ORDER.length) * 100);
  el("activity-fill").style.width = Math.max(0, pct) + "%";
  timeline.innerHTML = "";
  (track.events || []).forEach((event) => {
    const li = document.createElement("li");
    const marker = document.createElement("span");
    marker.className = "event-dot";
    const label = document.createElement("span");
    label.className = "event-label";
    label.textContent = event.label || event.stage;
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = "+" + fmtMs(event.at_ms);
    li.appendChild(marker); li.appendChild(label); li.appendChild(time);
    timeline.appendChild(li);
  });
}

async function pollActivity(requestId) {
  try {
    const data = await apiGet("/api/activity?request_id=" + encodeURIComponent(requestId));
    if (requestId !== latestRequestId) return; // never mix concurrent requests
    renderActivity(data.track || data.current);
  } catch (err) {
    // Polling is observational; a transient failure must not break inference.
  }
}

function beginActivity(requestId, fileName, source) {
  latestRequestId = requestId;
  renderActivity({
    request_id: requestId, file_name: fileName, source,
    phase: "received", phase_label: "Receiving audio file",
    status: "running", elapsed_ms: 0,
    events: [{ stage: "received", label: "Receiving audio file", at_ms: 0 }],
  });
  return setInterval(() => pollActivity(requestId), 130);
}

function endActivity(timer, requestId) {
  clearInterval(timer);
  pollActivity(requestId);
}

// ------------------------------------------------------- prediction/history

function renderPrediction(prediction, timing) {
  const pct = (prediction.confidence * 100).toFixed(1);
  const lines = ["Predicted: " + prediction.label + "  (" + pct + "%)"];
  const file = prediction.file_name ||
    (prediction.clip ? prediction.clip.file_name : "");
  if (prediction.clip) {
    lines.push("Clip: " + prediction.clip.source_dataset + " / " + prediction.clip.file_name);
    lines.push("Dataset label for this clip: " + prediction.clip.label);
  } else {
    if (prediction.file_name) lines.push("File: " + prediction.file_name);
    if (prediction.duration_seconds != null) {
      lines.push("Duration: " + Number(prediction.duration_seconds).toFixed(2) + "s");
    }
    lines.push("Source: uploaded WAV");
  }
  const hero = el("prediction");
  hero.textContent = lines.join("\n");
  hero.classList.remove("error");
  hero.classList.remove("flash");
  void hero.offsetWidth;
  hero.classList.add("flash");
  el("result-chip").textContent = "COMPLETED";
  el("result-file").textContent = file || "—";
  el("result-file").title = file || "";
  el("result-audio").textContent = prediction.duration_seconds != null
    ? fmtSeconds(prediction.duration_seconds)
    : (prediction.clip && prediction.clip.duration_seconds != null
      ? fmtSeconds(prediction.clip.duration_seconds) : "—");
  el("result-total").textContent = timing ? fmtMs(timing.total_ms) : "—";
  el("result-confidence").textContent = pct + "%";

  const stages = el("result-stages");
  stages.innerHTML = "";
  if (timing && timing.stages_ms) {
    STAGE_ORDER.forEach((key) => {
      if (!(key in timing.stages_ms)) return;
      const row = document.createElement("div");
      row.className = "stage-row";
      const label = document.createElement("span");
      label.textContent = STAGE_LABELS[key] || key;
      const value = document.createElement("strong");
      value.textContent = fmtMs(timing.stages_ms[key]);
      row.appendChild(label); row.appendChild(value);
      stages.appendChild(row);
    });
  }
  const bars = el("top-k");
  bars.innerHTML = "";
  prediction.top_k.forEach((pair) => {
    const name = pair[0];
    const prob = pair[1];
    const bar = document.createElement("div");
    bar.className = "bar";
    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = name;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = Math.max(1, prob * 100).toFixed(0) + "%";
    track.appendChild(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = (prob * 100).toFixed(1) + "%";
    bar.appendChild(label); bar.appendChild(track); bar.appendChild(value);
    bars.appendChild(bar);
  });
}

function renderHistory(data) {
  const s = data.stats || {};
  el("stat-count").textContent = String(s.count || 0);
  el("stat-avg").textContent = s.count ? fmtMs(s.avg_ms) : "—";
  el("stat-min").textContent = s.count ? fmtMs(s.min_ms) : "—";
  el("stat-max").textContent = s.count ? fmtMs(s.max_ms) : "—";
  if (!s.init_ms && el("stat-init").textContent === "—") {
    el("stat-init").textContent = "—";
  }
  const tbody = el("history-table").querySelector("tbody");
  const rows = data.history || [];
  tbody.innerHTML = "";
  rows.slice().reverse().forEach((row) => {
    const tr = document.createElement("tr");
    const cells = [
      (() => { try { return new Date(row.timestamp).toLocaleTimeString(); } catch (e) { return String(row.timestamp || ""); } })(),
      row.file_name || "—",
      row.label || "—",
      row.confidence != null ? (row.confidence * 100).toFixed(1) + "%" : "—",
      fmtMs(row.total_ms),
      row.decision || "—",
    ];
    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  el("history-empty").classList.toggle("show", rows.length === 0);
}

async function refreshHistory() {
  try {
    renderHistory(await apiGet("/api/history"));
  } catch (err) {
    // Telemetry polling must never interrupt the main workflow.
  }
}

// ------------------------------------------------------------ existing flow

function renderQuickAction(state) {
  const box = el("quick-action");
  const pending = state.pending_quick_action;
  if (!pending) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  el("qa-text").textContent = "Quick action awaiting confirmation: " + pending;
}

function renderState(state) {
  el("clock").textContent = Number(state.now).toFixed(2);
  el("current-mode").textContent = state.current_mode || "(none)";
  el("preferred-mode-view").textContent = state.preferred_mode || "(none)";
  const decision = state.decision;
  const vstate = decision ? decision.verification_state : "idle";
  const badge = el("state-badge");
  badge.textContent = vstate;
  badge.className = "state-badge state-" + vstate;

  if (decision) {
    el("result-decision").textContent = decision.verified
      ? "Verified: " + (decision.audio_mode || "no mode change")
      : (decision.verification_state || "idle");
  } else {
    el("result-decision").textContent = "—";
  }

  const out = el("decision");
  const reasons = [];
  const log = [];
  if (!decision) {
    out.textContent = "No decision yet. Feed a detection to run the supervisor.";
  } else {
    const lines = [];
    const dclass = decision.detected_class || "(none)";
    lines.push("detected: " + dclass + (decision.detected_class
      ? "  conf " + (decision.detected_confidence * 100).toFixed(1) + "%" : ""));
    lines.push("trigger enabled: " + decision.trigger_enabled +
      "   verified: " + decision.verified);
    lines.push("state: " + decision.verification_state +
      "   mode: " + (decision.audio_mode || "(none)") +
      "   preferred: " + (decision.preferred_mode || "(none)"));
    if (decision.release_timer_active) {
      lines.push("release timer: " +
        Number(decision.release_remaining_seconds || 0).toFixed(1) + "s remaining");
    }
    if (decision.cooldown_active) lines.push("cooldown active (duplicate alerts suppressed)");
    if (decision.restored) lines.push("preferred mode restored");
    out.textContent = lines.join("\n");
    if (decision.notification) reasons.push({ text: "Notification: " + decision.notification, cls: "notify" });
    if (decision.warning) reasons.push({ text: "Warning: " + decision.warning, cls: "warning" });
    decision.reasons.forEach((reason) => reasons.push({ text: reason, cls: "" }));
    decision.log.forEach((entry) => log.push({ text: entry, cls: "" }));
  }
  renderList(el("reasons"), reasons);
  renderList(el("log"), log);
  renderQuickAction(state);
}

async function submitInference(path, body, fileName, source) {
  const requestId = newRequestId();
  body.request_id = requestId;
  const timer = beginActivity(requestId, fileName, source);
  try {
    const res = await apiPost(path, body);
    // A newer submission owns the shared result cards; never let an older
    // response overwrite its prediction, timing, or supervisor state.
    if (requestId === latestRequestId) {
      renderPrediction(res.prediction, res.timing);
      renderState(res.state);
    }
    refreshHistory();
    return res;
  } catch (err) {
    if (requestId === latestRequestId) {
      setStatus(el("prediction"), "Error: " + err.message, true);
      el("result-chip").textContent = "FAILED";
    }
    return null;
  } finally {
    endActivity(timer, requestId);
  }
}

async function onApply() {
  setStatus(el("policy-status"), "Applying...", false);
  try {
    const res = await apiPost("/api/preferences", buildPreferencesPayload());
    renderState(res.state);
    const count = res.state.enabled_classes.length;
    el("policy-count").textContent = count + " ENABLED";
    setStatus(el("policy-status"), "Applied. " + count + " class(es) enabled.", false);
  } catch (err) {
    setStatus(el("policy-status"), err.message, true);
  }
}

async function onAnalyze() {
  const clipId = el("clip-select").value;
  if (!clipId) return;
  setStatus(el("prediction"), "Classifying… first inference may load model artifacts.", false);
  el("result-chip").textContent = "PROCESSING";
  el("top-k").innerHTML = "";
  await submitInference("/api/classify", { clip_id: clipId }, clipId, "validation_clip");
}

async function onFeed() {
  try {
    const res = await apiPost("/api/step", {});
    renderState(res.state);
  } catch (err) {
    el("decision").textContent = "Error: " + err.message;
  }
}

async function onAdvance() {
  const seconds = Number(el("advance").value);
  try {
    const res = await apiPost("/api/step", { label: null, advance_seconds: seconds });
    renderState(res.state);
  } catch (err) {
    el("decision").textContent = "Error: " + err.message;
  }
}

async function onQuickAction(action) {
  try {
    const res = await apiPost("/api/quick-action", { action });
    renderState(res.state);
    el("qa-result").textContent = res.result.note +
      " (executed: " + res.result.executed + ")";
  } catch (err) {
    el("qa-result").textContent = err.message;
  }
}

async function onUpload() {
  const input = el("upload-wav");
  const runBtn = el("upload-run");
  const file = input && input.files ? input.files[0] : null;
  if (!file) {
    setStatus(el("prediction"), "No WAV file selected.", true);
    return;
  }
  if (!file.name.toLowerCase().endsWith(".wav")) {
    setStatus(el("prediction"), "Only .wav files are allowed.", true);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    setStatus(el("prediction"), "File too large (limit 10 MB).", true);
    return;
  }
  if (runBtn) runBtn.disabled = true;
  setStatus(el("prediction"), "Uploading and classifying WAV…", false);
  el("result-chip").textContent = "PROCESSING";
  el("top-k").innerHTML = "";
  try {
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      const slice = bytes.subarray(i, Math.min(i + chunk, bytes.length));
      for (let j = 0; j < slice.length; j++) binary += String.fromCharCode(slice[j]);
    }
    await submitInference("/api/classify-upload", {
      file_name: file.name,
      data: btoa(binary),
    }, file.name, "uploaded");
  } catch (err) {
    setStatus(el("prediction"), "Error: " + (err.message || "upload failed"), true);
    el("result-chip").textContent = "FAILED";
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
}

async function init() {
  try {
    cfg = await apiGet("/api/config");
  } catch (err) {
    el("honesty").textContent = "Could not load configuration: " + err.message;
    return;
  }
  buildModeSelect(el("preferred-mode"), cfg.preferred_mode_default);
  buildPolicyTable();
  buildClipSelect();

  el("apply").addEventListener("click", onApply);
  el("analyze").addEventListener("click", onAnalyze);
  el("feed").addEventListener("click", onFeed);
  el("advance-btn").addEventListener("click", onAdvance);
  el("reset").addEventListener("click", onApply);
  el("qa-confirm").addEventListener("click", () => onQuickAction("confirm"));
  el("qa-reject").addEventListener("click", () => onQuickAction("reject"));
  el("preset-commute").addEventListener("click", presetCommuting);
  el("preset-home").addEventListener("click", presetHome);
  el("preset-clear").addEventListener("click", clearAll);
  el("upload-run").addEventListener("click", onUpload);
  el("model-retry").addEventListener("click", async () => {
    try {
      renderModel(await apiPost("/api/model/init", {}));
      refreshModel();
    } catch (err) {
      setStatus(el("model-error"), err.message, true);
    }
  });

  await onApply();
  renderActivity(null);
  refreshModel();
  refreshHistory();
  historyTimer = setInterval(refreshHistory, 4000);
}

init();
