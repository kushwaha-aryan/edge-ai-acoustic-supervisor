let cfg = null;
const policyRows = {};

const STAGE_ORDER = [
  "reception", "audio_decode", "scaler_load", "mlp_load", "metadata_load",
  "yamnet_load", "embedding", "scaling", "mlp_classification", "aggregation",
  "other_overhead",
];
const STAGE_LABELS = {
  reception: "Reception",
  audio_decode: "Audio decode",
  scaler_load: "Scaler load",
  mlp_load: "MLP load",
  metadata_load: "Metadata load",
  yamnet_load: "YAMNet load",
  embedding: "Embedding",
  scaling: "Scaling",
  mlp_classification: "MLP classification",
  aggregation: "Aggregation + supervisor",
  other_overhead: "Other overhead",
};
const PHASE_LABELS = {
  received: "Receiving audio file",
  reception: "Receiving audio file",
  audio_decode: "Decoding audio",
  embedding: "Extracting YAMNet embeddings",
  scaling: "Scaling extracted features",
  mlp_classification: "Running MLP classification",
  aggregation: "Aggregating predictions",
  supervisor: "Evaluating supervisor policy",
  completed: "Prediction completed",
  failed: "Prediction failed",
};
let modelTimer = null;
let historyTimer = null;
let latestRequestId = null;

function el(id) {
  return document.getElementById(id);
}

async function apiGet(path) {
  const res = await fetch(path, { cache: "no-store" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || res.statusText);
  }
  return data;
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || res.statusText);
  }
  return data;
}

function setStatus(node, text, isError) {
  if (!node) return;
  node.textContent = text || "";
  try { node.classList.toggle("error", Boolean(isError)); } catch (e) {}
}

function fmtMs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n < 1000) return n.toFixed(n < 10 ? 2 : 1) + " ms";
  return (n / 1000).toFixed(2) + " s";
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
  if (defaultMode) {
    select.value = defaultMode;
  }
}

function numberCell(className, value, step, min, max) {
  const td = document.createElement("td");
  const input = document.createElement("input");
  input.type = "number";
  input.className = className;
  input.value = value;
  input.step = step;
  input.min = min;
  if (max !== undefined) {
    input.max = max;
  }
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
    tr.appendChild(conf.td);
    const confirm = numberCell("p-confirm", defaults.confirm_windows, 1, 1);
    tr.appendChild(confirm.td);
    const release = numberCell("p-release", defaults.release_seconds, 0.5, 0);
    tr.appendChild(release.td);
    const cool = numberCell("p-cool", defaults.cooldown_seconds, 0.5, 0);
    tr.appendChild(cool.td);
    const priority = checkboxCell("p-priority", false);
    tr.appendChild(priority.td);
    const critical = checkboxCell("p-critical", false);
    tr.appendChild(critical.td);

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
      enabled: on.input,
      target,
      conf: conf.input,
      confirm: confirm.input,
      release: release.input,
      cool: cool.input,
      priority: priority.input,
      critical: critical.input,
      qalabel,
      notify: notify.input,
    };
    tbody.appendChild(tr);
  });
}

function buildClipSelect() {
  const select = el("clip-select");
  select.innerHTML = "";
  const groups = {};
  cfg.clips.forEach((clip) => {
    if (!groups[clip.label]) {
      groups[clip.label] = [];
    }
    groups[clip.label].push(clip);
  });
  cfg.classes.forEach((name) => {
    const clips = groups[name];
    if (!clips) {
      return;
    }
    const optgroup = document.createElement("optgroup");
    optgroup.label = name;
    clips.forEach((clip) => {
      const option = document.createElement("option");
      option.value = clip.clip_id;
      const dur = clip.duration_seconds != null
        ? " (" + clip.duration_seconds.toFixed(1) + "s)"
        : "";
      option.textContent = clip.source_dataset + " / " + clip.file_name + dur;
      optgroup.appendChild(option);
    });
    select.appendChild(optgroup);
  });
}

function setRow(name, spec) {
  const row = policyRows[name];
  if (!row) {
    return;
  }
  row.enabled.checked = true;
  if (spec.target !== undefined) {
    row.target.value = spec.target || "";
  }
  if (spec.conf !== undefined) {
    row.conf.value = spec.conf;
  }
  if (spec.confirm !== undefined) {
    row.confirm.value = spec.confirm;
  }
  if (spec.priority) {
    row.priority.checked = true;
  }
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
    critical: true, qalabel: "Call emergency contact"
  });
  setRow("Gunshot", {
    target: modeNamed("safety"), conf: 0.8, confirm: 1, priority: true,
    critical: true, qalabel: "Call emergency contact"
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
    if (item.cls) {
      li.className = item.cls;
    }
    node.appendChild(li);
  });
}

function renderPrediction(prediction) {
  const pct = (prediction.confidence * 100).toFixed(1);
  const lines = ["Predicted: " + prediction.label + "  (" + pct + "%)"];
  if (prediction.clip) {
    lines.push("Clip: " + prediction.clip.source_dataset + " / " + prediction.clip.file_name);
    lines.push("Dataset label for this clip: " + prediction.clip.label);
  } else {
    if (prediction.file_name) {
      lines.push("File: " + prediction.file_name);
    }
    if (prediction.duration_seconds != null) {
      lines.push("Duration: " + Number(prediction.duration_seconds).toFixed(1) + "s");
    }
    lines.push("Source: uploaded WAV");
  }
  el("prediction").textContent = lines.join("\n");
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
    bar.appendChild(label);
    bar.appendChild(track);
    bar.appendChild(value);
    bars.appendChild(bar);
  });
}

// ------------------------------------------------------------- monitor board

function renderModel(status) {
  const state = status.status || "not_started";
  const labels = {
    not_started: "checking",
    initializing: "loading",
    ready: "ready",
    failed: "failed",
  };
  el("monitor-model-status").textContent = labels[state] || state;
  el("monitor-init").textContent =
    state === "ready" || state === "failed" ? fmtMs(status.total_ms) : "—";
  setStatus(
    el("model-error"),
    state === "failed" ? (status.error || "Model initialization failed") : "",
    state === "failed"
  );
  el("model-retry").classList.toggle("hidden", state !== "failed");

  const list = el("monitor-init-stages");
  list.innerHTML = "";
  (status.stages || []).forEach((row) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "event-label";
    label.textContent = row.label || row.key;
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = row.state === "pending" ? "—" : fmtMs(row.duration_ms);
    li.appendChild(label);
    li.appendChild(time);
    list.appendChild(li);
  });
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
    if (modelTimer) {
      clearInterval(modelTimer);
      modelTimer = null;
    }
    setStatus(el("model-error"), "Could not read model status: " + err.message, true);
  }
}

function renderActivity(track) {
  const status = track ? track.status : "idle";
  el("monitor-status").textContent = status;
  el("monitor-phase").textContent = track
    ? (track.phase_label || PHASE_LABELS[track.phase] || track.phase)
    : "Waiting for audio";
  el("monitor-elapsed").textContent = !track ? "0 ms" : (
    status === "running" ? fmtMs(track.elapsed_ms) :
      status === "failed" ? "error" : "done"
  );

  const list = el("monitor-events");
  list.innerHTML = "";
  const events = track && track.events ? track.events : [];
  if (!events.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No processing events yet.";
    list.appendChild(li);
    return;
  }
  events.forEach((event) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "event-label";
    label.textContent = event.label || event.stage;
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = "+" + fmtMs(event.at_ms);
    li.appendChild(label);
    li.appendChild(time);
    list.appendChild(li);
  });
}

async function pollActivity(requestId) {
  try {
    const data = await apiGet(
      "/api/activity?request_id=" + encodeURIComponent(requestId)
    );
    if (requestId !== latestRequestId) return;
    renderActivity(data.track || data.current);
  } catch (err) {
    // Polling is observational; it must never interrupt inference.
  }
}

function beginActivity(requestId, fileName, source) {
  latestRequestId = requestId;
  renderActivity({
    request_id: requestId,
    file_name: fileName,
    source: source,
    phase: "received",
    phase_label: PHASE_LABELS.received,
    status: "running",
    elapsed_ms: 0,
    events: [{ stage: "received", label: PHASE_LABELS.received, at_ms: 0 }],
  });
  return setInterval(() => pollActivity(requestId), 130);
}

function endActivity(timer, requestId) {
  clearInterval(timer);
  pollActivity(requestId);
}

function renderTiming(timing, prediction) {
  const file = prediction.file_name ||
    (prediction.clip ? prediction.clip.file_name : "") || "—";
  el("monitor-file").textContent = file;
  el("monitor-total").textContent = timing ? fmtMs(timing.total_ms) : "—";

  const box = el("monitor-stages");
  box.innerHTML = "";
  if (!timing || !timing.stages_ms) return;
  STAGE_ORDER.forEach((key) => {
    if (!(key in timing.stages_ms)) return;
    const row = document.createElement("div");
    row.className = "monitor-stage";
    const label = document.createElement("span");
    label.textContent = STAGE_LABELS[key] || key;
    const value = document.createElement("strong");
    value.textContent = fmtMs(timing.stages_ms[key]);
    row.appendChild(label);
    row.appendChild(value);
    box.appendChild(row);
  });
}

function renderHistory(data) {
  const s = data.stats || {};
  el("monitor-count").textContent = String(s.count || 0);
  el("monitor-avg").textContent = s.count ? fmtMs(s.avg_ms) : "—";
  el("monitor-range").textContent = s.count
    ? fmtMs(s.min_ms) + " / " + fmtMs(s.max_ms) : "—";

  const rows = data.history || [];
  const tbody = el("history-table").querySelector("tbody");
  tbody.innerHTML = "";
  rows.slice(-8).reverse().forEach((row) => {
    const tr = document.createElement("tr");
    let time = "—";
    try {
      time = new Date(row.timestamp).toLocaleTimeString();
    } catch (e) {}
    [time, row.label || "—", fmtMs(row.total_ms)].forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  el("history-empty").classList.toggle("hidden", rows.length > 0);
}

async function refreshHistory() {
  try {
    renderHistory(await apiGet("/api/history"));
  } catch (err) {
    // Session telemetry polling must never interrupt the main workflow.
  }
}

async function submitInference(path, body, fileName, source) {
  const requestId = newRequestId();
  body.request_id = requestId;
  const timer = beginActivity(requestId, fileName, source);
  try {
    const res = await apiPost(path, body);
    // A newer request owns the shared readouts; never overwrite them with an
    // older response's prediction or timing.
    if (requestId === latestRequestId) {
      renderPrediction(res.prediction);
      renderState(res.state);
      renderTiming(res.timing, res.prediction);
    }
    refreshHistory();
    return res;
  } catch (err) {
    if (requestId === latestRequestId) {
      el("prediction").textContent = "Error: " + err.message;
    }
    return null;
  } finally {
    endActivity(timer, requestId);
  }
}

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

  const out = el("decision");
  const reasons = [];
  const log = [];
  if (!decision) {
    out.textContent = "No decision yet. Feed a detection to run the supervisor.";
  } else {
    const lines = [];
    const dclass = decision.detected_class || "(none)";
    lines.push("detected: " + dclass + (decision.detected_class
      ? "  conf " + (decision.detected_confidence * 100).toFixed(1) + "%"
      : ""));
    lines.push("trigger enabled: " + decision.trigger_enabled +
      "   verified: " + decision.verified);
    lines.push("state: " + decision.verification_state +
      "   mode: " + (decision.audio_mode || "(none)") +
      "   preferred: " + (decision.preferred_mode || "(none)"));
    if (decision.release_timer_active) {
      lines.push("release timer: " +
        Number(decision.release_remaining_seconds || 0).toFixed(1) + "s remaining");
    }
    if (decision.cooldown_active) {
      lines.push("cooldown active (duplicate alerts suppressed)");
    }
    if (decision.restored) {
      lines.push("preferred mode restored");
    }
    out.textContent = lines.join("\n");
    if (decision.notification) {
      reasons.push({ text: "Notification: " + decision.notification, cls: "notify" });
    }
    if (decision.warning) {
      reasons.push({ text: "Warning: " + decision.warning, cls: "warning" });
    }
    decision.reasons.forEach((reason) => reasons.push({ text: reason, cls: "" }));
    decision.log.forEach((entry) => log.push({ text: entry, cls: "" }));
  }
  renderList(el("reasons"), reasons);
  renderList(el("log"), log);
  renderQuickAction(state);
}

async function onApply() {
  setStatus(el("policy-status"), "Applying...", false);
  try {
    const res = await apiPost("/api/preferences", buildPreferencesPayload());
    renderState(res.state);
    setStatus(el("policy-status"),
      "Applied. " + res.state.enabled_classes.length + " class(es) enabled.", false);
  } catch (err) {
    setStatus(el("policy-status"), err.message, true);
  }
}

async function onAnalyze() {
  const clipId = el("clip-select").value;
  if (!clipId) {
    return;
  }
  el("prediction").textContent = "Classifying... (the first run loads YAMNet; this can take a few seconds)";
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
      await apiPost("/api/model/init", {});
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

async function onUpload() {
  const input = el('upload-wav');
  const runBtn = el('upload-run');
  const file = input && input.files ? input.files[0] : null;
  if (!file) {
    el('prediction').textContent = 'No WAV file selected.';
    return;
  }
  if (!file.name.toLowerCase().endsWith('.wav')) {
    el('prediction').textContent = 'Only .wav files are allowed.';
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    el('prediction').textContent = 'File too large (limit 10 MB).';
    return;
  }
  if (runBtn) runBtn.disabled = true;
  el('prediction').textContent = 'Classifying uploaded WAV... (the first run loads YAMNet; this can take a few seconds)';
  el('top-k').innerHTML = '';
  try {
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      const slice = bytes.subarray(i, Math.min(i + chunk, bytes.length));
      for (let j = 0; j < slice.length; j++) binary += String.fromCharCode(slice[j]);
    }
    const b64 = btoa(binary);
    await submitInference("/api/classify-upload", {
      file_name: file.name,
      data: b64,
    }, file.name, "uploaded");
  } catch (e) {
    el('prediction').textContent = 'Error: ' + (e.message || 'upload failed');
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
}
