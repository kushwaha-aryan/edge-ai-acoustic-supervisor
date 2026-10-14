let cfg = null;
const policyRows = {};

function el(id) {
  return document.getElementById(id);
}

async function apiGet(path) {
  const res = await fetch(path);
  const data = await res.json();
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
  const data = await res.json();
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
  try {
    const res = await apiPost("/api/classify", { clip_id: clipId });
    renderPrediction(res.prediction);
    renderState(res.state);
  } catch (err) {
    el("prediction").textContent = "Error: " + err.message;
  }
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

  await onApply();
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
    const res = await fetch('/api/classify-upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_name: file.name, data: b64 }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || 'Upload classification failed');
    }
    renderPrediction(data.prediction);
    renderState(data.state);
  } catch (e) {
    console.error(e);
    el('prediction').textContent = 'Error: ' + (e.message || 'upload failed');
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
}
