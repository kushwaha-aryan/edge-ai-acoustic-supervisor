# Edge AI Acoustic Supervisor

A lightweight, **locally-running** Edge AI system that samples ambient audio from a standard microphone, converts
short windows into 2D mel-spectrogram grids, and runs a compact Convolutional Neural Network (CNN) to **detect specific
environmental sound events**. The classifier is only the first stage: the system is specified as four stages —
**Perception → Context/Policy → Decision → Response** — ending in a **Context Response Engine** that turns a verified
trigger into coordinated actions (audio-mode change, notification, warning, optional user-confirmed quick action, and
state restoration).

The AI acts as a **high-level supervisor / decision layer** that sits on top of existing ANC hardware. It classifies
macro-windows (~200 ms) to detect context changes, while the actual audio path is handled by low-level scripts. It does
**not** perform real-time wave inversion.

First and foremost this is an **audio-classification-based acoustic supervision system**: it analyzes environmental
sound, classifies it, and uses the prediction to drive configurable supervisory decisions. It runs entirely in software
on a laptop. Adaptive noise cancellation on earbuds is a **potential future application** of this supervisor, not the
current deliverable: the present implementation performs no live-microphone capture and controls no ANC hardware.

---

## Quick start (Windows)

Prerequisites: Python 3.11+ and the dependencies in `requirements.txt`. A virtual environment `.venv/` is expected at
the repository root.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Launch the local interface (a Python standard-library HTTP server bound to `127.0.0.1`):

```powershell
.venv\Scripts\python scripts\run_supervisor_app.py
```

Open the printed URL (default `http://127.0.0.1:8000`) in a browser and stop the server with Ctrl+C. The first
classification loads the YAMNet + MLP model and can take a few seconds.

## Using the interface

The single-page interface has three panels.

**1. Trigger policy.** Enable the classes you want the supervisor to act on, and set per-class target mode, confidence
threshold, confirmation windows, release/cooldown timers, priority/critical flags, and an optional quick-action label.
Presets ("Commuting", "Home and Family") fill common configurations. Click **Apply preferences** to activate a profile;
the supervisor session resets to that policy.

**2. Audio clip, prediction, and upload.** There are two ways to get audio in:

- **Bundled validation clips** - pick one from the **Clip** dropdown (drawn from the locked validation split) and click
  **Analyze clip**.
- **Upload a WAV** - choose a local `.wav` file (max 10 MB, up to 60 seconds) and click **Run uploaded WAV**.

Both paths run the **same production pipeline**: audio -> the existing preprocessing -> **YAMNet + MLP (`mlpY01`)** ->
class probabilities -> the Phase 5 supervisor. Uploaded bytes are validated (WAV header, size, and duration when it can
be read) and written to a temporary file that is always deleted, so nothing is persisted. Non-WAV, oversized,
over-length, or corrupt files are rejected with a readable error.

**3. Supervisor decision.** After a classification the interface shows the **filename** (upload path only), the
**predicted class** and **confidence**, the Top-K distribution, and the **supervisor decision**: detected class, whether
the class is an enabled trigger, verification state, current vs. preferred audio mode, notification/warning, any
presented quick action, release-timer/cooldown state, reasoning, and the event log. "Feed detection", "Advance time",
and the quick-action confirm/dismiss buttons drive the temporal verification rules.

### Current limitations

- **Software simulation only** - audio-mode changes (Max ANC / Transparency / Safety / Standby) are software state; no
  earbud or noise-cancellation hardware is controlled.
- **No validated live-microphone inference** - there is no live-capture path; inputs are the bundled validation clips
  or a user-uploaded WAV file.
- **No physical earbud / ANC hardware integration** - earbuds and adaptive noise cancellation are a potential future
  application, not part of this implementation.
- **Model quality is reported on the validation split only.** The held-out **test** split has not been evaluated, so no
  test-set accuracy or generalization claim is made here.
- The quick action is presentation-only and never executes automatically.

---

## 1. Problem

Active Noise Cancellation (ANC) improves listening comfort in steady background noise — but it can also **reduce
awareness of important environmental sounds**. A user on a busy street may need to hear an approaching emergency siren
or a vehicle horn; a parent may need to hear a doorbell or a child crying. Because ANC reduces everything uniformly,
the user has to remove the earbud or disable ANC manually — which is exactly the moment awareness matters most.

The core problem is therefore not "remove noise better". It is: **decide, locally and quickly, when the audio scene
contains a sound the user actually cares about right now.**

## 2. Proposed Solution

A lightweight local acoustic-event detection + response system that allows users to **select which supported
environmental sounds should cause a response — and what that response should be**.

Four stages are kept deliberately separate:

| Stage | Question it answers |
|---|---|
| **Perception — detection** | "What sound is currently present?" |
| **Context / policy** | "Does it matter to this user right now?" (which classes are enabled as triggers) |
| **Decision — verification** | "Is the evidence strong and stable enough to act?" (confidence + temporal rules) |
| **Response — Context Response Engine** | "What should the system do?" (audio mode, notification, warning, quick action, restoration) |

This separation is what makes the system practical: the model does the hard audio work once, the user decides what
matters, verification prevents flicker, and the response engine handles the actions — all instantly, without
retraining anything.

## 3. Core Architecture

Overall flow: **Ambient Audio → Detection → User Trigger Policy → Confidence + Temporal Verification → Context
Response Engine → Response Actions → State Restoration.**

```
Ambient microphone
   ↓  [PERCEPTION]
Audio preprocessing (raw PCM → mel-spectrogram, ~200 ms windows)
   ↓
Lightweight acoustic event classifier
   ↓
Probability / confidence per supported sound class
   ↓  [CONTEXT / POLICY]
Check user-selected trigger profile (which classes are enabled)
   ↓  [DECISION]
Temporal / event verification (sustained evidence? threshold met?)
   ↓
IF verified:
   ↓  [RESPONSE]
Context Response Engine
   ├─→ Audio mode (Max ANC / Transparency / Safety / Standby)
   ├─→ Notification ("Siren detected — Transparency mode enabled")
   ├─→ Warning / priority alert (configured high-priority events)
   ├─→ User-confirmed quick action (configured critical events — user must confirm)
   └─→ State management: release timer, cooldown, restore preferred mode
   ↓
IF detected but NOT an enabled trigger → no response; preferred mode unchanged
```

The AI sits on top of existing ANC hardware and outputs a **decision signal**. On real ANC hardware this decision would
be sent as a command to that hardware's own ANC system; in this project the mode change is simulated at software level.

**ML architecture (on-device, fixed):** one lightweight CNN, **<100K parameters** target, 16 kHz input, 200 ms windows
with 50% overlap, 64 mel bands, FFT 1024, hop 512, CPU/local inference. Large pretrained audio transformers (e.g.,
AST at 86M+ params) are **not** the on-device model. **Final model (decision D14, 2026-10-14):** the project's
official final / best-performing model is **YAMNet + MLP** (validation clip accuracy **0.8109**, macro-F1 **0.7719**),
which supersedes the YAMNet + logistic path (D13) that is kept unchanged as an archived reference; the locked CNN
remains the on-device architecture (YAMNet exceeds the <100K edge budget).

## 4. Context Response Engine

The fourth stage: pure software logic that consumes a **verified trigger** and produces coordinated response
actions. The CNN never decides what to do — only what is present.

| # | Response type | Description |
|---|---|---|
| **A** | Audio-mode response | Switch to the policy-mapped mode (Max ANC / Transparency / Safety / Standby / none) |
| **B** | Notification | Human-readable message accompanying the response (e.g., "Siren detected — Transparency mode enabled") |
| **C** | Warning / priority alert | Elevated-urgency alert for configured high-priority events (e.g., smoke alarm) |
| **D** | User-confirmed quick action | For configured **critical** events: surfaces an action shortcut (e.g., "call emergency services") that runs **only after explicit user confirmation** |
| **E** | State management | Release timer → restore preferred mode after the trigger disappears; cooldown / duplicate-alert suppression; response log |

**Safety boundary (normative):** the quick action **never executes automatically** — the system may only *present*
it, and the user must explicitly confirm. It is disabled by default, applies only to user-configured critical
events, and is a UI-shortcut prototype: **no claim of guaranteed emergency detection**, and "automatic emergency
calling" is out of scope.

**Supervisor interface (demo):** the demo makes the pipeline visible — detected event + confidence, active profile,
trigger enabled/disabled, verification state, current audio mode, notification/warning state, available
user-confirmed quick action, and state restoration.

Demo pipeline: Speaker/phone → laptop mic → raw PCM → preprocessing → CNN → event probability → user policy →
temporal verification → Context Response Engine → supervisor interface.

## 5. Example Trigger Classes

Candidate classes include:

| Trigger class | Typical relevance |
|---|---|
| Emergency siren | Road / emergency-vehicle awareness |
| Vehicle horn | Traffic awareness |
| Doorbell | Home / delivery awareness |
| Baby or child crying | Home / family awareness |
| Smoke alarm | Home safety awareness |

**The supported class list was ratified in Phase 1 (2026-10-08, decision D8).** It is the alphabetical 17-class
taxonomy in `TRIGGER_CLASS_NAMES` (`src/config.py`), finalized against dataset availability, audio quality,
sufficient sample counts, class balance, relevance to earbud situational awareness, and the adequacy of evaluation.
The 18th candidate, `Smoke_Fire_Alarm`, remains **unresolved** (0 conservative public clips were found; W2).

The system does **not** claim to detect every possible environmental sound. Users enable only the supported acoustic
events they want as triggers.

## 6. User Customization

Users can create or select **profiles** — simple policy configurations that enable the trigger classes relevant to
their current situation, and configure how the Context Response Engine responds to them:

| Example profile | Enabled triggers (illustrative) |
|---|---|
| **Commuting** | Siren, Vehicle horn |
| **Travel** | Siren, Vehicle horn, station/announcement-related sounds *(only if a suitable dataset and class can be supported)* |
| **Home / Family** | Doorbell, Baby/child crying, Smoke alarm |
| **Custom** | Any user-defined subset of the supported classes |

Example behaviour (probabilities are illustrative examples, **not measured results**):

```
[Perception]     Model output:   {siren: 0.94, horn: 0.03, doorbell: 0.01, crying: 0.02}
[Context/Policy] User profile:   {siren: enabled, horn: enabled, doorbell: disabled, crying: disabled}
[Decision]       Verification:   threshold met, sustained across windows → VERIFIED
[Response]       Result:         TRIGGER — audio mode switches, notification shown,
                                 release timer armed, preferred mode restored later
```

If the model correctly detects a **doorbell** under that same profile, **no response of any kind occurs** (no mode
change, no notification), because the user did not enable the doorbell as a trigger — a required demo case.

**Architectural principle: profiles are policy configurations, not separate ML models.** One model is trained on the
supported acoustic event classes; switching or editing a profile is an instant configuration change. **No model is
retrained when the user changes their trigger profile.**

## 7. Edge-AI Motivation

Local inference is intended to **reduce dependence on cloud processing** and to provide **low-latency responses**:

- All compute runs locally on the device where audio is captured — **offline and private**, no network round-trip.
- Response latency is bounded by the inference cadence (~200–300 ms), which matters when the user may be missing an
  approaching vehicle or an alarm.
- A **lightweight model** and **short audio windows** keep the system viable on consumer hardware (and potentially
  portable to embedded targets later).
- Audio never needs to leave the device.

## 8. Project Status

**Current state (2026-10-12):** dataset acquisition, validation, preprocessing, and the Phase 3A model training +
validation analysis are EXECUTED, including the controlled experiments (preset A, preset D and preset E with
`--pool-freq-only`, clip-level evaluation), the YAMNet frozen-embedding baseline (fit + report; now the official
final / best model, decision D14), and two isolated architecture-comparison experiments (per-class binary CNNs with
a logistic head and with an MLP
head; experimental only). The **Phase 5 decision layer** (Context Response Engine) is implemented as pure software
with unit tests (§8 item 4), and a **local interactive interface** now wires that decision layer and the real
inference path together so the whole policy → decision → response flow is visible and operable (§8 item 5). All
Phase 3B **test-split** evaluation, longer-hyperparameter training, live-capture (Phase 4) actuation, and the Phase 6
demo remain **gated on explicit user authorization**.

In phase order, what is actually in place:

1. **Phase 1 - Dataset Investigation, Trigger-Class Definition & Class Mapping** (`data/`, `src/data/`): 10,565
   curated clips across 17 classes acquired (UrbanSound8K, FSD50K, ESC-50, owlgebra-ai/babycry), mappings fixed and
   provenance manifest built; full read-only validation verdict **PASS WITH WARNINGS**; `Smoke_Fire_Alarm`
   unresolved (0 conservative public clips found).
2. **Phase 2 - Signal Processing Pipeline** (`src/preprocessing/`): EXECUTED - 10,281 kept clips (284 excluded per
   audit flags), 15 s truncation, mono 16 kHz, 200 ms / 50%-overlap mel windows (64,5,1) scaled with train-frozen
   dB bounds, source-group-aware covariate-balanced 70/15/15 split (train 7,021 / val 1,618 / test 1,642), all D11
   leakage gates PASS; Phase 2B distribution gate **PASS WITH WARNINGS** (warnings documented, not blocking).
3. **Phase 3 - Model Design, Training & Evaluation** (`src/models/`, `src/training/`): locked 56,657-param CNN
   (Conv2D 16/32/64 → Dense 32 → Dropout 0.3 → softmax 17) trained and analysed on the locked 74,830-window
   **validation** split (test split never opened). Documented models are **not** conflated:
   - `data/models/baseline_A.keras` (preset A, locked CNN): val window accuracy **0.4243**, macro-F1 **0.4048**
     (74,830 windows); clip mean-softmax 0.4994 / 0.4689 / 0.5193.
   - `data/models/lightweight_cnn.keras` — byte-identical to `archive/weighted_epoch1.keras` (epoch-1 class-weighted
     checkpoint): val window accuracy **0.3804**, macro-F1 **0.3861**; clip mean-softmax 0.5414 / 0.5113 / 0.5321.
   Presets B/C/D/E approved; preset A remains the locked baseline. Preset D features built (train 127,398 / val
   29,363 windows); preset D and E trained with `--pool-freq-only` (25,937 params, time axis preserved).
YAMNet frozen-embedding baseline scaffolded, extracted (8,639 clips, 87,985 frames, 0 failures), and **fit**:
validation-side-by-side logistic-head 0.7769 / 0.7426 / 0.7791 vs zero-shot 0.6273 / 0.5955 / 0.6135 vs the CNN
(`baseline_A` clips) 0.4994 / 0.4689 / 0.5193. The YAMNet + MLP head (trained with seed 42; three-seed comparison
reported in the blueprint) achieved validation clip accuracy **0.8109** and macro-F1 **0.7719** (mlpY01) and was
**adopted (decision D14, 2026-10-14) as the project's official final / best-performing model**, superseding the
YAMNet + logistic path (D13) which is kept unchanged as an archived reference. The locked CNN remains the on-device
model (YAMNet exceeds the <100K edge budget); see §2.6.
   - Two **isolated architecture-comparison experiments** (experimental only, **NOT adopted**; self-contained at the
     repo root, nothing in `data/` or the locked pipeline modified, test split never opened): a per-class
     **binary-CNN** first stage — 17 binary CNNs (one sigmoid per trigger class; 56,129 params each) trained on the
     same locked preset-A windows — with the second-stage **head** varied:
     - `experiment_binary_cnn_lr/` — CNN → sklearn logistic head: window val 0.4732, **clip val 0.5902** (720-clip
       demo 417/720 = 57.9%).
     - `experiment_binary_cnn_mlp/` — CNN → tf.keras MLP head (17→64→17): window val 0.4748, **clip val 0.5828**
       (720-clip demo 419/720 = 58.2%).
     Both are far below the YAMNet pipeline (0.7769) and roughly equal to each other, so swapping the head does
     **not** close the gap — the frozen-CNN feature stage is the limit. See blueprint §2.7.

4. **Phase 5 - Trigger Policy, Temporal Verification & Context Response Engine** (`src/routing/`): decision layer
   IMPLEMENTED as pure software (no ML, no retraining, perception stage never imported). `preferences.py` defines the
   per-class `TriggerPolicy` / `UserPreferences` (validated against the 17-class taxonomy and the audio-mode set);
   `supervisor.py` consumes a prediction (`.label` / `.confidence`) plus preferences and returns an explicit
   `SupervisorDecision` implementing response types A–E (audio-mode response incl. the no-action case, notification,
   high-priority warning, user-confirmed quick action, state management: release timer + preferred-mode restoration,
   cooldown/duplicate suppression, log). The §2A.3 quick-action boundary is enforced in code: `executed` is always
   `False` and nothing is dispatched. Time is explicit (`now`) so behavior is deterministic. Covered by 38 stdlib
   `unittest` tests (`tests/test_routing_preferences.py`, `tests/test_routing_supervisor.py`), run with
   `python -m unittest discover -s tests -t . -p "test_*.py"`. Audio-mode actuation and live capture (Phase 4) are
   not wired up yet; proposed defaults (`MODE_TRANSPARENCY`, `AUDIO_MODES`, `RESPONSE_COOLDOWN_SECONDS`, all-disabled
   default preference set) await user approval (blueprint §2A.6).

5. **Application interface** (`src/app/`, `scripts/run_supervisor_app.py`): a **local, single-page interface** that
   wires the real final inference path (YAMNet + MLP `mlpY01`, decision D14) and the Phase 5 decision layer into one
   visible workflow — choose a trigger policy, classify a bundled validation clip **or upload a local WAV file**, watch
   the supervisor return its decision and state, and confirm or dismiss a presented quick action. Served with the Python
   standard library only (`http.server`; no web framework, no new dependency). Clips come from the existing validation
   split (`data/splits/clip_splits.csv` val rows joined on `clip_id` with `data/provenance/phase2a_clip_ledger.csv` kept
   rows); uploaded WAVs are validated and written to a temp file that is always removed; the predictor is loaded
   lazily and only allow-listed `clip_id`s are ever classified. It is explicitly
   **SIMULATED**: no live-microphone capture, no earbud-hardware control, and confirming a quick action is
   presentation-only (`executed` is always `False`). Launch with
   `.venv/Scripts/python scripts/run_supervisor_app.py` and open the printed URL (default `http://127.0.0.1:8000`).
   Covered by `tests/test_app_service.py` and `tests/test_app_server.py` (no TensorFlow/audio/network in tests).

Splits must pass the mandatory validation gate (proportion + per-class distribution + leakage + spectrogram +
artifact checks) before any training - Phase 2A/2B gates currently PASS. Phase 4 live capture and Phase 6 not started.

Before any training, dataset splits must pass a mandatory validation gate: proportion check, per-class distribution
similarity, duplicate/leakage check, representative spectrogram comparison across splits, and detection of excessive
silence or preprocessing artifacts.

## 9. Research / Reference Basis

The concept is informed by the following publication, used as a **design reference / prior-art reference** for the
context-aware noise-cancellation adjustment problem setting:

> Shailesh Maheshwari, Vedant Maheshwari, Rayansh Maheshwari.
> *"Context-Aware Adjustment of Noise Cancellation for Wearable Audio Devices."*
> Technical Disclosure Commons, 27 October 2025.
> <https://www.tdcommons.org/cgi/viewcontent.cgi?article=10034&context=dpubs_series>

This project **does not reproduce or reimplement** that work, and does not use its dataset, trained model, or
experimental results. Any implementation here is the project's own work.

The project is also a BTech prototype, not a certified safety-critical product: it does not guarantee detection of
emergency sounds, and it does not claim to control ANC on commercial earbuds unless real hardware integration exists.

---

## Repository Layout

```
data/               Raw + processed datasets (gitignored)
  raw/              Downloaded dataset archives (validated copies)
  processed/        Cached .npy mel-spectrogram arrays (Phase 2, preset A)
  processed_preset_D/  Preset D train+val mel arrays (Phase 3A controlled experiment)
  embeddings/       YAMNet frozen-embedding cache (final-model feature source; decision D14)
  splits/           Stratified train/val/test split metadata (Phase 1)
  provenance/       Per-clip source/label/metadata records (Phase 1)
src/                All source code
  config.py         Single source of truth for paths, hyperparameters
  data/             Phase 1 - dataset investigation, filtering, class mapping, splitting
  preprocessing/    Phase 2 - audio -> mel-spectrogram conversion
  models/           Phase 3 - CNN architecture definition
  training/         Phase 3 - training loop, evaluation, metrics
  inference/        Production prediction entry point: YAMNet embeddings + MLP head (Phase 4 live capture next)
  routing/          Phase 5 - trigger policy, temporal verification, Context Response Engine (decision layer implemented)
  app/              Local interactive interface (stdlib HTTP server + single-page UI) over inference + routing
notebooks/          Jupyter notebooks for exploration & visualization
scripts/            CLI entry points to run phases end-to-end (run_supervisor_app.py launches the interface)
tests/              Unit tests (Phase 5 decision layer + application interface)
results/            Saved models, metrics, plots (gitignored)
experiment_binary_cnn_lr/   Isolated CNN -> logistic comparison experiment (experimental only; runs/ gitignored)
experiment_binary_cnn_mlp/  Isolated CNN -> MLP comparison experiment (experimental only; runs/ gitignored)
docs/               Documentation (see Source Documents below)
  original_proposal/  Archived original PDFs + STATUS.md (superseded, historical)
```

## Source Documents

- **`FINAL_PROJECT_BLUEPRINT.md`** — the project's **living specification** and single source of truth.
  Consolidates and supersedes the two PDFs; records every change with a reason.
- `docs/original_proposal/Edge_AI_Earbud_Project_Blueprint .pdf` — original architecture & viva defense guide
  (superseded; archived with `docs/original_proposal/STATUS.md`)
- `docs/original_proposal/Edge_AI_Earbud_Phase_Plan.pdf` — original phase-wise implementation roadmap
  (superseded; archived with `docs/original_proposal/STATUS.md`)

## Scope Guardrails

- Laptop-only simulation; no physical hardware / PCB build; no custom earbud hardware development.
- Software-level ANC/Transparency/Safety mode routing; no claim of controlling ANC on commercial earbuds.
- **No automatic emergency action**: the quick action is presentation-only and requires explicit user confirmation;
  no guaranteed emergency detection; not a certified safety-critical product.
- The project is a four-stage **supervisor**, not merely a sound classifier.
- No embedded / microcontroller deployment this semester (future work only).
- Training data from suitable public datasets (UrbanSound8K, ESC-50, AudioSet-derived, and other appropriate public
  sources as justified) — chosen for quality, diversity, balance, provenance and task suitability, not to hit a fixed
  count. Real-mic recording is limited to demo/live validation samples only.
- No cloud dependency as the primary path; local inference is the project goal.