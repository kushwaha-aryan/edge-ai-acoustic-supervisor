# Edge AI Acoustic Supervisor — FINAL PROJECT BLUEPRINT

**Title:** Edge AI Acoustic Supervisor — Smart Context-Aware Framework for Next-Generation Earphones

**Project type:** BTech AIML Final Year Project

**Status:** Living specification. This document is the single source of truth for the project. It consolidates and
supersedes `Edge_AI_Earbud_Project_Blueprint .pdf` and `Edge_AI_Earbud_Phase_Plan.pdf` (archived under
`docs/original_proposal/`, see §12), preserving their architecture,
goals, terminology, and constraints, and records any intentional changes made during the project.

**Revision:** Rev 3 — **Complete Context-Aware Acoustic Supervisor.** The system is specified as four conceptual
stages — **Perception → Context/Policy → Decision → Response** — with a dedicated **Context Response Engine**
coordinating audio-mode changes, notifications, priority warnings, user-confirmed quick actions, and state
restoration. (Rev 2 established the user-selectable acoustic trigger architecture; Rev 1's three broad
acoustic-class framing is retained below as historical record. All three revisions are recorded in
[§11 Change Log](#11-change-log).)

**Prior-art / design reference (not reproduced):** "Context-Aware Adjustment of Noise Cancellation for Wearable
Audio Devices" — Shailesh Maheshwari, Vedant Maheshwari, Rayansh Maheshwari, Technical Disclosure Commons,
27 October 2025. <https://www.tdcommons.org/cgi/viewcontent.cgi?article=10034&context=dpubs_series>

> **Scope of the reference:** this publication is used as a **design reference / prior-art reference** for the
> context-aware ANC-adjustment problem setting only. The project does **not** claim to reproduce, reimplement, or
> derive its dataset, trained model, or experimental results from it. Any overlap is conceptual, and any actual
> implementation is the project's own work.

> Rule: Every change to requirements, datasets, or approach MUST be recorded below under
> **[Change Log](#11-change-log)** with the reason. Never change requirements silently. This document must stay
> consistent with the actual implementation as the project evolves.
>
> **Auto-update standing rule:** Whenever any project-level change occurs or is discovered — covering source,
> dataset strategy, architecture, phases, preprocessing, model, evaluation, or requirements — this document MUST
> be updated automatically to reflect the new final decision. This is a standing instruction that applies to all
> future sessions and contexts; do not wait for the user (or any reminder) to request the update.

---

## 1. Executive Project Summary

This project builds a Smart, Context-Aware **Edge AI Controller** for next-generation earphones. Rather than
attempting to filter or manipulate continuous raw audio with a heavy ML pipeline, the system acts as an intelligent
**Acoustic Supervisor**.

The system is specified as **four deliberately separated conceptual stages** (the core architectural framing of
Rev 3):

1. **Perception (detection)** — *"What sound is currently present?"* A lightweight acoustic event classifier scores
   the supported environmental sound classes for each short window.
2. **Context / policy** — *"Does it matter to this user, right now?"* A user-configurable trigger policy decides
   which detected classes are enabled as triggers (per-class enable flags, confidence threshold, profile context).
3. **Decision (verification)** — *"Is the evidence strong and stable enough to act?"* Temporal verification requires
   sustained evidence across consecutive windows (per policy entry, so safety-relevant triggers may act on a single
   high-confidence window).
4. **Response** — *"What should the system do about it?"* A **Context Response Engine** converts a verified trigger
   into coordinated response actions: audio-mode change, notification, priority warning, optional user-confirmed
   quick action, and state management (release timer + restoration).

These four questions — detection / policy / decision / response — are distinct and must never be conflated in
documentation, demos, or code structure. The Context Response Engine is specified in
[§2A Context Response Engine](#2a-context-response-engine-supervisor-response-subsystem).

Operating entirely locally on a host device (simulated via laptop deployment), the framework:

1. Samples ambient audio via a standard, non-ANC wired microphone.
2. Processes input into compact 2D structural arrays (mel-spectrograms).
3. Executes real-time classification inferences on ~200 ms macro-windows (perception).
4. Consults the user's trigger profile (which classes are enabled as triggers) (context/policy).
5. Verifies the candidate trigger temporally (sustained evidence across consecutive windows) (decision).
6. Hands the verified trigger to the Context Response Engine, which routes the configured response actions —
   audio-mode change (Maximum ANC / Transparency / Safety), notification, warning, optional user-confirmed quick
   action — and manages state (trigger release timer → restore preferred mode; duplicate/cooldown suppression).

### 1.1 Conceptual flow (Rev 3 architecture)

Overall flow: **Ambient Audio → Acoustic Event Detection → User Trigger Policy → Confidence + Temporal
Verification → Context Response Engine → Response Actions → State Restoration.**

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
Temporal / event verification (sustained evidence? confidence threshold met?)
   ↓
IF a selected trigger is verified:
   ↓  [RESPONSE]
Context Response Engine
   ├─→ Audio-mode response (Max ANC / Transparency / Safety / Standby)
   ├─→ Notification (e.g., "Siren detected — Transparency mode enabled")
   ├─→ Warning / priority alert (configured high-priority events)
   ├─→ User-confirmed quick action (configured critical events — requires EXPLICIT confirmation)
   └─→ State management: release timer, cooldown / duplicate suppression,
       restore preferred mode when the trigger is absent for a configured period
   ↓
IF a detected sound is NOT an enabled trigger:
   → no response; preferred audio mode continues unchanged
```

### 1.2 Supported trigger classes

The system detects **specific environmental sound events that the user selects as triggers**. Candidate classes
include, but are not limited to:

| Candidate trigger class | Typical relevance |
|---|---|
| Emergency siren | Road / emergency-vehicle awareness |
| Vehicle horn | Traffic awareness |
| Doorbell | Home / delivery awareness |
| Baby or child crying | Home / family awareness |
| Smoke alarm | Home safety awareness |

> **Scope statement (must be honoured in all documentation and demos):** the system does **not** claim to detect
> every possible environmental sound. The user **selects which supported acoustic events are enabled as triggers**
> from the classes the trained model supports. The **final supported class list was ratified in Phase 1** as the
> 17-class taxonomy in §5.1 (decision D8, 2026-10-08); the 18th candidate, `Smoke_Fire_Alarm`, remains
> **UNRESOLVED** (no conservative public clips found).

### 1.3 Prior three-class framing — SUPERSEDED (historical record, retained)

> **STATUS: SUPERSEDED by the user-selectable acoustic trigger architecture (§1.1–§1.2). Retained as the
> historical record of Rev 1.** It must not be reintroduced as a requirement.

| Detected Class | Triggered Mode | Intended Behavior |
|---|---|---|
| **Class A — Continuous Low-Freq Hum** | **Maximum ANC Mode** | Signal detected → trigger Max ANC state (e.g., phase-masking/noise profile) |
| **Class B — Transient Danger Profile** | **Safety Transparency Mode** | Signal detected → trigger immediate transparency / direct mic passthrough |
| **Class C — Low-Amplitude Quiet** | **Low-Power Standby Mode** | Signal detected → scale down DSP processing / reduce sampling to conserve power |

Why it was superseded is recorded in the [Change Log](#11-change-log) entry of 2026-10-05. In short: forcing
unrelated sounds into three overly broad buckets (e.g. lumping a doorbell and an emergency siren into one
"transient danger" class) reduced interpretability, made class balance and dataset sourcing harder, and weakened the
evaluation story. The modes themselves (Maximum ANC / Transparency / Safety / Low-Power Standby) remain valid —
what changed is **what triggers them**.

> **Important clarification (signal vs. analogy):** The masking noise played during a demo in Max ANC Mode is NOT real
> noise cancellation — it is an audible stand-in so a human watching the demo can perceive that a mode-switch decision
> fired. The actual system output is a **discrete decision signal** (e.g., "siren 0.94 and siren is an enabled
> trigger → enter Transparency mode"). On real ANC hardware, this decision would be sent as a command/API call to
> that hardware's own ANC system. The project's contribution is the **classification-and-decision layer**, not audio
> cancellation itself. Do not present masking noise as the project's noise-cancellation mechanism.

### 1.4 Example behaviour (ILLUSTRATIVE — not measured results)

The following examples show how the four stages interact. The probabilities are **illustrative examples only and
are not measured model outputs** — real numbers come only from Phase 3 evaluation.

**Example 1 — enabled, verified trigger (response fires):**

```
[Perception]    Model output:      {siren: 0.94, horn: 0.03, doorbell: 0.01, crying: 0.02}
[Context/Policy] Active profile:   {siren: enabled, horn: enabled, doorbell: disabled, crying: disabled}
[Decision]      Verification:      0.94 ≥ threshold, sustained across N consecutive windows → VERIFIED
[Response]      Response Engine:   audio mode → Transparency
                                   notification → "Siren detected — Transparency mode enabled"
                                   state → release timer armed; preferred mode restored after the
                                   siren is absent for the configured period; cooldown suppresses duplicates
```

**Example 2 — correctly detected but disabled (no response):**

```
[Perception]    Model output:      {doorbell: 0.91, ...}          ← detection is correct
[Context/Policy] Active profile:   {doorbell: disabled}           ← user did not enable it
[Decision]      Verification:      not attempted (policy gate failed first)
[Response]      Response Engine:   NO action; preferred audio mode unchanged
```

A correct detection with a disabled policy entry must produce **no response at all** — this is the practical
value of the detection/policy separation, and it is a required demo case.

---

## 2. Technical & Machine Learning Architecture

The system treats 1D audio sequences as **2D spatial feature profiles**, building on foundations from Andrew Ng's ML
Specialization.

- **Input Spatialization:** Raw time-series signals are converted into Mel-Spectrograms (or MFCCs) using `librosa`.
  This maps acoustic frequency distributions over brief temporal steps (200 ms blocks).
- **Neural Network Design:** A lightweight CNN accepts the 2D feature grids. Feature-extraction layers extract
  acoustic invariants, fully-connected layers map the properties, and a final **Softmax** layer resolves
  classification across the **N supported acoustic event classes** (`N` finalized in Phase 1 — see §5.1).
- **Evaluation Framework:** Optimization tracks **Precision, Recall, and F1-score** per class across edge
  conditions, not just accuracy. **Minimizing False Negatives** for safety-relevant selected triggers is a core
  boundary constraint.
- **No invented targets:** architecture depth/width, class count, and per-class performance targets are finalized
  only after Phase 1 establishes the class set and the dataset. Only the *edge constraints* (short windows, low
  latency, lightweight model, local processing) are fixed in advance.

### 2.1 Four-stage separation (core architectural principle)

The project distinguishes **four** stages. Each answers a different question and lives in a different layer.
Never collapse these into a single "classifier decides" story — in particular, **response is not detection, and
decision is not policy**.

| Stage | Question answered | Responsibility | Changes when the user edits settings? |
|---|---|---|---|
| **Perception — detection** | "What sound is currently present?" | Single lightweight CNN over supported acoustic event classes | No — model is trained once |
| **Context / policy** | "Does it matter to this user?" | Per-class enable flags, active profile (Commuting / Home-Family / Custom), confidence threshold | Yes — instantly, at runtime |
| **Decision — verification** | "Is the evidence strong and stable enough to act?" | Temporal verification across consecutive windows (per-entry rules; single-window allowed for safety-relevant triggers) | Partially — per-entry persistence rules are configuration; the algorithm itself is fixed |
| **Response — Context Response Engine** | "What should the system do?" | Audio-mode change, notification, warning, optional user-confirmed quick action, release timer + restoration, cooldown | Yes — response mapping is configuration |

Example (stages annotated):

```
[Perception]     Model output:      {siren: 0.94, horn: 0.03, doorbell: 0.01, crying: 0.02}
[Context/Policy] User profile:      {siren: enabled, horn: enabled, doorbell: disabled, crying: disabled}
[Decision]       Verification:      sustained evidence + confidence satisfied → VERIFIED
[Response]       System behaviour:  TRIGGER → Context Response Engine executes the configured response
```

If the model detects a **doorbell** under that same profile, the detection may be perfectly correct, but **no
response of any kind occurs** (no mode change, no notification) because the user did not enable the doorbell as a
trigger. This separation is what makes the system practical and customizable.

### 2.2 Do not retrain per profile

**A separate model must NOT be trained every time the user changes their trigger profile.** The project ships
**one model** trained on the supported acoustic event classes; a user-configurable **policy layer** decides which
detected classes cause an action. Changing a profile is a configuration change (no retraining, no re-export).

### 2.3 Trigger profiles (OPTIONAL higher-level demonstrations)

Profiles are **collections of user-selected trigger policies** built on top of the acoustic classifier. They are
**not** model classes and **not** separate models, and they are **optional demonstrations**, not mandatory
requirements.

| Example profile | Enabled triggers (illustrative) |
|---|---|
| **Commuting** | Siren, Vehicle horn |
| **Travel** | Siren, Vehicle horn, Station/announcement-related sounds — *only if a suitable dataset and class can be supported* |
| **Home / Family** | Doorbell, Baby/child crying, Smoke alarm |
| **Custom** | Any user-defined subset of supported classes (profiles are free-form configurations, not a fixed menu) |

> A profile is a **configuration policy only**: it sets which classes are triggers and how the Context Response
> Engine responds to them. Switching profiles is a runtime configuration change — **never** a model change, never
> a retraining event (§2.2).

### 2.4 Machine-Learning Gap Bridging Matrix (from Blueprint)

| Foundation Concept (from Specialization) | Project-Specific Implementation Extension |
|---|---|
| Supervised Learning & Image Inputs | 1D raw audio → `librosa.stft` / `librosa.feature.melspectrogram` → 2D visual matrices |
| Vectorization via NumPy Matrices | Multi-dimensional array ops process real-time buffer blocks; no heavy Python loops |
| Static Tensor Evaluation | Continuous, low-latency streaming pipeline via PyAudio / sounddevice |
| Multi-Class Cross-Entropy Optimization | Softmax classification across the supported environmental event classes |

### 2.5 Why a Supervisor (latency rationale — viva defense)

The model does NOT perform real-time wave inversion/phase cancellation (which would violate physical buffer-latency
limits of ADC/DAC pipelines). Instead, ML performs classification inferences every **200–300 ms**, while the audio
routing path is handled instantaneously by low-level scripts. This hybrid design keeps latencies within real-world
engineering standards.

### 2.6 ML architecture — LOCKED

The machine-learning architecture is **frozen and does not change** as documentation evolves:

| Parameter | Fixed value |
|---|---|
| Model | One lightweight **CNN** (Conv2D/Dense stack, Softmax over N classes) |
| Parameter budget | **< 100K parameters** target (edge-deployable) |
| Inference | CPU / local only, every ~200–300 ms cadence |
| Sample rate | 16 kHz |
| Window | 200 ms, 50% overlap (step 1,600 samples) |
| Features | Mel-spectrogram, 64 mel bands, FFT 1024, hop 512 |
| Class count `N` | **17**, ratified in Phase 1 (decision D8, 2026-10-08) — see §5.1 |

**Do NOT replace the CNN with large pretrained/audio-transformer architectures** (e.g., AudioSet-style AST models
at 86M+ parameters). Such models violate the edge/parameter budget. Hugging Face / pretrained audio models may be
used **only as optional external baselines** for comparison in the report — they are never the project's on-device
(deployed) model. **Exception (decision D13, 2026-10-12):** the **YAMNet + logistic** pipeline is adopted as the
project's **official final / best-performing model** for the report — validation clip accuracy **0.7769**, well above
the locked CNN (clip 0.4994) — while the locked CNN above **remains the on-device architecture** (YAMNet is ~3.7M
params / ~17 MB, far over the `<100K` edge budget, so it cannot run on the target device; the finalized edge model is
the CNN). The four-stage supervisor architecture (§2.1, §2A) is independent of the classifier internals: whatever N
and whatever profile is active, the same one model feeds the same policy → decision → response chain.

### 2.7 Explored architecture variants (experimental — NOT adopted)

Two isolated comparison experiments tested whether a **two-stage** design (per-class binary CNNs feeding one
combining head), or a different **second-stage head**, could beat the locked pipeline. Both are self-contained at
the repo root and reuse the locked Phase 2A preset-A windows (train 324,613 / val 74,830) and the same 200 ms
preprocessing; they do **not** modify `data/`, the frozen preprocessing, the split, the locked CNN, or the YAMNet
pipeline, and the test split is never opened.

| Variant | First stage (per-class CNNs) | Second stage (head) | Window val acc | Clip val acc |
|---|---|---|---|---|
| `experiment_binary_cnn_lr/` (exp01) | 17 binary CNNs, 56,129 params each, one sigmoid per class | sklearn `LogisticRegression` (lbfgs) | 0.4732 | **0.5902** |
| `experiment_binary_cnn_mlp/` (mlp01) | same 17 binary CNNs (reused, frozen) | tf.keras MLP 17 → 64 → 17 (softmax) | 0.4748 | **0.5828** |
| YAMNet + logistic (**official final / best model**, `scripts/demo_table.py`) | frozen YAMNet embeddings | LDA-prewhitened logistic | — | **0.7769** |

**Conclusion (honest):** both binary-CNN variants land at ~0.58–0.59 clip accuracy — far below the YAMNet pipeline
(0.7769) and essentially equal to each other — so the second-stage **head choice (logistic vs MLP) is not the
bottleneck**; the frozen-CNN feature stage is. Each per-class binary CNN is individually strong (per-class binary val
accuracy ≈ 0.76–0.98) yet combined clip accuracy stays low, showing the limit is in the combining / feature quality,
not the head. Neither binary-CNN variant is adopted: the locked single-CNN architecture (§2.6) remains the on-device
model, and YAMNet + logistic is adopted as the project's official final / best-performing model (§2.6 Exception,
decision D13). These variants broaden method coverage (per-class binary CNNs + a logistic head + an MLP head) but
change no other requirement.

---

## 2A. Context Response Engine (supervisor response subsystem)

> Numbered `2A` so that all existing section numbers and cross-references (§3, §5.1, §6, §7, §8, §10, §11) stay
> stable.

The **Context Response Engine** is the fourth stage of the architecture: it consumes a **verified trigger** (from
perception + policy + decision) and produces **coordinated response actions**, then manages the resulting state.
It is pure software logic — no ML inside it; the CNN never decides *what to do*, only *what is present*.

### 2A.1 Response types

| # | Response type | Description | Example |
|---|---|---|---|
| **A** | **Audio-mode response** | Switch to the mode mapped in the active policy entry (Max ANC / Transparency / Safety / Standby / none) | Siren verified → Transparency |
| **B** | **Notification** | Human-readable, non-urgent on-screen/log message accompanying a response | "Siren detected — Transparency mode enabled" |
| **C** | **Warning / priority alert** | Elevated-urgency alert for configured **high-priority** events (e.g., smoke alarm, siren), regardless of whether an audio mode changed | Persistent warning banner + log entry |
| **D** | **User-confirmed quick action** | For configured **critical** events only: surface an action shortcut that executes **only after explicit user confirmation** (e.g., a "call emergency services" shortcut the user must click/tap) | Siren verified → prompt shown → **user confirms** → action runs |
| **E** | **State management** | Release timer (restore preferred mode after trigger absent for a configured period), cooldown / duplicate-alert suppression, response log | Siren ends → configured period later → preferred mode restored, no duplicate alerts |

### 2A.2 Response E — state management details

- **Release timer:** armed when a response fires; when the trigger has been absent for the configured period, the
  preferred (user-default) audio mode is restored.
- **Cooldown / duplicate suppression:** a verified trigger that persists does not re-fire notifications/warnings
  every window; responses are edge-triggered with a configurable cooldown.
- **Restoration is part of the response, not an afterthought** — every mode change must have a defined path back
  to the preferred mode.

### 2A.3 Response D — user-confirmed quick action: HARD BOUNDARY

This is the most safety-sensitive feature; the following rules are **normative**:

1. The quick action **NEVER executes automatically.** It may only be *presented* by the system.
2. Execution requires **explicit, deliberate user confirmation** (a click/tap/keypress on the surfaced action).
   No countdown-to-auto-send, no implicit confirmation, no confirmation-on-behalf-of-user.
3. It applies **only to events the user has explicitly configured as critical** in the active profile. It is
   disabled by default.
4. Documentation and demos MUST state: this is a **UI shortcut prototype**, not a certified emergency system; the
   project makes **no claim of guaranteed emergency detection** (§8) and no claim of reliability sufficient for
   real emergency use.
5. "Automatic emergency calling" is **out of scope and forbidden** as a described capability.

### 2A.4 Notification / warning wording

- Warnings and notifications state what was detected, the active policy decision, and the response taken — they
  must never imply certainty beyond the model (e.g., prefer "Siren detected" as an event report, not "Ambulance
  approaching").
- All displayed probabilities/confidences in the UI are **live model outputs**, and any numbers shown in
  documentation are illustrative examples only (§1.4).

### 2A.5 Supervisor interface (demo-visible state)

The demo makes the supervisor's internal state **visible** in a software interface. The interface MUST show at
least:

- detected event (class) and confidence
- active profile
- trigger enabled / disabled for that event (policy gate result)
- verification state (pending / verified / released)
- current audio mode (+ preferred mode)
- notification / warning state
- available user-confirmed quick action (if configured for this event)
- state-management status (release timer running, cooldown active, restored)

**Demo pipeline:** Speaker/phone → laptop mic → raw PCM → preprocessing → CNN → event probability → user policy →
temporal verification → Context Response Engine → supervisor interface (fields above).

---

## 3. Hardware Simulation Strategy & Device Requirements

100% of this architecture runs locally as an **On-Device Offline Edge AI Node** using consumer hardware. No GPU or
external microcontroller required.

| Component | Requirement |
|---|---|
| Processing Engine | Consumer laptop CPU (Intel Core i5/i7, AMD Ryzen 5/7, or Apple Silicon M-series). No external GPU required. |
| Acoustic Sensor Node | Built-in laptop mic array or standard wired earphone/headphone inline microphone |
| Dynamic Evaluation Rig | A standalone smartphone emitting candidate trigger sounds (sirens, vehicle horns, doorbell, crying, smoke alarm) at controlled distances from the mic during live demos |

### Software Simulation Execution Matrix

The **mode is no longer bound to a class identity**. It is bound to the user's policy entry for the detected
class.

| Target Operational State | Trigger source (revised) | Python Simulation Execution Path |
|---|---|---|
| Maximum ANC | A selected trigger (or ambient-context rule) that the user mapped to Maximum ANC | Trigger a local looping phase-masking audio profile (e.g., optimized brown or pink noise) |
| Safety Transparency | A selected high-priority trigger (e.g., siren, smoke alarm) mapped to Safety Transparency | Open a direct, unbuffered microphone line loop piping live ambient feed to speakers |
| Low-Power Standby | A selected trigger mapped to standby (e.g., sustained quiet-ambient class) | Scale down processing loops and reduce microphone sampling frequency |
| **No action** | Detected class that is **not enabled** in the active profile | Preferred audio mode continues unchanged |

> The matrix above covers **response type A (audio-mode)** only. Notifications, warnings, user-confirmed quick
> actions, and state management (types B–E) are specified in [§2A](#2a-context-response-engine-supervisor-response-subsystem)
> and are also visible in the supervisor interface.

---

## 4. Acoustic Physics: Bypassing Built-in Noise Suppression (**Raw PCM Streams**)

**Problem:** Consumer devices ship with built-in Environmental Noise Cancellation (ENC) and automatic gain control. These
assume distant, low-amplitude, or non-vocal audio is "background garbage" and scrub it out. Distant shouts (50 m) or car
horns (15 m) drop in physical power per the **Inverse-Square Law**, and stock drivers often remove them — which would
blind the neural network.

**Solution:** Request raw PCM capture with a **clean input flag** (via PyAudio/sounddevice) to bypass OS-level filtering
*where the driver allows it*. This is intended to convert the microphone into a truer raw acoustic sensor. If a wave
can physically vibrate the diaphragm, it is captured — making distant horns/shouts more likely to appear as clear
frequency components for AI classification.

> **Honesty constraint:** raw-PCM "bypass" is driver- and platform-dependent and must be **verified and reported
> empirically** in Phase 4 (see §7 Phase 4). Never present it as guaranteed.

---

## 5. Dataset Architecture

### 5.1 Trigger classes — ratified list and how the class list was decided

The model is trained over a set of **specific environmental sound events**. The final class list was **ratified in
Phase 1 as the 17-class taxonomy below** (decision D8, 2026-10-08; see Change Log). It was determined against the
following criteria:

1. Availability of suitable **public datasets**
2. **Audio quality** of the available material
3. **Sufficient sample counts** to train and evaluate reliably
4. **Class balance** across the retained classes
5. **Relevance** to earbud situational awareness
6. Ability to **evaluate the model reliably** (well-defined labels, realistic negatives, measurable metrics)

Priority is given to classes for which **established public datasets already exist**. UrbanSound8K / ESC-50 are
**not assumed to be automatically sufficient** — Phase 1 investigated multiple public datasets and selected the
combination giving the strongest coverage for the chosen trigger classes.

**Ratified class list (D8, 2026-10-08):** the alphabetical 17-class taxonomy recorded in `src/config.py`
(`TRIGGER_CLASS_NAMES`): Aircraft, Alarm, Baby_Crying, Car_Engine, Dog_Bark, Doorbell, Drilling, Footsteps,
Glass_Breaking, Gunshot, Help_Shouting, Jackhammer, Knocking, Motorcycle, Siren, Train, Vehicle_Horn.
`Smoke_Fire_Alarm` (the intended 18th class) has **0 clips** and remains **candidate/UNRESOLVED** — no placeholder
row exists and no smoke-alarm claim is made (W2; dataset acquisition entry in the Change Log). `Doorbell` (53 clips)
and `Motorcycle` (238 clips) are the thinnest classes; additional sources may still be investigated per the flexible
sufficiency rule (§5.2).

### 5.2 Dataset Requirements (AUTHORITATIVE — revised 2026-10-05)

> These requirements supersede the earlier "~1,000 clips per class with exactly equal final A/B/C counts" rule
> (see §1.3 and the [Change Log](#11-change-log)). The fixed numerical target no longer applies to the revised
> architecture.

1. **Sufficiency requirement:** Each selected trigger class should have **sufficient high-quality, diverse public
   data for reliable training and evaluation**. Classes should be **reasonably balanced where practical**.
2. **Shortage handling:** If one class has insufficient data, **investigate additional appropriate public sources**
   before reducing or compromising dataset quality.
3. **No target-driven dataset inflation:** A dataset is **never added merely to reach a numerical target**. Dataset
   quality, diversity, class balance, provenance, and suitability for the actual trigger task take priority over
   hitting a fixed count.
4. **Provenance:** Keep source dataset, original label, and sample information for every accepted clip (§5.4).
5. **Padding policy:** **Avoid heavy zero-padding of short clips.** Excessive zero-padding makes clips mostly
   artificial silence and distorts class characteristics. Prefer naturally suitable clips/segments and **reject
   unsuitable clips rather than manufacturing large amounts of artificial silence**. Any necessary padding must be
   **limited and documented**.
6. **Real-mic recording:** Not a primary data source; limited to demo / live validation samples (see §8).

### 5.3 Source Datasets (candidates — investigate, do not assume)

| Dataset | Potential relevance | Notes (verify license/access before download) |
|---|---|---|
| **UrbanSound8K** | vehicle horns, sirens, engine idling, machinery | Academic. Fixed label taxonomy; inspect against the candidate trigger list. |
| **ESC-50** | broader ambient sounds: rain, wind, water, hums, quiet ambience | CC-BY-NC (check academic-use terms). |
| **AudioSet-derived resources** | YouTube-tagged "siren", "alarm", "baby crying", "horn" | Large; access/licensing and label noise must be assessed. |
| **Other public environmental-sound datasets** | any suitable/allowed source that improves coverage of a chosen trigger class | Must pass license/access check and be logged under provenance + change log. |

*License/access verification is a prerequisite before downloading or using any source. No dataset is adopted merely
to increase the sample count.*

### 5.4 Provenance & Metadata

For each accepted clip, record (persisted under `data/provenance/`):

- Source dataset & version / release
- Original label / class label in the source
- Sample / filename identifier (and grouping id, to prevent near-duplicate leakage between splits)
- Duration, sample rate, derived acoustic features (RMS/energy, etc.)
- Accepted trigger class (the final class id/name)
- Any filtering, trimming, or limited padding applied (with the parameters used)

---

## 6. Train / Validation / Test Split Validation (MANDATORY GATE)

After dataset curation, use a **70 / 15 / 15 stratified split**, stratified per trigger class.

Before ANY model training, the following MUST be verified for each class:

1. **Proportion check:** Mathematically verify train/val/test have approximately the expected 70/15/15 proportions.
2. **Distribution check:** Verify that the **actual audio characteristics/distributions** of train, validation, and
   test are reasonably similar for the same class (e.g., duration, RMS/energy, and relevant features) — not merely
   similar sample counts.
3. **Leakage check:** Check for duplicates or data leakage between splits (including near-duplicates from the same
   source recording/session).
4. **Visual check:** Generate spectrogram comparisons for the same class across Train / Validation / Test (one class
   → Train vs Val vs Test panel per supported class).
5. **Artifact check:** Detect **excessive silence** or **preprocessing artifacts** introduced by trimming, padding, or
   conversion.

These checks target: excessive silence, bad padding, preprocessing differences, and distribution mismatch.

> **Gate rule:** Do NOT train if a significant split/distribution problem is found. Resolve the issue and document it
> before attempting training.

Quantitative thresholds and the verdict rules live in `scripts/run_phase2b_split_gate.py` (D9 amendment,
Change Log 2026-10-08): distribution checks use exact two-sample KS statistics per class over duration / RMS dB /
clip feature mean with **flag** and **severe** levels (severe → FAIL; flags → PASS WITH WARNINGS), and cells with
fewer than 15 clips per side are **underpowered: reported, never blocking** (sampling noise at n≈51–60 classes such
as Doorbell cannot be balanced out of significance without over-fitting the split).

---

## 7. Phase-Wise Execution Plan

### Phase 1 — Dataset Investigation, Trigger-Class Definition & Class Mapping

**Estimate:** 10–15 hrs

- Investigate **multiple public datasets** (not only UrbanSound8K/ESC-50) and verify license/access conditions.
- Select the **final trigger class list** against the six criteria in §5.1, and record the decision + reasons here.
- Write the source-label → trigger-class mapping; discard irrelevant classes.
- Enforce dataset requirements §5.2 (sufficiency, balance where practical, provenance, limited/documented padding).
- **Hold out ~15–20 real-mic recordings per selected class** (phone-emitted, laptop-mic captured) for later live demo
  validation. These are held out and not used in any training split.
- Stratified 70/15/15 clip-level split (all windows derived from one source clip stay in the same split).
- Run the split-validation gate (§6) before proceeding.

### Phase 2 — Signal Processing Pipeline

**Estimate:** 10–12 hrs

Convert raw audio into the 2D spectrogram inputs the CNN consumes.

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 16,000 Hz | Standard; keeps compute light |
| Window size (analysis block) | 200 ms | Matches blueprint inference cadence |
| Window step / frame hop | Window step = 1,600 samples (50% overlap of the 200 ms window); STFT hop = 512 samples (50% of n_fft) | Smooths classification, avoids boundary artifacts |
| Feature type | Mel-spectrogram, 64 mel bands | Chosen over raw MFCC for CNNs |
| FFT window (n_fft) | 1024 | `librosa.stft`-adjacent |
| Normalization | Log-scale dB (`librosa.power_to_db`, ref 1.0), then scale to [0,1] with dB bounds frozen from train-split statistics | Chosen over per-clip min-max so values stay comparable across clips and splits |

Tasks:
- Write `librosa`-based conversion: raw wav → mel-spectrogram → fixed-size 2D array (pad/truncate to constant shape).
  **Note:** padding must respect §5.2 padding policy — prefer naturally suitable segments; avoid excessive
  zero-padding; any necessary padding must be limited and documented.
- Batch-convert the whole dataset; cache as `.npy` arrays to avoid recomputing each training run.
- Visualize a few spectrograms per class to sanity-check separability (e.g., a siren should look distinct from a
  doorbell chime and from a hum).

### Phase 3 — Model Design, Training & Evaluation

**Estimate:** 20–25 hrs

Train a lightweight CNN for the supported acoustic event classes and evaluate it against the false-negative
constraint for safety-relevant triggers.

**Architecture:** final layer width (`N` supported classes), filter counts, and depth are **finalized after Phase 1**,
keeping the project's fixed edge constraints: short windows, low inference latency, lightweight model, local
processing. The original Conv2D/Dense stack from the source blueprint remains the starting candidate and must stay
genuinely edge-deployable (roughly a sub-100K-parameter target).

**Training config:** Adam optimizer with learning-rate scheduling, categorical cross-entropy, batch size ~32, early
stopping on validation loss. **Class weighting** is decided after the class set is known, biasing against false
negatives for safety-relevant selected triggers.

**Evaluation plan:**
- Report accuracy (secondary); **headline metrics are per-class Precision, Recall, F1.**
- Confusion matrix — explicitly inspect the false-negative rate for safety-relevant trigger classes.
- **Do not invent performance targets before the class set and dataset are established** (Phase 1). Any numeric
  accuracy/recall target must be derived from the real dataset and reported honestly.
- Latency benchmark: measure inference time per 200 ms window on the target laptop CPU against the inference budget
  (design budget: single-digit-to-low-tens of ms per window, well inside the 200–300 ms cadence).

### Phase 4 — Live Capture & Raw PCM Pipeline

**Estimate:** 12–15 hrs

- Set up PyAudio/sounddevice stream with a raw input flag, bypassing OS-level ENC/AGC **where the driver allows**.
- Implement a rolling 200 ms buffer with the SAME hop/overlap settings as training preprocessing (mismatch is the
  most common real-world bug).
- Run the held-out real-mic recordings (Phase 1) through the live pipeline first — isolates model bugs from pipeline
  bugs.
- Validate the physics claim empirically: confirm a phone-emitted siren/horn at distance is picked up and detected;
  **document actual vs claimed detection range honestly**.

### Phase 5 — Trigger Policy, Temporal Verification & Context Response Engine

**Estimate:** 8–10 hrs

- Build the **user trigger policy layer**: a per-class enable/disable configuration, per-class confidence threshold,
  and a per-class target mode mapping (Max ANC / Transparency / Safety / Standby / none). **No retraining** when the
  policy changes.
- Implement **temporal verification**: require the candidate class to persist across consecutive windows before
  switching, to avoid flicker on noisy boundary frames. The persistence requirement is **per policy entry**, so
  safety-relevant triggers can require only a single high-confidence window.
- Build the **Context Response Engine** (§2A): response types A–E —
  - audio-mode response per §3 matrix, including the explicit **no-action** case for detected-but-not-enabled
    classes;
  - **notifications** (type B) with the §2A.4 wording rules;
  - **warning / priority alerts** (type C) for configured high-priority events;
  - **user-confirmed quick action** (type D) strictly per the §2A.3 hard boundary — presentation only, explicit
    user confirmation required, disabled by default, no automatic execution;
  - **state management** (type E): release timer + preferred-mode restoration, cooldown / duplicate-alert
    suppression, response log.
- Implement the **release timer**: after a trigger is absent for a configured period, restore the preferred audio
  mode.
- Output is a **decision signal** to be handed to existing ANC hardware (see §1.3 clarification).

### Phase 6 — Demo, Report & Deployment Polish

**Estimate:** 15–20 hrs

- Live demo script: phone plays candidate trigger sounds at set distances; laptop mic + pipeline classifies, applies
  the active user policy, verifies temporally, and drives the Context Response Engine — modes switch visibly,
  notifications/warnings appear, the release timer restores the preferred mode, and the case where a correctly
  detected sound causes **no** change because it is not an enabled trigger is demonstrated.
- Demonstrate the **supervisor interface** (§2A.5) showing the full state: detected event + confidence, active
  profile, trigger enabled/disabled, verification state, current audio mode, notification/warning state,
  available user-confirmed quick action (if configured), and state restoration.
- Demonstrate **response diversity**: at least one audio-mode response (A), one notification (B), one warning (C),
  and the user-confirmed quick action flow (D) presented-but-confirmed-only — plus restoration/cooldown (E).
- Demonstrate **profile switching** (Commuting / Home-Family / Custom) as a **policy switch**, not a model switch.
- GitHub repo: clean README, architecture diagram, honest metrics table (precision/recall/F1 per class), sample
  spectrograms, demo GIF/video.
- Final report: merge blueprint conceptual sections with actual results (real numbers only) and viva defense answers.
- **Do NOT attempt embedded/ARM Cortex-M deployment this semester** — mention only as future work.

---

## 8. Scope Guardrails & Safety / Scope Claims (Fixed Constraints)

- **No physical hardware / PCB build** — laptop simulation only.
- **No embedded / microcontroller deployment** — future work only (mention in report/viva as a portability path via
  TensorFlow Lite / NanoEdge AI Studio for ARM Cortex-M).
- **No custom audio dataset recording as primary data source** — public datasets first; real-mic recording limited
  to demo validation samples only.
- **No claim of guaranteed emergency-sound detection.** This is a BTech prototype, not a certified safety-critical
  product.
- **No claim that ANC is actually being controlled on commercial earbuds** unless real hardware integration exists.
  Today the ANC/Transparency/Safety routing is simulated at software level.
- **No universal sound coverage claim** — only the supported classes can be detected, and only user-enabled
  classes act as triggers.
- **Profiles are policy configurations, not models.** No model is retrained when a profile changes.
- **No automatic emergency action of any kind.** The user-confirmed quick action (§2A.3) is presentation-only and
  requires explicit user confirmation; "automatic emergency calling" is forbidden as a described capability. The
  quick action is a UI-shortcut prototype, disabled by default, never presented as safety-certified or reliable
  enough for real emergency use.
- **Do not frame the project as "merely sound classification."** It is a four-stage supervisor
  (perception → context/policy → decision → response) with a Context Response Engine; the classifier is one
  component of it.
- If the timeline slips, cut Phase 6 depth BEFORE cutting Phase 3 evaluation rigor — a working, honestly-evaluated
  model matters more than a polished GitHub page.

---

## 9. Acoustic Physics & "Edge AI" Viva Defense Points

Reference answers preserved from Blueprint §5:

- **Q1 (Why isn't ML noise cancellation slow?)** — ML is a high-level supervisor on 200–300 ms macro-windows; audio
  routing is done instantly by low-level scripts. Not real-time inversion.
- **Q2 (How is laptop = Edge AI?)** — All compute runs locally on the node where data is collected; offline + private,
  no network dependence. Portable to embedded later.
- **Q3 (How does a cheap wired mic capture distant danger?)** — Software noise-gates/driver filters, not the capsule, are
  the limitation. Raw PCM capture attempts to bypass them; any wave that vibrates the diaphragm can be captured. (Verify
  driver behaviour empirically; do not overclaim.)
- **Q4 (Why isn't this a basic filter project?)** — Static linear filters (Wiener/spectral subtraction) fail on
  non-stationary/transient noises. This uses a non-linear CNN acoustic signature classifier + a host behaviour change.
- **Q5 (Why user-selectable triggers instead of fixed classes?)** — Real users care about different events in different
  situations. Separating *detection* from *trigger policy* means one trained model serves commuting, travel, and
  home/family needs, and the system only acts on events the user actually cares about. It also improves
  interpretability: each class is a specific, nameable acoustic event rather than a broad bucket.
- **Q6 (Why not one model per profile?)** — Retraining per profile is impractical on-device and unnecessary. A single
  multi-class model plus a configurable policy layer gives instant customization with one training run.
- **Q7 (Isn't this just sound classification?)** — The classifier only answers *perception* ("what sound is
  present?"). The project is a four-stage **supervisor**: policy decides *whether it matters to this user*,
  temporal verification decides *whether the evidence is strong enough to act*, and the Context Response Engine
  decides *what to do* — audio-mode change, notification, warning, user-confirmed quick action — and manages state
  (restoration, cooldown). None of those stages live inside the model.

### Production path (practicality answer)

The system is a **decision layer** over existing ANC hardware (AirPods/Sony/Bose etc.). On real devices it would call
the device's existing ANC/transparency control API instead of running masking-noise/mic-passthrough scripts. Pipeline:
**classify → check user policy → verify → Context Response Engine → command existing hardware (plus notification /
warning / user-confirmed action)** — directly portable without building ANC hardware.

---

## 10. Git / GitHub Workflow — Standing Rule

GitHub serves as the project's **development history and backup**, not just a final-upload target.

1. **Checkpoint reminders.** Before moving between meaningful milestones, check whether current work should be
   committed/pushed. Remind at checkpoints such as: completing a phase; completing a major part of a phase; finishing
   a significant verified-and-working implementation; completing an important validation/checkpoint; before a major
   architectural or project-level change. Do NOT interrupt trivial edits. At a real checkpoint, say explicitly:
   *"Git checkpoint reached. The current work is stable enough to commit and push before we continue."* and suggest a
   concise commit message. Reminders are advisory only.
2. **No automatic commits/pushes.** Never run `git commit` / `git push` unless the user explicitly asks.
3. **Phase transitions.** Before starting a new major phase: (a) verify the current phase is complete per this document;
   (b) confirm implementation and documentation are consistent; (c) state whether this is a good Git checkpoint; if yes,
   recommend committing and pushing before continuing.
4. **Incomplete/incorrect earlier work.** Do not treat earlier committed work as permanently fixed. If work was
   implemented unnecessarily, a blueprint requirement was missed, an implementation differs from the spec, a better
   dataset/source/approach is justified, a validation step fails, or a decision needs to change — **stop and assess**,
   then correct the project properly (never preserve a mistake merely to keep Git history clean). Any project-level change
   is recorded in this document + Change Log. Git history reflects genuine evolution; do not rewrite history to hide
   mistakes unless explicitly requested.
5. **Checkpoint principle.** Prefer: *work → verify → document → Git checkpoint → continue* over one giant final commit.
6. **Repository cleanliness.** Before recommending a major push, verify: no generated datasets/audio committed; no
   secrets/API keys; temp files ignored; important source + docs included; `.gitignore` appropriate; repo is
   understandable to a GitHub viewer.
7. **Final project.** The repo should show the genuine development process and contain the final clean implementation,
   documentation, methodology, evaluation results, and relevant materials. Do not fabricate commits, results, metrics,
   experiments, or history.

---

## 11. Change Log

Every material change to the original Blueprint + Phase Plan is recorded here. New entries are appended at the top.

| Date | Change | Reason | Status |
|---|---|---|---|
| 2026-10-12 | **Project renamed: the official title is now "Edge AI Acoustic Supervisor" (naming/documentation only).** The human-facing project title was changed from "Edge AI Earbud Supervisor" to **Edge AI Acoustic Supervisor** in `README.md` (title heading), `FINAL_PROJECT_BLUEPRINT.md` (title heading; the `**Title:**` field now reads "Edge AI Acoustic Supervisor — Smart Context-Aware Framework for Next-Generation Earphones"), and the module docstring of `src/config.py`. All other prose is unchanged: architecture, phases, decisions, experiment results, metrics and the implementation plan are preserved verbatim. The GitHub repository is **not** renamed in this task (the remote stays `edge-ai-earbud-supervisor`), and the archived original PDF filenames (`docs/original_proposal/Edge_AI_Earbud_Project_Blueprint .pdf`, `Edge_AI_Earbud_Phase_Plan.pdf`) are left as-is because they are historical file names. No model code, trained weights, datasets, experiment results, or the inference pipeline were changed; no model was retrained and the test split was not accessed. | Naming task requested by the user: adopt "Edge AI Acoustic Supervisor" as the official project name, consistently, without altering technical content. | **Done (naming only)** |
| 2026-10-12 | **GitHub-readiness artifact/model audit completed; mlp run metadata made portable; locked-CNN training history now persisted.** Read-only audit of all tracked model artifacts (no files/history changed) found the repo already GitHub-ready: every required inference artifact is tracked (locked CNN `baseline_A.keras`, `lightweight_cnn.keras`, `presetD/E_freqpool.keras`, the 17+17 experiment CNNs, and the final `data/models/yamnet_logistic/yamnet_demo/{classifier.joblib, scaler.joblib, class_names.txt, pipeline_description.txt}`); the largest tracked file is 712 KB and tracked `*.keras`/`*.joblib` total ~26.5 MB; `.git` is 19.30 MiB (1,912 loose objects, 0 packs). **No Git LFS needed** (far under the 100 MiB/file hard limit and the 1 GiB repo guidance) and **no `.gitignore` change required** (datasets `data/raw` 8 GB, `data/processed*` 606 MB, `data/embeddings` 85 MB, `data/models/yamnet_hub` 17 MB, `experiment_binary_cnn_mlp/cache` 26 MB, `.venv/`, `.idea/`, `urbansound8k_header.bin`, `results/*` are all correctly ignored; no secrets found). Two approved follow-ups implemented: (1) **portable run metadata** - `experiment_binary_cnn_mlp/train_binary_cnn_mlp.py` gains a `_portable()` helper and now writes `cnn_source` and `z_cache.dir` repo-relative in POSIX form (absolute only if outside the repo root); the committed `experiment_binary_cnn_mlp/runs/mlp01/meta.json` was retro-edited to the same form (`experiment_binary_cnn_lr/runs/exp01`, `experiment_binary_cnn_mlp/cache`) - representation-only, since no consumer reads these fields (the demo reads `mlp.clip_val_accuracy`) and no metric/history value changed. (2) **training-history persistence** - `scripts/run_phase3a_training.py` now writes `data/models/<run-name>.history.json` (run name, epochs, parameter count, per-epoch metric series, per-epoch seconds, elapsed seconds) alongside the saved model, overwrite-guarded like the model, so future locked-CNN runs keep their per-epoch history (existing `exp01`/`mlp01` history unaffected; no retraining performed). Documentation-only + source change; no data/model-parameter/eval artifact altered; test split untouched. | Standing rule 1 (living spec must reflect the final decision): record the GitHub-readiness audit verdict and the two approved follow-ups at the time they were produced. | **Done (user-approved 2026-10-12)** |
| 2026-10-12 | **DECISION D13 - YAMNet + logistic adopted as the project's official final / best-performing model (dual framing).** User-directed: the official final model is the YAMNet + logistic pipeline (`scripts/demo_table.py`; validation clip accuracy **0.7769**, macro-F1 0.7426) - the strongest result in the project. **Scope (dual framing, per user choice):** YAMNet + logistic is the official **final / best-reported model**, while the locked single-CNN (§2.6) **remains the on-device architecture**; YAMNet (~3.7M params, ~17 MB) far exceeds the `<100K` edge budget, so it is not the on-device model. §2.6 amended with an explicit Exception clause; §2.7 table row relabeled "official final / best model"; README §2/§3/§8 updated; open decision #4 updated (default NO on-device; adopted as final best model). This supersedes the "open decision #4 ... unchanged" clause in the 2026-10-12 experiments entry directly below. Documentation-only; no source/data/model/eval artifacts changed. | Standing rule 1 (living spec must reflect the final decision) + explicit user decision that YAMNet + logistic is the official final model. | **Decided (user, 2026-10-12)** |
| 2026-10-12 | **Two isolated architecture-comparison experiments added (experimental — NOT adopted): per-class binary CNNs with a logistic head, and with an MLP head.** Self-contained folders at the repo root (`experiment_binary_cnn_lr/`, `experiment_binary_cnn_mlp/`); neither modifies `data/`, the frozen preprocessing, the split, the locked CNN, or the YAMNet pipeline, and neither opens the test split. Both reuse the locked Phase 2A preset-A windows (train 324,613 / val 74,830). First stage = 17 per-class binary CNNs, one sigmoid each, 56,129 params per CNN; the second stage is the only difference. (1) `experiment_binary_cnn_lr/` (exp01, CNN -> sklearn `LogisticRegression` lbfgs; 8 epochs; 4,571.6 s): window val 0.4732 / **clip val 0.5902**; 720-clip demo 417/720 = 57.9%, 0 errors. (2) `experiment_binary_cnn_mlp/` (mlp01, CNN -> tf.keras MLP 17 -> 64 -> 17 softmax; frozen CNNs reused; window-level feature matrix `z` cached; head early-stopped, best head at epoch 1): window val 0.4748 / **clip val 0.5828**; 162.9 s; 720-clip demo 419/720 = 58.2%, 0 errors. Both trail the YAMNet + logistic comparator (0.7769) and are essentially equal to each other, so the second-stage head (logistic vs MLP) is not the bottleneck — the frozen-CNN feature stage is (per-class binary val accuracy ~0.76–0.98, yet combined clip accuracy ~0.59). Documented as blueprint §2.7; README §8 status and repository layout refreshed. Neither variant is adopted; §2.6 locked architecture and open decision #4 (YAMNet not adopted as the deployed perception layer) are unchanged. | Standing rule 1 (living spec must reflect the final decision): record the explored architecture variants and their measured validation results, and formally state that the head swap did not close the gap to the YAMNet comparator. | **Experimental — not adopted** |
| 2026-10-11 | **Change Log housekeeping pass: stale status cells superseded and two scaffold-entry errors corrected (documentation-only).** (1) Status cells that still read "pending user authorization" for work since executed now carry "superseded: executed 2026-10-10, see later entries" — the YAMNet fit-driver prep rows (`--compare-clip-eval` arg fix; scikit-learn 1.9 `multi_class` fix), the Phase 3A preset-scaffold row, and the YAMNet frozen-embedding baseline row. (2) YAMNet scaffold mapping error corrected: the "covers 15 of 17 classes - Glass_Breaking and Gunshot have no YAMNet label and are never predicted" note was wrong; the mapping now covers all 17 classes — Gunshot -> YAMNet idx 421 ("Gunshot, gunfire") and Glass_Breaking -> idx 435 ("Glass") + 437 ("Shatter"), confirmed by an empty "Unmapped classes" list in the current runs. (3) Clip-eval metric label corrected in the 2026-10-09 clip-eval row: the 0.5414 / 0.5113 / 0.5321 figures are the `clip_mean_softmax` level, not "clip majority-vote mean-softmax"; the clip majority-vote level is 0.5272 / 0.5003 / 0.5207. The row's "comparator row the YAMNet fit report must reproduce" pointer is superseded (executed comparator = `baseline_A` clip_mean_softmax 0.4994 / 0.4689 / 0.5193). (4) Note added to the YAMNet comparator row that `scripts/results/phase3a_embedding/yamnet_fit2/` is an intermediate run with the incomplete mapping (zero-shot 0.5321 / 0.4692 / 0.4907; Glass_Breaking + Gunshot unmapped) and must not be quoted; current YAMNet results are `yamnet_fit2_fixed` and `yamnet_demo` (zero-shot 0.6273 / 0.5955 / 0.6135, no unmapped classes). Documentation-only; no source/data/model/eval artifacts changed, no scripts run. Audit performed against the working tree at commit `8c8b005`; findings kept in chat, no separate audit file created. | Standing rules 1-2 (living spec must reflect the final decision; do not preserve mistakes) — Change Log had stale statuses and two misstatements that post-dated execution, so they were annotated in place rather than silently rewritten. | **Done (user approved 2026-10-11)** |
| 2026-10-10 | **Documentation audit corrections: original proposal PDFs archived, class list ratified, model-metric attribution fixed.** (1) Both original PDFs moved to `docs/original_proposal/` with a `STATUS.md` mapping every superseded claim (3-class scheme, `Dense(3)` head, "Recall on Class B ≥ 90%", <50 ms latency, raw-PCM/OS-bypass, 30-50 epochs, microcontroller portability, effort estimates) to its current blueprint/code position; blueprint §12 companion-source paths and README Source Documents updated. (2) "Final class list is not fixed yet" language superseded everywhere (§1.2 scope statement, §4 class count, §5.1 heading/text, README §5) by the ratified 17-class D8 taxonomy, with `Smoke_Fire_Alarm` restated as candidate/UNRESOLVED. (3) Metric attribution corrected so `baseline_A.keras` (window 0.4243 / macro-F1 0.4048; clip mean-softmax 0.4994/0.4689/0.5193) is no longer conflated with `lightweight_cnn.keras` (byte-identical to the epoch-1 weighted checkpoint `archive/weighted_epoch1.keras`: window 0.3804 / macro-F1 0.3861; clip mean-softmax 0.5414/0.5113/0.5321 — SHAs verified 2026-10-10); README §8 states all figures are validation-set only and no test-split (Phase 3B) result exists. (4) README §8 status refresh: preset A/D/E runs + YAMNet fit executed; only Phase 3B + longer training remain gated. (5) `src/config.py` docstring corrected from "Rev 2 / class list unset" to Rev 3 / ratified D8. Documentation-only; no source/data/model/eval artifacts changed; PDF archives are byte-preserved moves. | Standing rule 2 (correct mistakes, do not preserve them) + user-approved doc audit: the original PDFs and pre-ratification wording no longer describe the system; validation figures must be attributable to the exact model that produced them. | **Done (user approved 2026-10-10)** |
| 2026-10-10 | **YAMNet fit-stage comparator clarified (correction to the 2026-10-09 scaffold entry below).** The executed YAMNet fit reports (`scripts/results/phase3a_embedding/{yamnet_demo, yamnet_fit2, yamnet_fit2_fixed}/report.md`) all record the CNN comparator as `scripts/results/phase3a_clip_eval/baseline_A (clip_mean_softmax level)` with figures 0.4994 / 0.4689 / 0.5193 — NOT `weighted_epoch1_fixed` as the scaffold entry intended. **[NOTE 2026-10-10: `scripts/results/phase3a_embedding/yamnet_fit2/` is an INTERMEDIATE run produced before the zero-shot mapping was completed (old mapping, zero-shot accuracy 0.5321 / macro-F1 0.4692 / weighted-F1 0.4907, with Glass_Breaking and Gunshot left unmapped) — it must NOT be quoted as a result. The current YAMNet baseline results are `yamnet_fit2_fixed` and `yamnet_demo` (zero-shot accuracy 0.6273 / macro-F1 0.5955 / weighted-F1 0.6135, no unmapped classes).]** The baseline_A attribution is correct: `yamnet_demo` ran with the default comparator path. No artifacts were changed; this entry only removes the misleading "must reproduce weighted_epoch1_fixed" expectation. | Standing rule 2 (do not preserve mistakes): the executed run's own report is authoritative over the scaffold-time intent. | **Corrected 2026-10-10** |
| 2026-10-10 | **Preset E added to the controlled experiments, trained, and evaluated fairly against presets A and D on the identical preset E validation clip set.** `src/config.py` PRESETS now A-E: E = 2000 ms window / 16,000-sample 50% step / FFT 1024 / hop 512 / 64 mel -> (64, 61, 1). Build via gitignored `scripts/tmp_build_preset_features.py` into `data/processed_preset_E/` (test split never opened; frozen dB bounds -100.0 .. 33.83852005004883): clips shorter than the 2000 ms window are EXCLUDED (no padding/looping) - train 26,945 / val 6,229 windows from 5,819 / 1,337 kept clips, 1,202 / 281 zero-window clips (preset D by contrast: 0 / 0). Build outputs verified: spec_shape (64, 61, 1), labels match window_index class_id, features in [0, 1], window/clip counts consistent. Trained `presetE_freqpool` (`--preset E --pool-freq-only --run-name presetE_freqpool --confirm-train`): 25,937 params under the 100K edge cap; best epoch 10 val_acc 0.4575 / val_loss 1.8254, train_acc 0.4576 / train_loss 1.6293; saved `data/models/presetE_freqpool.keras`. Fair clip evals: `scripts/run_phase3a_clip_eval.py` gains `--restrict-clips-to-preset R` (limits evaluation to preset R's `window_index_val.csv` clip set, masks features/labels/window_index and filters val_clips, prints per-class kept/total, notes the restriction in the report) so every model is compared on the identical 1,337-clip preset E val set (zero-window clips: none). clip_mean_softmax level: `presetE_freqpool` acc 0.4847 / macro-F1 0.4482 / wF1 0.5026 (6,229 windows); `baseline_A_onE` (preset A model, preset A analysis) acc 0.5123 / 0.4771 / 0.5302 (72,101 windows); `presetD_onE` (preset D model, preset D analysis) acc 0.5557 / 0.4927 / 0.5608 (28,470 windows). Result: on the identical clip set preset D keeps its lead and preset E trails. Reports: `scripts/results/phase3a_clip_eval/{presetE_freqpool, baseline_A_onE, presetD_onE}/`. | Standing rule 1: record the preset E decision, build, training and the fair clip-eval comparison at the time they were produced. | **Executed** |
| 2026-10-09 | **YAMNet baseline fit driver: added `--compare-clip-eval` and hardened the fit-stage argument handling.** The fit stage referenced an undefined `args.compare_clip_eval` (AttributeError). New optional `--compare-clip-eval PATH` (default `scripts/results/phase3a_clip_eval/baseline_A`) is defined for `--stage fit`; a missing compare directory or summary file only prints a one-line skip notice and the report records "CNN comparator: skipped" - it never crashes the run. Removed `n_jobs` from `LogisticRegression` (no effect in scikit-learn 1.9, emits a FutureWarning). Statically checked every `args.<name>` against the `add_argument` definitions (stage, run-name, extract-dir, c, max-iter, compare-clip-eval - all match). The results directory is still created (`mkdir exist_ok=False`, refuses to reuse a run-name) only after the fit and all metric computation succeed, so a crash leaves no empty blocking run directory. Split, features, seeds and metrics untouched; py_compile green, ASCII-only; fit stage not run. | Standing rule 1: record the fit-stage argument/guard fixes at the time they were produced. | **Fixed - ready to run (fit stage still pending user authorization); superseded: executed 2026-10-10, see later entries** |
| 2026-10-09 | **YAMNet baseline fit driver fixed for the installed scikit-learn (1.9.0).** `scripts/run_phase3a_embedding_baseline.py` `_run_fit()` no longer passes `multi_class="multinomial"` (removed from `LogisticRegression` in current scikit-learn; lbfgs is multinomial for multiclass by default, so behavior is unchanged) and now prints the installed scikit-learn version in the fit-stage console output. No other call in the fit stage depends on removed or changed scikit-learn arguments (checked: `StandardScaler`, `accuracy_score`/`f1_score`/`precision_score`/`recall_score` all use currently supported keyword arguments). Split, features, seeds and metrics untouched; fit stage still not run. py_compile green, ASCII-only. | Standing rule 1: record the fit-stage compatibility fix at the time it was produced. | **Fixed - ready to run (fit stage still pending user authorization); superseded: executed 2026-10-10, see later entries** |
| 2026-10-09 | **Phase 3A controlled experiments scaffolded (analysis presets B/C/D approved, 2026-10-09; nothing trained yet by the preparation pass).** `src/config.py` gains `ANALYSIS_PRESETS` / `PRESETS` A-D: preset A = the locked 200 ms / step 1,600 root with frozen dB bounds in `data/processed/feature_stats.json`; B = 500 ms / 350 ms step / FFT 1024 / hop 512 / 64 mel (64,30,1); C = 500 ms / 400 ms step / FFT 512 / hop 256 / 64 mel (64,22,1); D = 500 ms / 250 ms step / FFT 1024 / hop 512 / 64 mel (64,14,1). Training driver `scripts/run_phase3a_training.py` is preset-aware (`--preset {A,B,C,D}`, default A = byte-identical default behavior; per-preset loader/stats/shape routing) and gains `--pool-freq-only` ((2,1) pools keep the time axis; a GlobalAveragePooling2D head replaces Flatten so long-window presets stay under the 100K edge-parameter cap - the Flatten head on preset D would be 482,641). Dry-runs verified: A defaults 56,657 params with the locked 16/32/64 + flatten + dense-32 ladder unchanged; D `--pool-freq-only` 25,937 params with time axis 14 preserved through both pools. Preset D train+val features built by the gitignored scratch `scripts/tmp_build_preset_features.py` (test split never opened; pipeline identical to A incl. the -60 dBFS floor; dB bounds frozen from A's stats): 8,639 clips (7,021/1,618), 0 zero-window, train 127,398 / val 29,363 windows (134,218/31,030 raw minus 6,820/1,667 silent-dropped); outputs `data/processed_preset_D/` (features/labels/window_index + feature_stats.json, spec_shape (64,14,1)); presets B/C not built (not needed for the controlled comparison). Dry-runs only - no model compiled, fitted, or saved this pass. | Standing rule 1: record the approved analysis-preset decision and the Phase 3A controlled-experiment scaffold at the time they were produced. | **Approved (presets 2026-10-09) - training runs pending user authorization; superseded: executed 2026-10-10, see later entries** |
| 2026-10-09 | **YAMNet frozen-embedding baseline scaffolded (PROPOSED as an experimental baseline - NOT ADOPTED as the perception layer); extract stage executed, fit stage NOT run.** Env: `.venv` gains tensorflow-hub 0.16.1 + tf-keras 2.21.0 (locked tf 2.21.0 / keras 3.15.1 / numpy 2.5.3 / protobuf 7.36.1 untouched); setuptools 84 -> 80.10.2 restores `pkg_resources` (dry-run green). Driver `scripts/run_phase3a_embedding_baseline.py`: stage `extract` (validated split-aware dataset, train+val only, test never opened; scores/embeddings from the cached YAMNet hub model) and stage `fit` (LDA-prewhitened logistic head over frozen 1,024-d embeddings; zero-shot YAMNet-521->17 mapping covers 15 of 17 classes - Glass_Breaking and Gunshot have no YAMNet label and are never predicted **[CORRECTED 2026-10-10: this was a scaffold-time error. The mapping now covers all 17 classes - Gunshot -> YAMNet idx 421 ("Gunshot, gunfire"); Glass_Breaking -> YAMNet idx 435 ("Glass") AND 437 ("Shatter"). Both demo and fit2_fixed runs print an empty "Unmapped classes" list]**; report suite: summary/per-class/top-confusions/within-vs-cross-dataset/baby-crying-recall-by-sample-rate/confusion matrices, with the CNN comparator read from `weighted_epoch1_fixed`). Extract EXECUTED: `yamnet_extract` -> 8,639 clips, 87,985 frames, 0 failures, 217.7 s -> `data/embeddings/yamnet/yamnet_extract/` (embed_mean/embed_std (8639,1024) f32, score_mean (8639,521) f32, index csv 8,639 x 6 no NaN, 521-name class map). The fit stage is implemented and wired but never executed this pass; the locked on-device CNN (section 2.6 / Phase 3) is unchanged - YAMNet remains an optional external baseline only. | Standing rule 1: record the environment decision, the independent-baseline scaffold, and the executed extract at the time they were produced. | **PROPOSED - NOT ADOPTED as the perception layer (optional baseline per section 2.6); fit pending user authorization; superseded: executed 2026-10-10, see later entries** |
| 2026-10-09 | **Phase 3A clip-eval bug fixed and re-run into a fresh report dir.** `scripts/run_phase3a_clip_eval.py` bug: the class-balancing filter interplayed with the clip print and scoped FA/P[stratified] denominators incorrectly on the majority-vote branch; corrected and re-run into `scripts/results/phase3a_clip_eval/weighted_epoch1_fixed/`. Verified numbers on the locked 74,830-window val split (1,618 kept clips): window macro-F1 0.3861 / accuracy 0.3804; clip_mean_softmax accuracy 0.5414 / macro-F1 0.5113 / weighted-F1 0.5321 (the clip majority-vote level on the same run is accuracy 0.5272 / macro-F1 0.5003 / weighted-F1 0.5207; "clip_mean_softmax" is the correct label for the 0.5414 row - the scaffold entry had folded the two levels into one mislabeled "clip majority-vote mean-softmax" row) [the "comparator row the YAMNet fit report must reproduce" pointer is superseded by the 2026-10-10 entry above: the executed YAMNet fit reports record `baseline_A (clip_mean_softmax)` 0.4994 / 0.4689 / 0.5193 as the comparator, not this run]. | Standing rule 1: record the bug fix and the verified measured numbers at the time they were produced. | **Fixed and verified** |
| 2026-10-08 | **Phase 3A EXECUTED — training script, saved model, and validation analysis.** New driver `scripts/run_phase3a_training.py` trains the locked section 7 CNN (Conv2D 16/32/64 → 2× MaxPool on the 64×5 grid → Dense 32 → Dropout 0.3 → softmax 17; **56,657 params** under the 100K edge limit; Adam 1e-3, sparse CE, batch 32, ReduceLROnPlateau + EarlyStopping patience 5 with `restore_best_weights`, `DEFAULT_EPOCHS = 4` overridable via `--epochs N`; Keras 3.15 API — seed helper is `set_random_seed`). After training it saves the full Keras model to `data/models/lightweight_cnn.keras` (directory auto-created; user-added save step; model on disk 729,178 bytes). Analysis-only driver `scripts/run_phase3a_validation_eval.py` (no retraining, test split untouched, model/preprocessing/config/dataset unmodified) loaded that model and evaluated the locked **74,830-window validation split: val loss 2.1081, val accuracy 0.4243 (31,749/74,830), macro F1 0.4048, weighted F1 0.4278** — reproducing the training run's printed val loss/acc exactly, confirming saved weights are the restored best-epoch weights. Best classes: Baby_Crying F1 0.825 (precision 0.980), Siren 0.698, Jackhammer 0.580, Vehicle_Horn 0.567 (precision 0.860). Weakest: **Doorbell (support 428) never predicted — precision/recall/F1 all 0.000**, Glass_Breaking recall 0.091 (F1 0.158, 56% of its windows → Footsteps), Motorcycle recall 0.129 (F1 0.210). Two dominant confusion sinks: **Footsteps** (Train→Footsteps 2,235; Footsteps precision 0.256 vs recall 0.735) and **Train** (Car_Engine→Train 1,919 = 35% of Car_Engine; Train precision 0.236). Artifacts: `scripts/results/phase3a_validation/{validation_report.md, per_class_metrics.csv, confusion_matrix.csv}`. | Standing rule 1: record on the living spec the Phase 3A script creation, the user-directed model-save addition (`data/models/`), and the validation-set analysis results. | **Analysis complete** — longer training and Phase 3B test evaluation pending user approval [PARTIALLY SUPERSEDED 2026-10-11: the epoch-ceiling check was run (the baseline early-stopped at epoch 6 and restored epoch 1 under both the 10-epoch and 30-epoch ceilings, so more epochs did not help); preset E (2 s window) reached its 10-epoch cap at epoch 10 while still improving and was not extended; Phase 3B test-split evaluation remains pending user authorization (open decision #3)] |
| 2026-10-08 | **Phase 2B gate RESOLVED — v1 FAIL → D9 v2 covariate-balanced split → v2 verdict PASS WITH WARNINGS (EXIT=0).** v1 (counts-only split) failed with 11 severe + 34 flag cells: the count-only greedy balanced class counts but never the covariates (real shifts: Jackhammer RMS, Baby_Crying feature mean, Siren/Vehicle_Horn val). Resolution (user-approved option A, see D9 amendment entry): driver `scripts/run_phase2a_preprocessing.py` gained a `covariates` stage (stages now [1/5] ledger → [2/5] split → [3/5] features → [4/5] covariates → [5/5] gates; cold start = `all` → `covariates` → `all`); `src/preprocessing/splits.py` rewritten to three phases (count greedy → covariate hill-climb with means/spreads + binned-quantile histogram → exact-KS repair with lexicographic accept over (flag-threshold crossings, energy)). Iteration findings recorded: an initial sum-only KS energy traded threshold crossings for gains elsewhere (0 severe but 16 flags) until the lexicographic rule was adopted; a stale-`cur_ids` bug (phase-1 moves never updated per-split row membership) was found by comparing the objective's crossing count against the gate's flag count and fixed by rebuilding membership after phase 1. Final split: **train 7,021 / val 1,618 / test 1,642 (68.29% / 15.74% / 15.97%)**, 5,201 groups, seed 42; features re-extracted in 191.9 s (474,931 windows: 324,613 / 74,830 / 75,488; frozen dB range now **[−100.00, 33.84]** from the new train split); Stage-1 D11 gates **PASS** (max proportion deviation 0.03, 0 shared hashes). Phase 2B gate v2 (`scripts/run_phase2b_split_gate.py`): C1 max_dev 0.0300, problems 0; C2 pooled feature-mean delta max 0.0011 (flag 0.03), KS cells **0 severe / 3 warnings** (Jackhammer rms train–test D=0.2087, Vehicle_Horn rms train–val D=0.1653, Siren rms train–val D=0.1557), 6 underpowered cells reported not blocking; C3 leakage PASS; C4 17/17 panels rendered (honesty line: reviewer sign-off pending); C5 artifacts OK; report `scripts/results/phase2b_split_gate/phase2b_report.md`. | Living spec standing rule 1: full §6 gate cycle executed; FAIL root-caused, split amended under approved option A, v2 verdict recorded at the time it was produced. | **PASS WITH WARNINGS** — training may be proposed next (warnings documented for manual reviewer) |
| 2026-10-08 | **D9 AMENDED — split objective extended from counts-only to covariate balance (user decision: option A, covariate-balanced split).** The seeded source-group split still starts from the original count greedy (seed 42 kept, 70/15/15 targets, groups atomic), but is now followed by two deterministic hill-climb phases in `src/preprocessing/splits.py`: (2) single-group moves minimizing a per-class objective over count deviation, covariate means/spreads (class-sd units) for `processed_duration_s` / `rms_db` / `clip_feature_mean`, plus a binned-quantile shape term (10 equal-count rank bins per covariate per class, split histograms matched to class histogram); (3) exact-KS repair (custom `_ks_d`, exact vs scipy) with a lexicographic accept rule — fewer cells crossing the gate C2 flag thresholds (0.20 duration / 0.15 RMS / feature mean) always wins, ties break on the full energy (cell terms + squared KS with a 3× barrier above threshold). Feasibility: per-class count band max(3% of class size, initial deviation) — stricter than G4's 5%; underpowered rule `KS_MIN_N=15` clips per side — cells below it are reported but never optimized or blocking (over-fitting sampling noise, e.g. Doorbell n≈51). Added `compute_clip_covariates()` in `features.py` and `data/provenance/clip_covariates.csv` (10,281 rows; affine-equivalent to the raw mel-dB clip mean under frozen bounds, so it never needs refreshing when the split changes). | Phase 2B v1 FAIL root cause: count-only balancing left real distribution shifts between splits that the mandatory §6 gate flags as severe — the split must be balanced on the covariates the gate measures. | **Approved** (user decision 2026-10-08) — implemented and verified by the Phase 2B v2 entry above |
| 2026-10-08 | **W1 RESOLVED — UrbanSound8K made self-contained.** The 5,732 US8K clips were copied from the Kaggle cache (`~\.cache\kagglehub\...\versions\1\fold*`) into `data/raw/urbansound8k/fold*/` (~1.4 GB), every copy verified against the Phase 1.5/2A ledger SHA-256 (0 mismatches), and `data/manifests/master_audio_manifest.csv` `local_path` rewritten to project-relative `data\raw\urbansound8k\...` (matching the FSD50K/ESC-50 convention; manifest content otherwise untouched). Gates re-run after the rewrite: **PASS** with identical splits (seed 42 deterministic), and a spot-read through `features.load_clean` succeeds. `.gitignore` gains `!data/processed/feature_stats.json` so the frozen dB-bounds provenance is tracked while the 580 MB `.npy` feature files stay ignored. Phase 1.5 warning W1 (Change Log 2026-10-07) is now closed; `scripts/tmp_w1_copy_us8k.py` records the migration (gitignored as scratch). | User decision 2026-10-08: copy US8K into `data/raw/` now so the repository does not depend on the Kaggle cache. | **Resolved** |
| 2026-10-08 | **Phase 2A preprocessing pipeline EXECUTED — all D11 leakage gates PASS.** Driver `scripts/run_phase2a_preprocessing.py` ran ledger → split → features → gates in 515 s. Results: 10,565 manifest rows → **10,281 kept / 284 excluded** (254 extreme-short, 30 silent, no overlap) exactly as approved; split **7,110 / 1,577 / 1,594 clips (69.16% / 15.34% / 15.50%)** across **5,201 source groups** (of 5,341 ledger groups; 140 fully excluded), seed 42, 4 multi-class W6 groups atomic, **max per-class proportion deviation 2.74 pp** (tolerance ±5 pp, documented in `src/preprocessing/gates.py`); features: **474,931 windows** (train 326,550 / val 74,356 / test 74,025) from 510,336 raw with 35,405 quiet interior windows (6.9%) dropped at the −60 dBFS floor, frozen dB range **[−100.00, +34.16]**, artifacts `data/processed/features_{split}.npy` (float32, (N,64,5,1)) + `labels_{split}.npy` + `window_index_{split}.csv` + `feature_stats.json`; gates G1–G5 all PASS with **0 shared SHA-256 across splits** (`scripts/results/phase2a_gates.json`); provenance ledger `data/provenance/phase2a_clip_ledger.csv` (10,565 rows × 27 columns, §5.4 complete); full write-up `scripts/results/phase2a_report.md`. Operational readings recorded: D4 applied at window level too (−60 dBFS floor), G4 tolerance ±5 pp due to atomic groups up to 100 clips. **W1 still open** (US8K paths point into the Kaggle cache; copy to `data/raw/` deferred to user decision); Smoke_Fire_Alarm still 0 clips; §6 full gate (distribution + spectrogram panels) proposed as Phase 2B. | Living spec standing rule 1: Phase 2A executed under the approval entry below; results, tolerances and open items recorded at the time they were produced. | **PASS** — Phase 2B (split-quality visuals) and training may be proposed next |
| 2026-10-08 | **MAJOR — Phase 2A preprocessing policy APPROVED (decisions D1–D12 plus amendments OA1–OA4); implementation authorized.** Approved policy: exclusions driven by Phase 1.5 `file_health.csv` flags — drop 254 extreme-short clips under 0.5 s (D1) and 30 silent clips (D4); 284 excluded total with no overlap, 10,281 clips kept (smallest class Doorbell 51, largest Car_Engine 1,000); truncate every clip to its first 15.0 s (D2); mono downmix (D6) then soxr_hq resample to 16 kHz (D5); windowing 200 ms (3,200 samples) with **step 1,600 samples = true 50% overlap (OA1)**, about 149 windows per 15 s clip, windows at or below minus 60 dBFS dropped; per-window features via batched `librosa.stft` (center=False, n_fft 1024, STFT hop 512 = 50% of n_fft — HOP_LENGTH comment corrected, OA3) yielding exactly 5 frames, 64 mel bands, `power_to_db` referenced to 1.0, then scaled to [0,1] with dB bounds frozen from train-split statistics in `data/processed/feature_stats.json` — output shape **(64,5,1)** (`SPEC_SHAPE` amended from (64,63,1), OA2); deterministic seeded greedy approximately 70/15/15 split per class with source groups atomic across classes (D9), written to `data/splits/clip_splits.csv` + `data/splits/group_splits.csv`; D11 leakage gates mandatory before any training (SHA-256 disjoint across splits, each group maps to exactly one split, no clip_id in two splits, per-class proportions within tolerance); 17-class taxonomy populated alphabetically in `src/config.py` (D8) with Smoke_Fire_Alarm remaining candidate/UNRESOLVED and no placeholder rows; provenance ledger `data/provenance/phase2a_clip_ledger.csv` records every clip's outcome and parameters (§5.4). **OA4 (discovered during implementation): babycry group key corrected** — the Phase 1.5 subject regex required a 13-digit timestamp and silently dropped 106 DonateACry clips carrying 10-digit timestamps, undercounting subjects as 338 with 32 multi-clip; correct UUID extraction yields **170 DonateACry subjects, 54 multi-clip**, and the 237 ReCANVo clips come from only **14 recording sessions** (largest 53 clips), so ReCANVo groups by session prefix `YYMMDD_HHMM` instead of per-clip to stop same-session train/test contamination. UrbanSound8K Kaggle-cache absolute paths (W1) are read as-manifested for Phase 2A; copying into `data/raw/` remains a pending user decision. | User approved the Phase 2A proposal on 2026-10-08 (all 12 decisions; OA1 step 1,600; OA2/OA3 config amendments). OA4 is a Standing Rule 1 discovery: Phase 1.5 validator subject-regex bug plus the ReCANVo session structure found while implementing D9. | **Approved — implementation in progress** |
| 2026-10-07 | **Phase 1.5 Dataset Sanity & Validation executed — verdict: PASS WITH WARNINGS (gate passed; preprocessing may begin once warnings are handled).** Full read-only audit of all 10,565 manifest rows: 10,565/10,565 files present, readable, 0 missing/corrupt/zero-length, 0 SHA-256 mismatches (all recomputed), 0 exact duplicates, 0 cross-dataset source-ID leakage. New artifacts: `scripts/validate_dataset.py`, `scripts/make_diag_spectrograms.py`, `scripts/diag_numeric.py`, `scripts/results/dataset_validation/{file_health.csv,validation_data.json,diagnostics.csv,dataset_validation_report.md,spectrograms/*.png}`, `scripts/results/dataset_validation_summary.csv`; master manifest untouched. Key warnings to handle in preprocessing/splitting: **W1** UrbanSound8K `local_path`s are absolute paths into the Kaggle cache (`~\.cache\kagglehub\...`), not project files — copy into `data/raw/` to make the dataset self-contained; **W2** `Smoke_Fire_Alarm` has 0 clips (17/18 classes); **W3** 862 clips <1 s + 878 clips >15 s (trim/window policy); **W4** 685 clipped clips (Gunshot 41%); **W5** 30 silent + 308 low-energy clips (removal candidates); **W6** 4 US8K fsID groups span two classes (fsID 180937 & 77751: Drilling+Jackhammer; 106905: Car_Engine+Siren; 176638: Car_Engine+Vehicle_Horn) and babycry has 32 multi-clip subject UUIDs — source/group-aware split mandatory; **W7** 10 sample rates (8 kHz–192 kHz) + mono/stereo mix (US8K mixed, FSD50K/ESC-50 44.1 kHz mono, babycry 8 kHz mono); **W8** FSD50K `Alarm` semantically broad (241/733 co-labeled bicycle-bell/doorbell/clock/engine/train) and Help_Shouting 77/498 crowd-context — mapping review before training; **W9** imbalance 19:1, Doorbell 53 / Motorcycle 238 thin; **W10** FSD50K source granularity is file-level (residual hidden correlation possible). All 17 classes are source-group split-FEASIBLE for 70/15/15. US8K built-in `fold` column must NOT be used as the val/test split. | Living spec standing rule 1: Phase 1.5 validation gate outcome + dataset-strategy discoveries (US8K location, cross-class source groups, silence/clipping/short-long distributions, FSD50K label breadth) recorded at discovery time so preprocessing/splitting obligations are normative. | Accepted — gate passed with warnings; next phase = preprocessing (must address W1–W10). |
| 2026-10-07 | **Dataset acquisition phase executed — 17/18 project classes now have audio; `Smoke_Fire_Alarm` marked UNRESOLVED.** New sources: **FSD50K** (`Fhrozen/FSD50k`, 4,027 clips — 733 Alarm + 3,294 across Vehicle_Horn/Doorbell/Train/Aircraft/Motorcycle/Help_Shouting/Glass_Breaking/Gunshot/Knocking/Footsteps), **ESC-50** (306 clips of 9 categories, CC-BY-NC-3.0), **owlgebra-ai/babycry** (500 Baby_Crying clips from one parquet shard: DonateACry all labels except `burping` + ReCANVo `distress` rows of dysregulated/frustrated labels; ESC-50-source rows excluded as duplicates; per-row licenses ODbL/CC-BY-4.0 preserved), UrbanSound8K (5,732, pre-existing). Conservative ESC-50 mappings fixed: `crying_baby→Baby_Crying`, `clock_alarm→Alarm`, `door_wood_knock→Knocking` (an earlier `Alarm_Beep`/`Child_Crying`/`Knock` mapping had target classes outside the taxonomy). **Zenodo 3689288** (142 MB OSR/FSL domestic alarm corpus) downloaded but NOT used: its 24 alarm classes are anonymized (`pattern_01..24`) with no documented fire-alarm/doorbell mapping — zip retained unused. Leakage guards: FSD50K `fname` vs US8K `fsID`; ESC-50 `src_file` vs US8K `fsID` + FSD50K `source_id` (54 ESC-50 clips dropped); cross-dataset duplicate SHA-256 removed (4). Final master manifest `data/manifests/master_audio_manifest.csv`: **10,565 clips, all validation `ok`, 0 duplicate hashes, 16.3 audio hours** — Aircraft 338, Alarm 755, Baby_Crying 540, Car_Engine 1000, Dog_Bark 1000, Doorbell 53, Drilling 1000, Footsteps 585, Glass_Breaking 515, Gunshot 793, Help_Shouting 498, Jackhammer 1000, Knocking 387, Motorcycle 238, Siren 929, Train 468, Vehicle_Horn 466, **Smoke_Fire_Alarm 0 (UNRESOLVED — no conservative public source found; FSD50K `Fire` = crackling fire, not alarms)**. Doorbell (53) and Motorcycle (238) are below the ~500 sufficiency guide; additional sources may be investigated before Phase 1 training per the flexible sufficiency rule. Disk: ~3.7 GB. | Living spec standing rule 1: dataset strategy, mappings, and gaps recorded as decided. | Accepted |
| 2026-10-07 | **MAJOR — architecture expanded from a two-layer classifier+trigger scheme into a four-stage "Complete Context-Aware Acoustic Supervisor": Perception (detection) → Context/Policy (does it matter to this user?) → Decision (confidence + temporal verification) → Response.** Flow restated as: Ambient Audio → Acoustic Event Detection → User Trigger Policy → Confidence + Temporal Verification → Context Response Engine → Response Actions → State Restoration. New normative section **§2A Context Response Engine** defines response types **A** audio-mode, **B** notification ("Siren detected — Transparency mode enabled"), **C** warning/priority alert for configured high-priority events, **D** user-confirmed quick action for configured critical events (PRESENTATION ONLY — explicit user confirmation required, disabled by default, never automatic, no guaranteed emergency detection, "automatic emergency calling" forbidden), **E** state management (release timer, preferred-mode restoration, cooldown/duplicate suppression). §2.1 rewritten as the four-stage separation table (detection / policy / decision / response). Profiles documented as **configuration policies** (added **Custom** example) — never models; profile switch never requires retraining. §1.4 illustrative example behaviour added (probabilities labelled illustrative, not measured). §2.6 **ML architecture LOCKED**: CNN, <100K parameters, 16 kHz, 200 ms windows, 50% overlap, 64 mel, FFT 1024, hop 512, N finalized Phase 1 — explicitly NOT replaced by large audio transformers (e.g., AST 86M+); Hugging Face models only as optional external baselines. Supervisor interface + demo pipeline added (§2A.5); Phase 5 retitled and extended with response types A–E; Phase 6 demo updated (supervisor interface, response diversity, profile switching); §8 adds no-automatic-emergency-action and not-merely-classification guardrails; §9 adds Q7. **Documentation-only change** (README + blueprint): no source code, no datasets, no training, no implementation, no results, no commits. Section numbering of existing sections preserved (new section inserted as 2A) so all existing cross-references (§3, §5.1, §6, §7, §8, §10, §11) remain valid. | User-directed concept expansion: the deliverable is a complete context-aware acoustic supervisor with a response subsystem, not just an event classifier. Clarifies the decision/response boundary, makes the emergency-action safety boundary explicit, and freezes the ML architecture so documentation growth cannot drift the model design. | Accepted |
| 2026-10-05 | **MAJOR — three broad acoustic classes SUPERSEDED by a user-selectable acoustic trigger architecture.** The fixed 3-class scheme (A continuous hum / B transient danger / C quiet) is replaced by detection of *specific environmental sound events* (candidates: emergency siren, vehicle horn, doorbell, baby/child crying, smoke alarm) that the **user selects as triggers**. New two-layer separation is normative: **(1) model detection** = "what sound is present?" and **(2) user trigger policy** = "which detected sounds should cause an ANC/mode change?" Added: per-class confidence thresholds, temporal/event verification, per-trigger release timer, explicit no-action path for detected-but-not-enabled classes, and a hard rule that **changing a profile must not require retraining** (one model + configurable policy layer). Trigger profiles (Commuting / Travel / Home-Family) are documented as **optional** policy demonstrations, not model classes. Final class list is **not fixed** and is determined in Phase 1 against six criteria (§5.1). Dataset requirement replaced: the "exactly ~1,000 per class / equal A/B/C counts" rule is superseded by a **flexible sufficiency requirement** — each selected trigger class needs sufficient high-quality, diverse public data; classes reasonably balanced where practical; if a class is short, additional appropriate public sources are investigated before reducing or compromising quality; datasets are **never added merely to reach a numerical target**. §7 Phase 1 redefined as multi-dataset investigation + trigger-class selection; Phase 3/5 reframed around N classes and the policy layer; §6 gate generalized from A/B/C to per-class and extended with an explicit artifact/silence check; §8 adds safety/scope claims (no guaranteed emergency detection, no claim of controlling ANC on commercial earbuds, no universal sound coverage). Rev 1 three-class framing retained verbatim as superseded historical record in §1.3. Design/prior-art reference added: "Context-Aware Adjustment of Noise Cancellation for Wearable Audio Devices" (Maheshwari et al., Technical Disclosure Commons, 27 Oct 2025) — used as a **design reference only**, not reproduced; it provides no dataset, model, or results for this project. | The revised architecture (a) better matches the context-aware ANC use case, (b) allows use of established environmental sound datasets instead of forcing them into arbitrary buckets, (c) avoids forcing unrelated sounds into overly broad classes, (d) enables user-specific trigger policies, (e) should improve interpretability and potentially classification reliability, and (f) gives the project a clearer, more defensible evaluation strategy. | Accepted |
| 2026-09-07 | **Git/GitHub standing workflow established (§10):** checkpoint-based commit reminders (advisory, never automatic); explicit user request required to commit/push; pre-phase completeness + consistency check; correct-mistakes-over-preserve-history policy; cleanliness checks before major pushes; genuine-history principle. | User governance instruction. Defines how the project's development history and backup are maintained on GitHub throughout the project. | Accepted |
| 2026-09-07 | **Standing auto-update rule established:** every project-level change (source, dataset strategy, architecture, phases, preprocessing, model, evaluation, or requirements) is recorded in this document automatically at the time the change is made or discovered — no reminder required. Applies to all future sessions. | User governance instruction. Makes the change-log discipline self-executing so the spec never drifts from the implementation. | Accepted |
| 2026-09-07 | **Final dataset targets set to ~1,000 clips/class (from 300–500), with equal final A/B/C counts; shortage-handling order defined; padding policy tightened (avoid heavy zero-padding); provenance requirement added.** | User specification update. Encoded as authoritative §5.2. | **Superseded (2026-10-05)** — fixed 1,000/class and exact equality replaced by the flexible sufficiency requirement (§5.2); shortage-handling, padding and provenance rules retained in spirit and strengthened. |
| 2026-09-07 | **Mandatory pre-training split-validation gate added (§6): proportion check, per-class audio-distribution similarity, leakage check, and Train/Val/Test spectrogram comparison, with a do-not-train rule on failure.** | User specification update to guarantee split quality and catch bad padding/silence/distribution mismatch before training. | Accepted (generalized to per-class + artifact check, 2026-10-05) |
| 2026-09-07 | **Initial scoped change — "modify the plan when necessary using additional appropriate public data/sources" is explicitly permitted, but every such use must be documented (provenance §5.4 + this change log), never silently.** | User specification. Sets the rule for legitimate future deviation from the original Phase Plan's dataset scope. | Accepted |

*Future entries: whenever we use additional public data sources, finalize the trigger class list, adjust class mapping,

**Open decisions pending user approval (2026-10-10):**

| # | Subject | Awaiting |
|---|---|---|
| 1 | ~~Run the two prepared training runs (`baseline_A`; `presetD_freqpool`) and their evaluation drivers.~~ **EXECUTED** (2026-10-10: `presetE_freqpool` also trained/evaluated fairly, see Change Log). | DONE |
| 2 | ~~Run the YAMNet embedding `fit` stage.~~ **EXECUTED** (2026-10-10; validation side-by-side report in `scripts/results/phase3a_embedding/`, comparator = `baseline_A` clip-mean-softmax). | DONE |
| 3 | Longer training than `DEFAULT_EPOCHS` 4 and any Phase 3B test-split evaluation. | user OK |
| 4 | Adopt YAMNet or any external model as the on-device perception layer - CLOSED: **not** adopted on-device (per section 2.6). **Updated (decision D13, 2026-10-12):** YAMNet + logistic IS adopted as the project's **official final / best-performing model** (validation clip acc 0.7769); the locked CNN remains the on-device model. | default NO on-device; **adopted as final best model** |
alter hyperparameters, or change any requirement, record it here with the reason.*

---

## 12. References

1. **Design / prior-art reference (not reproduced):** Shailesh Maheshwari, Vedant Maheshwari, Rayansh Maheshwari.
   *"Context-Aware Adjustment of Noise Cancellation for Wearable Audio Devices."* Technical Disclosure Commons,
   27 October 2025. <https://www.tdcommons.org/cgi/viewcontent.cgi?article=10034&context=dpubs_series>
   — Cited as inspiration for the context-aware ANC-adjustment problem setting. The project does **not** claim to
   reproduce its implementation, dataset, model, or experimental results.
2. Andrew Ng — Machine Learning Specialization (foundations for the supervised-learning / image-input approach).
3. **Companion sources (initial, now superseded by this living document, archived under `docs/original_proposal/` with
   a `STATUS.md` mapping their claims to the current system):** `Edge_AI_Earbud_Project_Blueprint .pdf`,
   `Edge_AI_Earbud_Phase_Plan.pdf`.

---

*End of living specification. Rev 3.*