"""Global configuration and constants for the Edge AI Acoustic Supervisor.

Single source of truth for project paths and fixed hyperparameters.
Referenced by all other modules. Keep this file consistent with the
FINAL_PROJECT_BLUEPRINT.md living specification (Rev 3).

Rev 3 note: the original fixed 3-class scheme (Class A continuous hum /
Class B transient danger / Class C quiet) and the "exactly ~1,000 clips per
class with equal final counts" dataset rule are SUPERSEDED by the
user-selectable acoustic trigger architecture. See blueprint §1.3
(historical record) and §5.2 (authoritative dataset requirements).

The trigger class list was ratified in Phase 1 as the 17-class taxonomy in
TRIGGER_CLASS_NAMES below (decision D8, 2026-10-08; blueprint §5.1 / Change
Log). Smoke_Fire_Alarm, the intended 18th class, is candidate/UNRESOLVED
(0 conservative public clips found, W2) and is intentionally absent here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = DATA_DIR / "splits"
PROVENANCE_DIR = DATA_DIR / "provenance"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

# ---------------------------------------------------------------------------
# Audio signal parameters (Phase 2 fixed parameters)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000                # Hz, standard for speech/environmental audio
WINDOW_SIZE_MS = 200                # analysis block length
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SIZE_MS // 1000    # 3,200 samples per block
WINDOW_STEP = WINDOW_SAMPLES // 2   # 1,600 samples between windows = 50% overlap (OA1, approved 2026-10-08)
HOP_LENGTH = 512                    # STFT hop = 50% of N_FFT (OA3 comment fix; not the window step)
N_FFT = 1024                        # FFT window size
N_MELS = 64                         # mel bands
SPEC_FRAMES = (WINDOW_SAMPLES - N_FFT) // HOP_LENGTH + 1  # librosa center=False -> exactly 5 frames
SPEC_SHAPE = (N_MELS, SPEC_FRAMES, 1)  # (64, 5, 1) per window, no padding (OA2, approved 2026-10-08)

# ---------------------------------------------------------------------------
# Trigger class mapping (finalized in Phase 1, ratified as D8 on 2026-10-08)
# ---------------------------------------------------------------------------
# 17 of 18 planned classes have audio (blueprint Change Log 2026-10-07).
# Smoke_Fire_Alarm has 0 clips and remains candidate/UNRESOLVED: no
# placeholder id is created and no smoke-alarm claim is made (§1.2, W2).
# Ids are the alphabetical order of the names below, fixed here so that
# label <-> index mapping never shifts for the rest of the project.
TRIGGER_CLASS_NAMES: tuple[str, ...] = (
    "Aircraft",
    "Alarm",
    "Baby_Crying",
    "Car_Engine",
    "Dog_Bark",
    "Doorbell",
    "Drilling",
    "Footsteps",
    "Glass_Breaking",
    "Gunshot",
    "Help_Shouting",
    "Jackhammer",
    "Knocking",
    "Motorcycle",
    "Siren",
    "Train",
    "Vehicle_Horn",
)
TRIGGER_CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(TRIGGER_CLASS_NAMES)}
NUM_TRIGGER_CLASSES: int = len(TRIGGER_CLASS_NAMES)

# Dataset requirements (blueprint §5.2, revised 2026-10-05):
#   - each selected trigger class needs sufficient high-quality, diverse public
#     data for reliable training and evaluation
#   - classes reasonably balanced where practical
#   - if a class is short, additional appropriate public sources are
#     investigated before reducing or compromising quality
#   - a dataset is never added merely to reach a numerical target
# There is deliberately no fixed per-class sample-count constant here.
# Phase 2A (2026-10-08): no zero-padding is used — every kept clip yields
# whole 3,200-sample windows because clips < 0.5 s are excluded (D1).
MAX_ZERO_PAD_FRACTION: float = 0.0  # limited-padding threshold (§5.2 padding policy)

# Stratified split ratios (blueprint §6)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Phase 2A preprocessing policy (approved 2026-10-08 — blueprint Change Log)
# ---------------------------------------------------------------------------
MANIFEST_PATH = DATA_DIR / "manifests" / "master_audio_manifest.csv"
FILE_HEALTH_PATH = PROJECT_ROOT / "scripts" / "results" / "dataset_validation" / "file_health.csv"
FEATURE_STATS_PATH = PROCESSED_DATA_DIR / "feature_stats.json"
TRUNCATE_SECONDS = 15.0             # D2: keep only the first 15.0 s of each clip
EXCLUDE_SHORTER_THAN_S = 0.5        # D1: drop clips flagged extreme_short (< 0.5 s)
EXCLUDE_SILENT_DBFS = -60.0         # D4: drop clips flagged silent; also drops quiet windows
DB_POWER_REF = 1.0                  # librosa.power_to_db reference
SPLIT_SEED = 42                     # D9: deterministic greedy source-group split

# ---------------------------------------------------------------------------
# Analysis presets (Phase 3A controlled experiments, approved 2026-10-09)
# ---------------------------------------------------------------------------
# Each preset is a window/FFT/mel combination used to probe the confounds
# found by the Phase 3A diagnostics (short time axis, dataset confounding,
# 8 kHz Baby_Crying subset). Preset A is the locked Phase 2B baseline and
#     must never change: it points at the data/processed root with the frozen
#     dB bounds and spec shape (64, 5, 1) already recorded there. Presets B/C/D/E
#     write and read their own directories only and reuse preset A's frozen
#     global dB bounds and the same clip-level split manifest. All presets use
# SAMPLE_RATE (16 kHz), quiet-window dropping (RMS <= -60 dBFS), and
# librosa.melspectrogram with center=False; frames =
# 1 + (window_samples - n_fft) // hop_length.
PRESET_A = "A"
PRESETS: dict[str, dict] = {
    "A": {
        "description": "locked baseline: 200 ms, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 5, 1)",
        "window_ms": 200,
        "window_samples": 3_200,
        "window_step_samples": 1_600,
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 64,
        "spec_shape": (64, 5, 1),
        "processed_dir": PROCESSED_DATA_DIR,
        "feature_stats_path": FEATURE_STATS_PATH,
    },
    "B": {
        "description": "1000 ms window (certainty across longer context), 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 30, 1)",
        "window_ms": 1000,
        "window_samples": 16_000,
        "window_step_samples": 8_000,
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 64,
        "spec_shape": (64, 30, 1),
        "processed_dir": DATA_DIR / "processed_preset_B",
        "feature_stats_path": DATA_DIR / "processed_preset_B" / "feature_stats.json",
    },
    "C": {
        "description": "200 ms window with high spectral resolution (pitch detail), no overlap, FFT 512, hop 128, 64 Mel -> (64, 22, 1)",
        "window_ms": 200,
        "window_samples": 3_200,
        "window_step_samples": 3_200,
        "n_fft": 512,
        "hop_length": 128,
        "n_mels": 64,
        "spec_shape": (64, 22, 1),
        "processed_dir": DATA_DIR / "processed_preset_C",
        "feature_stats_path": DATA_DIR / "processed_preset_C" / "feature_stats.json",
    },
    "D": {
        "description": "500 ms window, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 14, 1) by the extraction formula",
        "window_ms": 500,
        "window_samples": 8_000,
        "window_step_samples": 4_000,
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 64,
        "spec_shape": (64, 14, 1),
        "processed_dir": DATA_DIR / "processed_preset_D",
        "feature_stats_path": DATA_DIR / "processed_preset_D" / "feature_stats.json",
    },
    "E": {
        "description": "2000 ms window, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 61, 1) by the extraction formula",
        "window_ms": 2000,
        "window_samples": 32_000,
        "window_step_samples": 16_000,
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 64,
        "spec_shape": (64, 61, 1),
        "processed_dir": DATA_DIR / "processed_preset_E",
        "feature_stats_path": DATA_DIR / "processed_preset_E" / "feature_stats.json",
    },
}

# ---------------------------------------------------------------------------
# Model architecture (Phase 3) — final shape decided after Phase 1
# ---------------------------------------------------------------------------
# These are the original starting-candidate values from the source blueprint.
# The output layer width must equal NUM_TRIGGER_CLASSES (17 since Phase 1),
# and per-class weighting / recall targets are decided only after the class
# set and dataset are established (blueprint §7 Phase 3).
CONV1_FILTERS = 16
CONV2_FILTERS = 32
CONV3_FILTERS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.3

# Training configuration
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
CLASS_WEIGHTS: dict[int, float] | None = None   # decided in Phase 1/3

# Evaluation targets
LATENCY_BUDGET_MS = 50                            # inference budget per window (design target)
INFERENCE_CADENCE_MS = 200                        # match WINDOW_SIZE_MS
# No per-class accuracy/recall targets are defined here: they must be derived
# from the real dataset after Phase 1 (blueprint §7 Phase 3).

# ---------------------------------------------------------------------------
# Live capture (Phase 4) / trigger policy + routing (Phase 5)
# ---------------------------------------------------------------------------
MAX_DEVICE_CHANNELS = 1
ROLLING_BUFFER_WINDOWS = 1
# Temporal verification: a candidate trigger must persist for this many
# consecutive windows before a mode switch (blueprint §7 Phase 5). Safety-
# relevant triggers may instead require a single high-confidence window.
TRIGGER_CONFIRM_WINDOWS = 3
TRIGGER_RELEASE_SECONDS = 5                       # restore preferred mode after trigger absent
TRIGGER_MIN_CONFIDENCE = 0.80                     # per-policy default threshold

# Audio-mode outputs (simulated; see blueprint §3). Modes remain valid; what
# triggers them is the user's policy, not a fixed class identity.
MODE_MAX_ANC = "Maximum ANC Mode"          # brown/pink noise mask playback
MODE_SAFETY = "Safety Transparency Mode"   # direct unbuffered mic passthrough
MODE_STANDBY = "Low-Power Standby Mode"    # reduced sampling rate loop