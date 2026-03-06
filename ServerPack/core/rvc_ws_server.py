import onnxruntime as ort
try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False
import asyncio
import struct
import traceback
import numpy as np
import json
import threading
import time
import contextlib

from dataclasses import dataclass, field
from typing import Dict, Optional

import websockets
from scipy.signal import resample_poly

import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_voice_changer_root() -> str:
    """Locate local w-okada backend root (voice_changer package)."""
    env_root = os.environ.get("VOICE_CHANGER_ROOT", "").strip()
    candidates = [
        env_root,
        os.path.join(BASE_DIR, "..", "..", "ThirdParty", "wokada", "voice_changer"),
        os.path.join(BASE_DIR, "voice_changer"),
        os.path.join(os.path.dirname(BASE_DIR), "voice_changer"),
        os.path.join(os.getcwd(), "_Scripts", "voice_changer"),
        os.path.join(os.getcwd(), "voice_changer"),
    ]

    checked = []
    for c in candidates:
        if not c:
            continue
        root = os.path.abspath(c)
        checked.append(root)
        if not os.path.isdir(root):
            continue
        probe = os.path.join(root, "RVC", "pipeline", "Pipeline.py")
        if os.path.isfile(probe):
            print(f"[BOOT] voice_changer root: {root}")
            return root

    raise FileNotFoundError(
        "voice_changer root not found. Checked:\n - " + "\n - ".join(checked)
    )

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

VOICE_CHANGER_DIR = resolve_voice_changer_root()
VOICE_CHANGER_PARENT = os.path.dirname(VOICE_CHANGER_DIR)
if VOICE_CHANGER_PARENT not in sys.path:
    sys.path.insert(0, VOICE_CHANGER_PARENT)

from voice_changer.RVC.pipeline.Pipeline import Pipeline
from voice_changer.RVC.embedder.FairseqHubert import FairseqHubert
from voice_changer.RVC.embedder.FairseqContentvec import FairseqContentvec
from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer
from voice_changer.RVC.inferencer.RVCInferencer import RVCInferencer
from voice_changer.RVC.pitchExtractor.RMVPEOnnxPitchExtractor import RMVPEOnnxPitchExtractor

# =========================
# CONFIG LOADER
# Priority: env var > server_config.json > code default
# =========================

def _load_server_config() -> dict:
    """BASE_DIR/server_config.json 읽기. 없으면 빈 dict 반환."""
    config_path = os.path.join(BASE_DIR, "..", "config", "server_config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print(f"[BOOT] server_config.json loaded: {config_path}")
            return cfg
        except Exception as e:
            print(f"[BOOT] Warning: failed to read server_config.json: {e}")
    return {}

def _cfg_get(env_key: str, cfg: dict, cfg_key: str, default):
    """환경변수 → config → default 순서로 값 반환."""
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val
    if cfg_key in cfg:
        return cfg[cfg_key]
    return default

_cfg = _load_server_config()

# =========================
# CONFIG
# =========================
WS_HOST = "0.0.0.0"
WS_PORT = 8765

IN_SR = 48000
MODEL_SR = 40000
RVC_SR = 16000
WINDOW = 160

# Default hop size (20ms @ 48kHz = 960 samples)
DEFAULT_HOP_MS = 20
DEFAULT_HOP_SAMPLES = IN_SR * DEFAULT_HOP_MS // 1000  # 960
MAX_BLOCK_SAMPLES = 48000  # max chunk size from UI
PERF_LOG_EVERY = 25  # performance log interval (frames)

# Minimum context required for stable RVC inference
MIN_CTX_48K = 4800  # 100ms @ 48kHz — minimum safe margin

# Path/model config — env var > server_config.json > default
_models_root_raw = _cfg_get("RVC_MODELS_ROOT", _cfg, "models_root", os.path.join(BASE_DIR, "models"))
MODELS_ROOT = os.path.abspath(_models_root_raw)

if not os.path.isdir(MODELS_ROOT):
    os.makedirs(MODELS_ROOT, exist_ok=True)
    print(f"[MODEL] Created models folder: {MODELS_ROOT}")

_hubert_raw  = _cfg_get("RVC_HUBERT_PATH",  _cfg, "hubert_path",      os.path.join(MODELS_ROOT, "hubert_base.pt"))
_rmvpe_raw   = _cfg_get("RVC_RMVPE_PATH",   _cfg, "rmvpe_model_path", os.path.join(MODELS_ROOT, "rmvpe.onnx"))
HUBERT_PATH       = os.path.abspath(_hubert_raw)
RMVPE_MODEL_PATH  = os.path.abspath(_rmvpe_raw)

DEFAULT_MODEL_ID = _cfg_get("RVC_DEFAULT_MODEL", _cfg, "default_model_id", "")

_device_raw  = _cfg_get("RVC_DEVICE",  _cfg, "device",  "cuda")
_is_half_raw = _cfg_get("RVC_IS_HALF", _cfg, "is_half", "true")
DEVICE  = _device_raw
IS_HALF = str(_is_half_raw).lower() not in ("false", "0", "no")
EMBEDDER_TYPE = _cfg_get("RVC_EMBEDDER_TYPE", _cfg, "embedder_type", "hubert")

# RVC parameters (defaults)
SID = 0
INDEX_RATE = 0.75
IF_F0 = 1
SILENCE_FRONT = 0.0
EMB_OUTPUT_LAYER = 9
REPEAT = 0
PROTECT = 0.5

FADE_MS = 12.0

# =========================
# PRESETS (SOLA/VAD parameters, independent of hop size)
# =========================
PRESETS = {
    # Default balanced preset
    0: dict(
        hop_ms=20,
        vad_rms=0.0020, vad_abs=0.0070, hold=3,
        agc_target=0.0, agc_max=0.0,
        overlap=960, search=608, max_step=192,
        extra_48k=3840,
        output_gain=0.7, limit=0.95,
        f0_up_key=0,
        index_rate=0.75
    ),
    1: dict(
        hop_ms=20,
        vad_rms=0.0020, vad_abs=0.0060, hold=2,
        agc_target=0.020, agc_max=2.0,
        overlap=1440, search=960, max_step=480,
        extra_48k=1920,
        output_gain=0.7, limit=0.95,
        f0_up_key=0,
        index_rate=0.75
    ),
    2: dict(
        hop_ms=20,
        vad_rms=0.0014, vad_abs=0.0070, hold=1,
        agc_target=0.018, agc_max=1.8,
        overlap=1200, search=600, max_step=360,
        extra_48k=1920,
        output_gain=0.7, limit=0.95,
        f0_up_key=0,
        index_rate=0.75
    ),
    3: dict(
        hop_ms=20,
        vad_rms=0.0010, vad_abs=0.0040, hold=3,
        agc_target=0.025, agc_max=2.0,
        overlap=1920, search=960, max_step=480,
        extra_48k=1920,
        output_gain=0.7, limit=0.95,
        f0_up_key=0,
        index_rate=0.75
    ),
    4: dict(
        hop_ms=20,
        vad_rms=0.0022, vad_abs=0.0065, hold=1,
        agc_target=0.015, agc_max=1.6,
        overlap=1200, search=600, max_step=320,
        extra_48k=1920,
        output_gain=0.7, limit=0.95,
        f0_up_key=0,
        index_rate=0.75
    ),
    5: dict(
        hop_ms=20,
        vad_rms=0.0010, vad_abs=0.0045, hold=8,
        agc_target=0.0, agc_max=0.0,
        overlap=4096, search=576, max_step=256,
        extra_48k=1920,
        output_gain=0.60, limit=0.95,
        f0_up_key=0
    ),
    # w-okada style low-latency real-time stabilization preset
    6: dict(
        hop_ms=20,
        vad_rms=0.0008, vad_abs=0.0032, hold=8,
        agc_target=0.0, agc_max=0.0,
        overlap=1920, search=0, max_step=128,
        extra_48k=3840,
        output_gain=0.70, limit=0.95,
        f0_up_key=0,
    ),
}

# =========================
# Model Info
# =========================
@dataclass
class ModelInfo:
    id: str
    label: str
    model_path: str
    index_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "has_index": self.index_path is not None,
            "f0_up_key": self.meta.get("f0_up_key", 0),
        }

# =========================
# Model Scanner
# =========================
def scan_models(root: str) -> Dict[str, ModelInfo]:
    models = {}
    if not os.path.isdir(root):
        print(f"[MODEL] Warning: models root not found: {root}")
        return models

    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue

        model_path = None
        for fname in os.listdir(folder):
            lower = fname.lower()
            if lower.endswith(".onnx") or lower.endswith(".pth"):
                model_path = os.path.join(folder, fname)
                break

        if model_path is None:
            continue

        meta = {}
        for fname in os.listdir(folder):
            if fname.lower().endswith(".json"):
                meta_path = os.path.join(folder, fname)
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    break
                except Exception as e:
                    print(f"[MODEL] Warning: failed to read {fname} for {name}: {e}")

        index_path = None
        for idx_name in ["index.bin", "added.index", f"{name}.index"]:
            p = os.path.join(folder, idx_name)
            if os.path.isfile(p):
                index_path = p
                break

        label = meta.get("label", name)
        models[name] = ModelInfo(
            id=name, label=label, model_path=model_path,
            index_path=index_path, meta=meta,
        )
        print(f"[MODEL] Found: {name} -> {label} (index={'yes' if index_path else 'no'})")

    return models

# =========================
# Utilities
# =========================
def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def make_cos2_windows(overlap: int):
    if overlap <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    t = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
    prev = (np.cos(t) ** 2).astype(np.float32, copy=False)
    cur = (np.cos(t[::-1]) ** 2).astype(np.float32, copy=False)
    return prev, cur

def sola_find_offset(prev_buf: np.ndarray, cur_audio: np.ndarray, overlap: int, search: int) -> int:
    cur_audio = np.asarray(cur_audio, dtype=np.float32).reshape(-1)
    prev_buf = np.asarray(prev_buf, dtype=np.float32).reshape(-1)

    N = int(cur_audio.shape[0])
    overlap = int(overlap)
    search = int(max(0, search))

    if overlap <= 0 or N <= overlap + 1 or prev_buf.size != overlap:
        return 0

    scan_len = min(N, overlap + search + 1)
    if scan_len <= overlap:
        return 0

    a = cur_audio[:scan_len].astype(np.float32, copy=False)
    b = prev_buf.astype(np.float32, copy=False)

    cor_nom = np.convolve(a, np.flip(b), mode="valid")
    cor_den = np.sqrt(
        np.convolve(a * a, np.ones(overlap, dtype=np.float32), mode="valid") + 1e-3
    )
    return int(np.argmax(cor_nom / cor_den))

def sola_apply_offset(
    prev_buf: np.ndarray,
    cur_audio: np.ndarray,
    overlap: int,
    offset: int,
    cur_strength: np.ndarray,
) -> np.ndarray:
    cur_audio = np.asarray(cur_audio, dtype=np.float32).reshape(-1)
    prev_buf = np.asarray(prev_buf, dtype=np.float32).reshape(-1)

    N = int(cur_audio.shape[0])
    overlap = int(overlap)
    offset = int(max(0, offset))

    out = np.zeros(N, dtype=np.float32)
    base = cur_audio[offset:]
    n = min(base.shape[0], N)

    if n > 0:
        out[:n] = base[:n]

    m = min(overlap, n)
    if m > 0:
        out[:m] *= cur_strength[:m]
        out[:m] += prev_buf[:m]

    return out

def _read_onnx_feats_dim(onnx_path: str) -> int | None:
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        for inp in sess.get_inputs():
            name = (inp.name or "").lower()
            if "feats" in name:
                shape = inp.shape
                if shape and isinstance(shape[-1], int):
                    return int(shape[-1])
    except Exception:
        pass
    return None

def _get_use_final_proj(model_path: str) -> bool:
    if not model_path.lower().endswith(".onnx"):
        return True
    exp = _read_onnx_feats_dim(model_path)
    if exp == 768:
        return False
    elif exp == 256:
        return True
    return True

def _read_pth_emb_dim(model_path: str) -> int | None:
    if not model_path.lower().endswith(".pth"):
        return None
    try:
        cpt = torch.load(model_path, map_location="cpu")
        weight = cpt.get("weight", {})
        emb = weight.get("enc_p.emb_phone.weight", None)
        if emb is None:
            return None
        shape = tuple(emb.shape)
        if len(shape) == 2 and isinstance(shape[1], int):
            return int(shape[1])
    except Exception:
        pass
    return None

def _detect_model_input_dim(model_path: str) -> int | None:
    lower = model_path.lower()
    if lower.endswith(".onnx"):
        return _read_onnx_feats_dim(model_path)
    if lower.endswith(".pth"):
        return _read_pth_emb_dim(model_path)
    return None

# =========================
# Session State
# =========================
@dataclass
class SessionState:
    # Ring buffer: accumulates hop input, tail extracted up to ctx_need
    ring48: np.ndarray               # 48kHz ring buffer
    ring_len: int = 0                # number of valid samples in ring buffer

    hop_samples: int = DEFAULT_HOP_SAMPLES

    pitchf: np.ndarray | None = None
    feature: np.ndarray | None = None
    last_out: np.ndarray = None
    sola_buffer: np.ndarray = None
    had_audio: bool = False
    prev_off: int = 0
    frame_idx: int = 0
    vad_hold: int = 0
    silence_run: int = 0
    slow_perf_run: int = 0
    prev_in: np.ndarray = None
    preset_id: int = 0
    params: dict = field(default_factory=lambda: PRESETS[0].copy())
    model_id: str = ""

# =========================
# GPU Production Guards
# =========================
def fatal_abort(msg: str):
    print(f"[FATAL] {msg}; aborting for production.")
    raise SystemExit(1)

def _extract_torch_model(inferencer):
    model = getattr(inferencer, "model", None)
    if model is not None and hasattr(model, "parameters"):
        return model
    return None

def _extract_onnx_providers(obj):
    if obj is None:
        return []
    # RMVPEOnnxPitchExtractor
    sess = getattr(obj, "onnx_session", None)
    if sess is not None and hasattr(sess, "get_providers"):
        return list(sess.get_providers())
    # OnnxRVCInferencer
    model = getattr(obj, "model", None)
    if model is not None and hasattr(model, "get_providers"):
        return list(model.get_providers())
    return []

def enforce_gpu_runtime_or_abort(engine=None):
    cuda_ok = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    ort_device = ort.get_device()
    ort_providers = ort.get_available_providers()

    print(f"[RUNTIME] torch.cuda.is_available() = {cuda_ok}")
    print(f"[RUNTIME] torch.cuda.get_device_name(0) = {cuda_name}")
    print(f"[RUNTIME] onnxruntime.get_device() = {ort_device}")
    print(f"[RUNTIME] onnxruntime providers = {ort_providers}")

    if not cuda_ok:
        fatal_abort("GPU not active (torch.cuda.is_available() is False)")

    if "CUDAExecutionProvider" not in ort_providers:
        fatal_abort("GPU not active (CUDAExecutionProvider missing in ORT providers)")

    if engine is not None:
        torch_model = _extract_torch_model(engine._inferencer)
        if torch_model is not None:
            try:
                param_dev = str(next(torch_model.parameters()).device)
            except StopIteration:
                param_dev = "unknown"
            print(f"[RUNTIME] net_g param device = {param_dev}")
            if not param_dev.startswith("cuda"):
                fatal_abort(f"GPU not active (net_g parameters on {param_dev})")
        else:
            print("[RUNTIME] net_g param device = n/a (onnx inferencer)")

        infer_providers = _extract_onnx_providers(engine._inferencer)
        if infer_providers:
            print(f"[RUNTIME] RVC inferencer providers = {infer_providers}")
            if "CUDAExecutionProvider" not in infer_providers:
                fatal_abort("GPU not active (RVC onnx inferencer is not using CUDAExecutionProvider)")

        rmvpe_providers = _extract_onnx_providers(engine.pitch)
        print(f"[RUNTIME] RMVPE providers = {rmvpe_providers}")
        if "CUDAExecutionProvider" not in rmvpe_providers:
            fatal_abort("GPU not active (RMVPE onnx session is not using CUDAExecutionProvider)")

# =========================
# RVC Engine (hop-based)
# =========================
class RvcEngine:
    def __init__(self):
        self.dev = torch.device(DEVICE)
        self._is_half = IS_HALF
        if DEVICE == "cpu":
            self._is_half = False

        # Auto-detect half precision support based on compute capability (before loading embedder)
        if self._is_half and self.dev.type == "cuda":
            try:
                major, minor = torch.cuda.get_device_capability(0)
                cc = major * 10 + minor
                if cc < 80:
                    print(f"[BOOT] GPU compute capability = {major}.{minor} (< 8.0). Falling back to float32 for stability.")
                    self._is_half = False
                else:
                    print(f"[BOOT] GPU compute capability = {major}.{minor}. float16 enabled.")
            except Exception:
                print("[WARN] Could not detect GPU compute capability. Falling back to float32.")
                self._is_half = False

        if EMBEDDER_TYPE == "contentvec":
            self.embedder = FairseqContentvec().loadModel(HUBERT_PATH, self.dev, self._is_half)
        else:
            self.embedder = FairseqHubert().loadModel(HUBERT_PATH, self.dev, self._is_half)

        # Force float32 if half precision is disabled (e.g. compute capability < 8.0)
        if not self._is_half and hasattr(self.embedder, 'model') and self.embedder.model is not None:
            self.embedder.model = self.embedder.model.float()
            print("[BOOT] Embedder forced to float32.")

        self.pitch = RMVPEOnnxPitchExtractor(
            file=RMVPE_MODEL_PATH,
            gpu=0 if DEVICE == "cuda" else -1
        )

        self._model_lock = threading.Lock()
        self._models: Dict[str, ModelInfo] = {}
        self._current_model_id: str = ""
        self._inferencer = None
        self._pipeline = None
        self._switching = False
        self._use_final_proj = True
        self._sola_win_cache = {}
        self._tensor_info_logged = False

        # Enable cuDNN benchmark for fixed-shape inference optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print(f"[BOOT] voice_changer root : {VOICE_CHANGER_DIR}")
        print(f"[BOOT] MODELS_ROOT        : {MODELS_ROOT}")
        print(f"[BOOT] HUBERT_PATH        : {HUBERT_PATH}  (exists={os.path.isfile(HUBERT_PATH)})")
        print(f"[BOOT] RMVPE_MODEL_PATH   : {RMVPE_MODEL_PATH}  (exists={os.path.isfile(RMVPE_MODEL_PATH)})")
        print(f"[BOOT] DEFAULT_MODEL_ID   : '{DEFAULT_MODEL_ID}' (empty = auto-select first)")
        print(f"[BOOT] DEVICE={DEVICE}  IS_HALF={self._is_half}  EMBEDDER={EMBEDDER_TYPE}")

        self._models = scan_models(MODELS_ROOT)

        if self._models:
            # Boot load order: DEFAULT_MODEL_ID → fallback to first compatible 256d model
            candidates = []
            if DEFAULT_MODEL_ID and DEFAULT_MODEL_ID in self._models:
                candidates.append(DEFAULT_MODEL_ID)
            for mid in self._models:
                if mid not in candidates:
                    candidates.append(mid)

            loaded = False
            for load_id in candidates:
                success, reason = self._load_model(load_id)
                if success:
                    loaded = True
                    break
                else:
                    print(f"[MODEL] Skipping {load_id} at boot: {reason}")
            if loaded:
                print(f"[BOOT] Loaded model ID    : {self._current_model_id}")
            else:
                print("[MODEL] Warning: no compatible 256d model found. Engine idle.")
        else:
            print("[MODEL] Warning: no models found in MODELS_ROOT. Engine idle.")

        print(f"[RVC-WS] Engine initialized. DEFAULT_HOP={DEFAULT_HOP_SAMPLES} ({DEFAULT_HOP_MS}ms), DEVICE={DEVICE}")
        print(f"[RVC-WS] Available models: {list(self._models.keys())}")

    def _load_model(self, model_id: str):
        """Returns (success: bool, reason: str).
        reason: 'ok' | 'model_not_found' | 'unsupported_768d' | 'unsupported_dim_N' | 'load_failed'
        """
        if model_id not in self._models:
            return False, "model_not_found"
        info = self._models[model_id]
        print(f"[MODEL] Loading: {model_id} ({info.model_path})")
        try:
            model_dim = _detect_model_input_dim(info.model_path)
            print(f"[MODEL] Detected input dim for {model_id}: {model_dim}")
            is_pth = info.model_path.lower().endswith(".pth")
            if model_dim == 768 and is_pth:
                print(f"[MODEL] Rejected 768-dim .pth model: {model_id}")
                return False, "unsupported_768d"
            if is_pth and model_dim not in (None, 256):
                print(f"[MODEL] Rejected unsupported dim={model_dim} .pth model: {model_id}")
                return False, f"unsupported_dim_{model_dim}"

            if info.model_path.lower().endswith(".onnx"):
                self._inferencer = OnnxRVCInferencer().loadModel(info.model_path, gpu=0)
            else:
                self._inferencer = RVCInferencer().loadModel(info.model_path, gpu=0)

            # Force GPU: ensure net_g is on cuda/eval/half consistently
            torch_model = _extract_torch_model(self._inferencer)
            if torch_model is not None:
                torch_model = torch_model.to(self.dev)
                if self._is_half:
                    torch_model = torch_model.half()
                else:
                    torch_model = torch_model.float()
                torch_model.eval()
                self._inferencer.model = torch_model

            self._use_final_proj = _get_use_final_proj(info.model_path)

            # Load FAISS index if available
            faiss_index = None
            if _FAISS_OK and info.index_path is not None and os.path.isfile(info.index_path):
                try:
                    faiss_index = faiss.read_index(info.index_path)
                    print(f"[MODEL] Index loaded: {info.index_path} (ntotal={faiss_index.ntotal})")
                except Exception as ie:
                    print(f"[MODEL] Warning: failed to load index for {model_id}: {ie}")
                    faiss_index = None
            else:
                if info.index_path is not None:
                    print(f"[MODEL] Index path set but faiss unavailable or file missing: {info.index_path}")

            self._pipeline = Pipeline(
                embedder=self.embedder,
                inferencer=self._inferencer,
                pitchExtractor=self.pitch,
                index=faiss_index,
                targetSR=MODEL_SR,
                device=self.dev,
                isHalf=self._is_half,
            )
            self._current_model_id = model_id
            print(f"[MODEL] Loaded: {model_id} (use_final_proj={self._use_final_proj})")
            enforce_gpu_runtime_or_abort(self)
            return True, "ok"
        except Exception as e:
            print(f"[MODEL] Failed to load {model_id}: {e}")
            traceback.print_exc()
            return False, "load_failed"

    def switch_model(self, model_id: str, st: SessionState):
        """Returns (success: bool, reason: str)."""
        with self._model_lock:
            if model_id == self._current_model_id:
                return True, "ok"
            if model_id not in self._models:
                return False, "model_not_found"
            self._switching = True
            try:
                self._reset_session_state(st)
                success, reason = self._load_model(model_id)
                if success:
                    st.model_id = model_id
                    meta = self._models[model_id].meta
                    if "f0_up_key" in meta:
                        st.params["f0_up_key"] = int(meta["f0_up_key"])
                return success, reason
            finally:
                self._switching = False

    def _reset_session_state(self, st: SessionState):
        st.pitchf = None
        st.feature = None
        st.had_audio = False
        st.prev_off = 0
        st.vad_hold = 0
        st.silence_run = 0
        st.slow_perf_run = 0
        st.ring_len = 0
        if st.ring48 is not None:
            st.ring48.fill(0.0)
        if st.prev_in is not None:
            st.prev_in.fill(0.0)
        if st.sola_buffer is not None:
            st.sola_buffer.fill(0.0)
        if st.last_out is not None:
            st.last_out.fill(0.0)

    def refresh_models(self) -> Dict[str, ModelInfo]:
        with self._model_lock:
            self._models = scan_models(MODELS_ROOT)
            return self._models

    def get_models(self) -> Dict[str, ModelInfo]:
        return self._models

    def get_current_model_id(self) -> str:
        return self._current_model_id

    def _get_sola_windows(self, overlap: int):
        if overlap not in self._sola_win_cache:
            self._sola_win_cache[overlap] = make_cos2_windows(overlap)
        return self._sola_win_cache[overlap]

    @property
    def pipeline(self):
        return self._pipeline

    # =========================================================
    # Core method: convert_hop() — called every hop
    # =========================================================
    def convert_hop(self, st: SessionState, x48_hop: np.ndarray, in_backlog: int = 0, out_backlog: int = 0, in_drops: int = 0) -> np.ndarray:
        """
        매 hop(기본 960 samples = 20ms @ 48kHz)마다 호출.
        1) hop을 링버퍼에 push
        2) ctx_need 만큼 tail을 추출해 RVC 추론
        3) SOLA로 경계 정렬
        4) hop_samples만 반환
        """
        x = np.asarray(x48_hop, dtype=np.float32).reshape(-1)
        hop = int(x.size)
        if hop <= 0:
            hop = st.hop_samples
            x = np.zeros((hop,), dtype=np.float32)

        with self._model_lock:
            if self._switching or self._pipeline is None:
                return np.zeros(hop, dtype=np.float32)
            pipeline = self._pipeline
            use_final_proj = self._use_final_proj

        st.frame_idx += 1
        t0 = time.perf_counter()
        t_pitch_ms = 0.0
        t_feature_ms = 0.0
        t_infer_ms = 0.0
        t_resample_ms = 0.0
        t_sola_ms = 0.0
        # block_frame based: use received floatCount as hop size

        # --- Read parameters ---
        p = st.params or PRESETS[0]
        VAD_RMS_TH = float(p.get("vad_rms", 0.0012))
        VAD_ABS_TH = float(p.get("vad_abs", 0.0045))
        HOLD_FRAMES = int(p.get("hold", 2))
        AGC_TARGET = float(p.get("agc_target", 0.0))
        AGC_MAX = float(p.get("agc_max", 0.0))
        OVERLAP = int(p.get("overlap", 960))
        SEARCH = int(p.get("search", int(0.012 * IN_SR)))
        MAX_STEP = int(p.get("max_step", 256))
        EXTRA = int(p.get("extra_48k", 1920))
        F0_UP_KEY = int(np.clip(int(p.get("f0_up_key", 0)), -12, 12))

        # Calculate ctx_need: minimum context to pass to RVC
        ctx_need = max(MIN_CTX_48K, hop + EXTRA + OVERLAP + SEARCH)

        # Ensure ring buffer capacity
        ring_cap = st.ring48.shape[0]
        if ctx_need > ring_cap:
            # Expand ring buffer (rare)
            new_ring = np.zeros(ctx_need * 2, dtype=np.float32)
            if st.ring_len > 0:
                new_ring[:st.ring_len] = st.ring48[:st.ring_len]
            st.ring48 = new_ring
            ring_cap = new_ring.shape[0]

        # Push hop into ring buffer
        if st.ring_len + hop > ring_cap:
            # Shift left, discard oldest data
            keep = ring_cap - hop
            if keep > 0 and st.ring_len > keep:
                st.ring48[:keep] = st.ring48[st.ring_len - keep:st.ring_len]
                st.ring_len = keep
            elif keep <= 0:
                st.ring_len = 0

        st.ring48[st.ring_len:st.ring_len + hop] = x
        st.ring_len += hop

        # Guardrails
        OVERLAP = max(0, min(OVERLAP, hop))
        SEARCH = max(0, SEARCH)
        MAX_STEP = max(0, MAX_STEP)

        # --- VAD ---
        absmax_in = float(np.max(np.abs(x)))
        r_in = rms(x)
        is_speech = (r_in > VAD_RMS_TH) or (absmax_in > VAD_ABS_TH)

        if is_speech:
            st.vad_hold = HOLD_FRAMES
            st.silence_run = 0
            gate_state = "speech"
        elif st.vad_hold > 0:
            st.vad_hold -= 1
            st.silence_run = 0
            gate_state = "hangover"
        else:
            st.silence_run += 1
            gate_state = "silence"

        if st.frame_idx % 25 == 0:
            print(f"[VAD] frame={st.frame_idx} gate={gate_state} hold={st.vad_hold} rms={r_in:.6f} hop={hop}")

        # --- Silence handling ---
        if gate_state == "silence":
            if st.had_audio:
                fade_len = min(int(FADE_MS * IN_SR / 1000.0), hop)
                y = np.zeros(hop, np.float32)
                if fade_len > 0 and st.last_out is not None and st.last_out.size >= fade_len:
                    tail = st.last_out[-fade_len:].copy()
                    t = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    w = 0.5 * (1.0 + np.cos(np.pi * t))
                    y[:fade_len] = tail * w
                st.had_audio = False
                st.prev_off = 0
                st.pitchf = None  # Reset pitchf on silence
                if st.sola_buffer is not None and st.sola_buffer.size > 0:
                    st.sola_buffer.fill(0.0)
                st.last_out = y
                return y

            y = np.zeros(hop, np.float32)
            if st.silence_run >= 5:
                st.prev_off = 0
                st.pitchf = None  # Reset pitchf on sustained silence
                if st.sola_buffer is not None and st.sola_buffer.size > 0:
                    st.sola_buffer.fill(0.0)
            st.last_out = y
            return y

        # hangover: still treated as speech, continue RVC inference
        # --- Hangover fast-exit: prevent trailing hiss artifact ---
        # Continuing RVC inference during hold can convert residual noise into audible hiss.
        if gate_state == "hangover":
            # (tunable) margin ratio to decide immediate hangover cutoff
            HANGOVER_CUT = 0.85  # adjust between 0.7~0.95 to taste
            if (r_in < VAD_RMS_TH * HANGOVER_CUT) and (absmax_in < VAD_ABS_TH * HANGOVER_CUT):
                # End of speech: fade-out on first frame, silence after
                fade_len = min(int(FADE_MS * IN_SR / 1000.0), hop)
                y = np.zeros(hop, np.float32)

                if st.had_audio and fade_len > 0 and st.last_out is not None and st.last_out.size >= fade_len:
                    tail = st.last_out[-fade_len:].copy()
                    t = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    w = 0.5 * (1.0 + np.cos(np.pi * t))  # 1->0
                    y[:fade_len] = tail * w

                st.had_audio = False
                st.vad_hold = 0
                st.prev_off = 0
                st.pitchf = None  # Reset pitchf on hangover exit
                if st.sola_buffer is not None and st.sola_buffer.size > 0:
                    st.sola_buffer.fill(0.0)
                st.last_out = y
                return y

        # --- AGC ---
        # Extract ctx from ring buffer (most recent ctx_need samples)
        actual_ctx = min(st.ring_len, ctx_need)
        x_ctx = st.ring48[st.ring_len - actual_ctx:st.ring_len].copy()

        if AGC_TARGET > 0.0 and AGC_MAX > 0.0:
            r_agc = rms(x_ctx)
            if r_agc > 1e-6:
                g = AGC_TARGET / r_agc
                g = float(np.clip(g, 0.2, AGC_MAX))
                x_ctx = x_ctx * g
            x_ctx = np.tanh(x_ctx * 1.5) / np.tanh(1.5)
        else:
            peak = float(np.max(np.abs(x_ctx)) + 1e-12)
            if peak > 0.98:
                x_ctx = x_ctx * (0.98 / peak)
            x_ctx = np.clip(x_ctx, -1.0, 1.0).astype(np.float32, copy=False)

        # --- Resample 48kHz → 16kHz for RVC ---
        t_rs_in0 = time.perf_counter()
        x16 = resample_poly(x_ctx, up=1, down=3).astype(np.float32, copy=False)
        t_resample_ms += (time.perf_counter() - t_rs_in0) * 1000.0
        # Ensure device/dtype consistency: create CUDA tensor directly each hop
        audio_dtype = torch.float16 if (self._is_half and self.dev.type == "cuda") else torch.float32
        audio_t = torch.as_tensor(x16, dtype=audio_dtype, device=self.dev)

        p_len = max(1, (len(x16) // WINDOW) + 1)

        if st.pitchf is None or st.pitchf.ndim != 1 or st.pitchf.shape[0] != p_len:
            st.pitchf = np.zeros((p_len,), dtype=np.float32)
        if st.feature is None or st.feature.ndim != 2 or st.feature.shape[0] != p_len:
            st.feature = np.zeros((p_len, 1), dtype=np.float32)

        # --- RVC inference ---
        t_inf0 = time.perf_counter()
        try:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=(self.dev.type == "cuda" and self._is_half)):
                    y_t, pitchf_buf, feats_buf = pipeline.exec(
                        sid=SID,
                        audio=audio_t,
                        pitchf=st.pitchf,
                        feature=st.feature,
                        f0_up_key=int(st.params.get("f0_up_key", F0_UP_KEY)),
                        index_rate=float(st.params.get("index_rate", INDEX_RATE)),
                        if_f0=IF_F0,
                        silence_front=SILENCE_FRONT,
                        embOutputLayer=EMB_OUTPUT_LAYER,
                        useFinalProj=use_final_proj,
                        repeat=REPEAT,
                        protect=PROTECT,
                        out_size=None,
                    )
        except Exception as _half_err:
            if "half precision" in str(_half_err).lower() or "HalfPrecision" in type(_half_err).__name__:
                print(f"[WARN] GPU does not support half precision. Switching to float32 and retrying...")
                self._is_half = False
                audio_t = audio_t.float()
                with torch.inference_mode():
                    y_t, pitchf_buf, feats_buf = pipeline.exec(
                        sid=SID,
                        audio=audio_t,
                        pitchf=st.pitchf,
                        feature=st.feature,
                        f0_up_key=int(st.params.get("f0_up_key", F0_UP_KEY)),
                        index_rate=float(st.params.get("index_rate", INDEX_RATE)),
                        if_f0=IF_F0,
                        silence_front=SILENCE_FRONT,
                        embOutputLayer=EMB_OUTPUT_LAYER,
                        useFinalProj=use_final_proj,
                        repeat=REPEAT,
                        protect=PROTECT,
                        out_size=None,
                    )
            else:
                raise
        t_infer_ms = (time.perf_counter() - t_inf0) * 1000.0

        # One-time log: input/model dtype+device and GPU memory summary
        if not self._tensor_info_logged:
            model_dtype = "n/a"
            model_device = "n/a"
            torch_model = _extract_torch_model(self._inferencer)
            if torch_model is not None:
                try:
                    p0 = next(torch_model.parameters())
                    model_dtype = str(p0.dtype)
                    model_device = str(p0.device)
                except StopIteration:
                    pass
            print(f"[RUNTIME] input tensor dtype/device = {audio_t.dtype}/{audio_t.device}")
            print(f"[RUNTIME] model param dtype/device = {model_dtype}/{model_device}")
            if self.dev.type == "cuda":
                alloc_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
                reserve_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
                print(f"[RUNTIME] cuda memory allocated={alloc_mb:.1f}MB reserved={reserve_mb:.1f}MB")
            self._tensor_info_logged = True

        # --- Update f0/feature buffers ---
        t_pitch0 = time.perf_counter()
        if pitchf_buf is not None:
            f0 = (
                pitchf_buf.detach().cpu().numpy()
                if hasattr(pitchf_buf, "detach")
                else np.asarray(pitchf_buf)
            ).astype(np.float32).reshape(-1)
            st.pitchf = f0
        t_pitch_ms = (time.perf_counter() - t_pitch0) * 1000.0

        t_feat0 = time.perf_counter()
        if feats_buf is not None:
            st.feature = (
                feats_buf.detach().cpu().numpy()
                if hasattr(feats_buf, "detach")
                else np.asarray(feats_buf)
            ).astype(np.float32)
            if st.feature.ndim == 1:
                st.feature = st.feature.reshape(-1, 1)
        t_feature_ms = (time.perf_counter() - t_feat0) * 1000.0

        # --- RVC output → 48kHz ---
        y_raw = y_t.detach().cpu().numpy() if hasattr(y_t, "detach") else np.asarray(y_t)
        y = np.squeeze(y_raw).reshape(-1)

        # Check dtype before float conversion (PTH models return int16)
        if y.dtype == np.int16 or y.dtype == np.int32:
            y = y.astype(np.float32) / 32768.0
        else:
            y = y.astype(np.float32, copy=False)

        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        absmax_pre = float(np.max(np.abs(y)) + 1e-12)
        if absmax_pre > 1.2:
            y = y / absmax_pre
        y = np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)

        # Resample model_sr → input_sr
        if MODEL_SR != IN_SR:
            t_rs_out0 = time.perf_counter()
            y = resample_poly(y, up=6, down=5).astype(np.float32, copy=False)
            t_resample_ms += (time.perf_counter() - t_rs_out0) * 1000.0

        # Ensure long buffer for SOLA: prevents zero-pad corruption after offset
        sola_need = max(hop + OVERLAP + SEARCH, hop)
        if y.size >= sola_need:
            y_long = y[-sola_need:]
        else:
            y_long = np.pad(y, (sola_need - y.size, 0))

        # --- Output Gain ---
        OUTPUT_GAIN = float(np.clip(float(p.get("output_gain", 0.7)), 0.05, 2.0))
        if OUTPUT_GAIN > 0.0:
            y_long = (y_long * OUTPUT_GAIN).astype(np.float32, copy=False)

        # --- Limiter ---
        LIMIT = float(p.get("limit", 0.95))
        SOFT = bool(int(p.get("soft_limiter", 1)))
        DRIVE = float(p.get("soft_drive", 1.35))

        if LIMIT > 0.0:
            absmax = float(np.max(np.abs(y_long)) + 1e-12)
            if absmax > LIMIT:
                y_long = (y_long / absmax * LIMIT).astype(np.float32, copy=False)
                if SOFT and DRIVE > 0.0 and LIMIT > 1e-6:
                    yn = (y_long / LIMIT).astype(np.float32, copy=False)
                    y_long = (LIMIT * (np.tanh(yn * DRIVE) / np.tanh(DRIVE))).astype(np.float32, copy=False)

        y_long = y_long.astype(np.float32, copy=False)

        # --- SOLA crossfade (long buffer slice + OLA) ---
        t_sola0 = time.perf_counter()
        prev_strength, cur_strength = self._get_sola_windows(OVERLAP)

        if st.sola_buffer is None or st.sola_buffer.shape[0] != OVERLAP:
            st.sola_buffer = np.zeros((OVERLAP,), np.float32)

        # Skip SOLA offset search on low energy to avoid corrupting prev_off
        LOW_ENERGY_TH = max(VAD_RMS_TH * 2.0, 0.003)
        y_rms = rms(y_long[:hop])

        if y_rms < LOW_ENERGY_TH:
            off_raw = st.prev_off
        else:
            off_raw = sola_find_offset(
                prev_buf=st.sola_buffer,
                cur_audio=y_long,
                overlap=OVERLAP,
                search=SEARCH,
            )

        # Clamp to [0, SEARCH] range first (handles search=0 case)
        off_raw = int(np.clip(off_raw, 0, SEARCH))
        # Then softly limit relative to prev_off
        off = int(np.clip(off_raw, st.prev_off - MAX_STEP, st.prev_off + MAX_STEP))
        off = int(np.clip(off, 0, SEARCH))
        st.prev_off = off

        # Generate hop via offset slice (avoids zero-padding shift)
        y_view = y_long[off:off + hop]
        if y_view.size < hop:
            y = np.pad(y_view, (0, hop - y_view.size)).astype(np.float32, copy=False)
        else:
            y = y_view.astype(np.float32, copy=True)

        # overlap-add
        if OVERLAP > 0:
            y[:OVERLAP] = y[:OVERLAP] * cur_strength + st.sola_buffer
        # fade-in on first speech frame
        if not st.had_audio:
            fade_in = min(int(FADE_MS * IN_SR / 1000.0), hop)
            if fade_in > 0:
                t = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
                w = 0.5 * (1.0 - np.cos(np.pi * t))
                y[:fade_in] *= w

        t_sola_ms = (time.perf_counter() - t_sola0) * 1000.0

        dt_ms = (time.perf_counter() - t0) * 1000.0
        hop_budget_ms = hop * 1000.0 / IN_SR
        if t_infer_ms > hop_budget_ms or dt_ms > hop_budget_ms:
            st.slow_perf_run = getattr(st, "slow_perf_run", 0) + 1
        else:
            st.slow_perf_run = 0

        if st.frame_idx % PERF_LOG_EVERY == 0:
            print(
                f"[PERF] frame={st.frame_idx} hop={hop} ctx={actual_ctx} off={off} "
                f"pitch_ms={t_pitch_ms:.2f} feat_ms={t_feature_ms:.2f} infer_ms={t_infer_ms:.2f} "
                f"resample_ms={t_resample_ms:.2f} sola_ms={t_sola_ms:.2f} proc_ms={dt_ms:.2f}"
            )
            if getattr(st, "slow_perf_run", 0) >= 3:
                print(f"[WARN] sustained slow inference: infer_ms={t_infer_ms:.2f}, proc_ms={dt_ms:.2f}, hop_budget_ms={hop_budget_ms:.2f}, in_q={in_backlog}, out_q={out_backlog}, in_drops={in_drops}")

        if OVERLAP > 0:
            if off < SEARCH and (off - (SEARCH + OVERLAP)) >= 0:
                buf_org = y_long[off - (SEARCH + OVERLAP): off - SEARCH]
            else:
                buf_org = y_long[-OVERLAP:]
            if buf_org.size < OVERLAP:
                buf_org = np.pad(buf_org, (0, OVERLAP - buf_org.size))
            st.sola_buffer = (buf_org * prev_strength).astype(np.float32, copy=False)
        st.had_audio = True

        # Final output is exactly hop size
        if y.size != hop:
            if y.size > hop:
                y = y[:hop]
            else:
                y = np.pad(y, (0, hop - y.size))

        st.last_out = y
        return y

ENGINE: RvcEngine | None = None

# =========================
# WebSocket Handler
# =========================

async def handler(ws):
    global ENGINE

    PRESET_NAME_MAP = {
        "BALANCED": 0,
        "LOW_LATENCY": 2,
        "LOWLATENCY": 2,
        "STABLE": 3,
        "P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5,
    }

    # Alias map: various client-side keys → internal keys
    PARAM_ALIASES = {
        "chunk": "overlap",
        "chunk_size": "overlap",
        "extraConvertSize": "extra_48k",
        "extraDataLength": "extra_48k",
        "extra_samples": "extra_48k",
        "block_samples": "hop_samples",
    }

    def _ensure_int(v, default=0):
        try: return int(v)
        except: return int(default)

    def _ensure_float(v, default=0.0):
        try: return float(v)
        except: return float(default)

    def _ensure_state_buffers(st: SessionState):
        ov = _ensure_int(st.params.get("overlap", 0), 0)
        if ov < 0: ov = 0
        if ov == 0:
            st.sola_buffer = np.zeros((0,), np.float32)
            return
        if st.sola_buffer is None or int(getattr(st.sola_buffer, 'shape', [0])[0]) != ov:
            st.sola_buffer = np.zeros((ov,), np.float32)

    def _apply_preset(st: SessionState, pid: int):
        if pid not in PRESETS:
            raise ValueError(f"unknown preset id: {pid}")
        prev_hop = int(st.hop_samples)
        st.preset_id = int(pid)
        st.params = PRESETS[int(pid)].copy()
        # Also update hop_samples from preset
        hop_ms = int(st.params.get("hop_ms", DEFAULT_HOP_MS))
        st.hop_samples = IN_SR * hop_ms // 1000
        _ensure_state_buffers(st)
        if st.hop_samples != prev_hop:
            _reset_stream_state_for_hop_change(st)

    def _apply_params_patch(st: SessionState, patch: dict):
        if not isinstance(patch, dict):
            raise ValueError("patch must be object/dict")
        prev_hop = int(st.hop_samples)
        p = st.params if isinstance(st.params, dict) else PRESETS[0].copy()

        # Resolve aliases
        resolved = {}
        for k, v in patch.items():
            real_key = PARAM_ALIASES.get(k, k)
            resolved[real_key] = v

        for k, v in resolved.items():
            if k in ("hold", "overlap", "search", "max_step", "f0_up_key", "extra_48k", "hop_ms", "hop_samples"):
                p[k] = _ensure_int(v, p.get(k, 0))
            else:
                p[k] = _ensure_float(v, p.get(k, 0.0))

        # Handle hop size change
        if "hop_ms" in resolved:
            hop_ms = max(5, min(1000, int(p.get("hop_ms", DEFAULT_HOP_MS))))
            p["hop_ms"] = hop_ms
            hop_s = IN_SR * hop_ms // 1000
            hop_s = max(240, min(48000, hop_s))
            if hop_s % 960 != 0:
                raise ValueError("block_samples must be multiple of 960 (Unity step=960)")
            p["hop_samples"] = hop_s
            p["hop_ms"] = hop_s * 1000 // IN_SR
            st.hop_samples = hop_s
        elif "hop_samples" in resolved:
            hop_s = max(240, min(48000, int(p.get("hop_samples", DEFAULT_HOP_SAMPLES))))
            if hop_s % 960 != 0:
                raise ValueError("block_samples must be multiple of 960 (Unity step=960)")
            p["hop_samples"] = hop_s
            p["hop_ms"] = hop_s * 1000 // IN_SR
            st.hop_samples = hop_s

        p["vad_rms"] = max(0.0, float(p.get("vad_rms", 0.0)))
        p["vad_abs"] = max(0.0, float(p.get("vad_abs", 0.0)))
        p["hold"] = max(0, int(p.get("hold", 0)))
        p["agc_target"] = max(0.0, float(p.get("agc_target", 0.0)))
        p["agc_max"] = max(0.0, float(p.get("agc_max", 0.0)))
        p["overlap"] = max(0, int(p.get("overlap", 0)))
        p["search"] = max(0, int(p.get("search", 0)))
        p["max_step"] = max(0, int(p.get("max_step", 0)))
        p["extra_48k"] = max(0, int(p.get("extra_48k", 1920)))
        p["output_gain"] = max(0.1, min(1.5, float(p.get("output_gain", 0.7))))
        p["limit"] = max(0.0, min(1.0, float(p.get("limit", 0.95))))
        p["f0_up_key"] = max(-12, min(12, int(p.get("f0_up_key", 0))))
        p["index_rate"] = max(0.0, min(1.0, float(p.get("index_rate", 0.75))))

        st.params = p
        _ensure_state_buffers(st)
        if st.hop_samples != prev_hop:
            _reset_stream_state_for_hop_change(st)

    def _make_state_payload(st: SessionState):
        return {
            "model_id": st.model_id,
            "preset_id": int(st.preset_id),
            "hop_samples": int(st.hop_samples),
            "block_samples": int(st.hop_samples),
            "hop_ms": int(st.hop_samples * 1000 // IN_SR),
            "params": dict(st.params) if isinstance(st.params, dict) else {},
        }

    def _make_models_payload():
        models = ENGINE.get_models()
        return {
            "models": [m.to_dict() for m in models.values()],
            "current": ENGINE.get_current_model_id(),
        }

    if ENGINE is None:
        ENGINE = RvcEngine()

    base = PRESETS.get(0, {}).copy()
    hop_ms = int(base.get("hop_ms", DEFAULT_HOP_MS))
    hop_samples = IN_SR * hop_ms // 1000
    ov0 = int(base.get('overlap', 0) or 0)
    if ov0 < 0: ov0 = 0

    # Initialize ring buffer (2 seconds capacity)
    RING_CAP = IN_SR * 2

    st = SessionState(
        ring48=np.zeros((RING_CAP,), dtype=np.float32),
        ring_len=0,
        hop_samples=hop_samples,
        pitchf=None,
        feature=None,
        last_out=np.zeros((hop_samples,), dtype=np.float32),
        sola_buffer=np.zeros((ov0,), dtype=np.float32),
        had_audio=False,
        preset_id=0,
        params=base,
        model_id=ENGINE.get_current_model_id(),
    )

    def _reset_stream_state_for_hop_change(st: SessionState):
        # Reset stream state on hop change
        ENGINE._reset_session_state(st)
        st.last_out = np.zeros((st.hop_samples,), dtype=np.float32)
        _ensure_state_buffers(st)

    print(f"[RVC-WS] CLIENT CONNECTED (hop={hop_samples}, {hop_ms}ms)")

    IN_Q_MAX = 8
    OUT_Q_MAX = 8
    in_q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(maxsize=IN_Q_MAX)
    out_q: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(maxsize=OUT_Q_MAX)
    queue_stats = {"in_drops": 0}

    async def convert_loop():
        while True:
            seq_i, frame_i = await in_q.get()
            if seq_i is None:
                in_q.task_done()
                break
            try:
                in_b = in_q.qsize()
                out_b = out_q.qsize()
                out_frame_i = ENGINE.convert_hop(st, frame_i, in_backlog=in_b, out_backlog=out_b, in_drops=queue_stats["in_drops"])
                if out_q.full():
                    try:
                        _ = out_q.get_nowait()
                        out_q.task_done()
                    except asyncio.QueueEmpty:
                        pass
                out_q.put_nowait((seq_i, out_frame_i))
            finally:
                in_q.task_done()

    convert_task = asyncio.create_task(convert_loop())

    def _clear_pending_queues():
        while True:
            try:
                _ = in_q.get_nowait()
                in_q.task_done()
            except asyncio.QueueEmpty:
                break
        while True:
            try:
                _ = out_q.get_nowait()
                out_q.task_done()
            except asyncio.QueueEmpty:
                break

    while True:
        try:
            msg = await ws.recv()

            if isinstance(msg, str):
                s_raw = msg.strip()

                if s_raw.startswith('{') and s_raw.endswith('}'):
                    print(f"[CTRL] JSON received: {s_raw[:200]}")
                    try:
                        req = json.loads(s_raw)
                        rtype = str(req.get('type', '')).strip()
                        rid = req.get('request_id', None)

                        if rtype == 'list_models':
                            ENGINE.refresh_models()
                            resp = {
                                'type': 'models',
                                'request_id': rid,
                                'ok': True,
                                **_make_models_payload(),
                            }
                            await ws.send(json.dumps(resp, ensure_ascii=False, separators=(',', ':')))
                            continue

                        if rtype == 'set_model':
                            model_id = req.get('model_id', '')
                            success, reason = ENGINE.switch_model(model_id, st)
                            resp = {
                                'type': 'ack',
                                'request_id': rid,
                                'ok': success,
                                'state': _make_state_payload(st),
                            }
                            if not success:
                                resp['error'] = reason
                            await ws.send(json.dumps(resp, ensure_ascii=False, separators=(',', ':')))
                            continue

                        prev_hop = int(st.hop_samples)

                        if rtype == 'set_preset':
                            pid = req.get('id', 0)
                            if isinstance(pid, str):
                                key = pid.strip().upper()
                                pid = PRESET_NAME_MAP.get(key)
                                if pid is None:
                                    raise ValueError(f"unknown preset name: {key}")
                            _apply_preset(st, int(pid))

                        elif rtype == 'set_params':
                            _apply_params_patch(st, req.get('patch', {}))

                        elif rtype == 'get_state':
                            pass

                        else:
                            raise ValueError(f"unknown control type: {rtype}")

                        if st.hop_samples != prev_hop:
                            _clear_pending_queues()

                        resp = {
                            'type': 'ack',
                            'request_id': rid,
                            'ok': True,
                            'state': _make_state_payload(st),
                        }
                        print(f"[CTRL] applied type={rtype} rid={rid} hop={st.hop_samples} preset={st.preset_id}")
                        await ws.send(json.dumps(resp, ensure_ascii=False, separators=(',', ':')))

                    except Exception as e:
                        resp = {'type': 'ack', 'request_id': None, 'ok': False, 'error': str(e)}
                        try:
                            await ws.send(json.dumps(resp, ensure_ascii=False, separators=(',', ':')))
                        except:
                            pass
                    continue

                s = s_raw.upper()
                if s.startswith('PSET'):
                    try:
                        prev_hop = int(st.hop_samples)
                        parts = s.split()
                        pid = int(parts[1]) if len(parts) > 1 else 0
                        _apply_preset(st, pid)
                        if st.hop_samples != prev_hop:
                            _clear_pending_queues()
                        await ws.send(f"OK PSET {pid}")
                        resp = {'type': 'ack', 'request_id': None, 'ok': True, 'state': _make_state_payload(st)}
                        await ws.send(json.dumps(resp, ensure_ascii=False, separators=(',', ':')))
                    except Exception as e:
                        resp = {'type': 'ack', 'request_id': None, 'ok': False, 'error': str(e)}
                        try:
                            await ws.send(json.dumps(resp, ensure_ascii=False, separators=(',', ':')))
                        except:
                            pass
                    continue

                continue

            if not isinstance(msg, (bytes, bytearray)) or len(msg) < 8:
                continue

            seq, count = struct.unpack_from('<II', msg, 0)

            # Validate block_frame range
            if count <= 0 or count > MAX_BLOCK_SAMPLES:
                if seq % 50 == 0:
                    print(f"[WARN] invalid block_frame count={count}. DROPPING frame seq={seq}.")
                continue

            # Fixed block mode: drop frame if count mismatches expected hop
            expected = int(st.hop_samples)
            if count != expected:
                if seq % 50 == 0:
                    print(f"[WARN] block_frame mismatch count={count} expected={expected}. DROPPING seq={seq}")
                continue

            need_bytes = 8 + count * 4
            if len(msg) < need_bytes:
                continue

            frame = np.frombuffer(msg, dtype=np.float32, offset=8, count=count).astype(np.float32, copy=True)

            # Enqueue input (drop oldest hop if queue is full)
            if in_q.full():
                try:
                    _ = in_q.get_nowait()
                    in_q.task_done()
                    queue_stats["in_drops"] += 1
                except asyncio.QueueEmpty:
                    pass
            in_q.put_nowait((seq, frame))

            # Dequeue output (return silence if empty, e.g. during warmup)
            out_seq = seq
            expected_hop = count
            try:
                out_seq, out_frame = out_q.get_nowait()
                out_q.task_done()
                if out_frame.size != expected_hop:
                    if out_frame.size > expected_hop:
                        out_frame = out_frame[:expected_hop]
                    else:
                        out_frame = np.pad(out_frame, (0, expected_hop - out_frame.size)).astype(np.float32, copy=False)
            except asyncio.QueueEmpty:
                out_frame = np.zeros((expected_hop,), dtype=np.float32)

            if seq % 25 == 0:
                print(
                    f"[SRV] seq={seq} send_seq={out_seq} hop={expected_hop} "
                    f"in_q={in_q.qsize()} out_q={out_q.qsize()} drops={queue_stats['in_drops']} "
                    f"in_rms={rms(frame):.4f} out_rms={rms(out_frame):.4f}"
                )

                        # Response is also block_frame length
            out = bytearray(8 + expected_hop * 4)
            struct.pack_into('<II', out, 0, out_seq, expected_hop)
            out[8:] = out_frame[:expected_hop].tobytes()
            await ws.send(out)

        except websockets.exceptions.ConnectionClosed:
            break
        except Exception as e:
            print('[RVC-WS] handler error:', repr(e))
            print(traceback.format_exc())
            break

    try:
        if not convert_task.done():
            await in_q.put((None, None))
            await convert_task
    except Exception:
        pass

    print('[RVC-WS] CLIENT DISCONNECTED')

async def main():
    global ENGINE
    enforce_gpu_runtime_or_abort()
    if ENGINE is None:
        ENGINE = RvcEngine()
    print(f"[RVC-WS] Server — hop-based architecture")
    print(f"[RVC-WS] listening ws://{WS_HOST}:{WS_PORT}")
    print(f"[RVC-WS] Available presets: {list(PRESETS.keys())}")
    async with websockets.serve(handler, WS_HOST, WS_PORT, max_size=2**22):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())