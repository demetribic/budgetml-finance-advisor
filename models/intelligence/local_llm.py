"""
models/intelligence/local_llm.py — In-process local LLM via llama-cpp-python.

Loads a GGUF model once (lazy, thread-safe) and exposes generate() / classify()
with the same call signature as the Ollama helpers in api/app.py.

The model is auto-downloaded from HuggingFace on first use if the file does not
exist locally.  Override any default with environment variables:

    LOCAL_LLM_MODEL_PATH   — absolute path to a local .gguf file (skip download)
    LOCAL_LLM_REPO_ID      — HF repo id  (default: bartowski/Qwen2.5-1.5B-Instruct-GGUF)
    LOCAL_LLM_FILENAME     — GGUF filename in that repo  (default: Q4_K_M variant)
    LOCAL_LLM_CTX          — context window tokens  (default: 2048)
    LOCAL_LLM_THREADS      — CPU threads for inference  (default: min(cpu_count, 8))
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_REPO_ID  = os.environ.get("LOCAL_LLM_REPO_ID",  "bartowski/Qwen2.5-1.5B-Instruct-GGUF")
_FILENAME = os.environ.get("LOCAL_LLM_FILENAME",  "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
_DEFAULT_PATH = (
    Path(__file__).parent.parent.parent / "models" / "saved" / "local_llm" / _FILENAME
)
_MODEL_PATH = Path(os.environ.get("LOCAL_LLM_MODEL_PATH", str(_DEFAULT_PATH)))
_N_CTX      = int(os.environ.get("LOCAL_LLM_CTX",     "4096"))
_N_THREADS  = int(os.environ.get("LOCAL_LLM_THREADS", str(min(os.cpu_count() or 4, 8))))

# Qwen2.5-Instruct chat template tokens
_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"

# ── Singleton state ───────────────────────────────────────────────────────────

_lock:      threading.Lock         = threading.Lock()
_llm:       object | None          = None   # llama_cpp.Llama instance
_llm_ready: bool | None            = None   # None=unchecked, False=failed, True=ok


# ── Internal helpers ──────────────────────────────────────────────────────────

def _download() -> bool:
    """Download the GGUF file from HuggingFace into _MODEL_PATH. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading local LLM: %s / %s → %s", _REPO_ID, _FILENAME, _MODEL_PATH)
        local = hf_hub_download(
            repo_id=_REPO_ID,
            filename=_FILENAME,
            local_dir=str(_MODEL_PATH.parent),
        )
        ok = Path(local).stat().st_size > 100_000  # sanity: >100 KB
        if ok:
            log.info("Download complete: %s", local)
        return ok
    except Exception as exc:
        log.warning("local_llm: download failed — %s", exc)
        return False


def _load() -> bool:
    """Ensure the model is loaded. Returns True when ready."""
    global _llm, _llm_ready
    with _lock:
        if _llm_ready is True:
            return True
        if _llm_ready is False:
            return False

        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError:
            log.warning("local_llm: llama-cpp-python not installed — run: pip install llama-cpp-python")
            _llm_ready = False
            return False

        if not _MODEL_PATH.exists():
            if not _download():
                _llm_ready = False
                return False

        try:
            log.info("Loading local LLM from %s (ctx=%d, threads=%d) …", _MODEL_PATH, _N_CTX, _N_THREADS)
            _llm = Llama(
                model_path=str(_MODEL_PATH),
                n_ctx=_N_CTX,
                n_threads=_N_THREADS,
                n_gpu_layers=0,   # CPU-only; safe on login/compute nodes without GPU alloc
                verbose=False,
            )
            _llm_ready = True
            log.info("Local LLM ready.")
            return True
        except Exception as exc:
            log.warning("local_llm: failed to load model — %s", exc)
            _llm_ready = False
            return False


def _fmt(system: str, user: str) -> str:
    """Format a prompt using the Qwen2.5-Instruct chat template."""
    return (
        f"{_IM_START}system\n{system}{_IM_END}\n"
        f"{_IM_START}user\n{user}{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if the local LLM is loaded and ready to use."""
    if _llm_ready is True:
        return True
    return _load()


def generate(
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.6,
) -> str | None:
    """
    Free-form text generation.  Returns the generated string or None on failure.

    prompt     — the user-facing instruction / question
    max_tokens — upper bound on output tokens (not counting the prompt)
    temperature — sampling temperature; 0.0 = greedy / deterministic
    """
    if not _load() or _llm is None:
        return None
    if not prompt or not prompt.strip():
        return None
    formatted = _fmt(
        system="You are a helpful personal finance assistant. Be concise and specific.",
        user=prompt,
    )
    try:
        with _lock:
            out = _llm(  # type: ignore[operator]
                formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=[_IM_END, _IM_START],
                echo=False,
            )
        text = str(out["choices"][0]["text"]).strip()
        return text or None
    except Exception as exc:
        log.warning("local_llm.generate failed: %s", exc)
        return None


def classify(text: str, categories: list[str]) -> str | None:
    """
    Zero-shot classification.  Returns one of `categories` or None.

    Uses greedy decoding (temperature=0) and stops at the first newline so the
    model outputs only the category name.
    """
    if not _load() or _llm is None:
        return None
    if not text or not categories:
        return None
    cat_str = ", ".join(categories)
    formatted = _fmt(
        system="You are a transaction classifier. Reply with only the category name, nothing else.",
        user=(
            f"Classify this bank transaction into exactly one of: {cat_str}.\n"
            f"Transaction: {text.strip()}\n"
            f"Category:"
        ),
    )
    try:
        with _lock:
            out = _llm(  # type: ignore[operator]
                formatted,
                max_tokens=12,
                temperature=0.0,
                stop=[_IM_END, _IM_START, "\n"],
                echo=False,
            )
        raw = str(out["choices"][0]["text"]).strip().lower().rstrip(".")
        # Exact match first
        for c in categories:
            if c.lower() == raw:
                return c
        # Fuzzy: first category whose name appears anywhere in the response
        for c in categories:
            if c.lower() in raw:
                return c
        return None
    except Exception as exc:
        log.warning("local_llm.classify failed: %s", exc)
        return None
