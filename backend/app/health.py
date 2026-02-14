from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import requests

from app.config import settings, Settings, ensure_directories

logger = logging.getLogger(__name__)


HealthResult = Tuple[bool, str | None]


def check_ollama_running(cfg: Settings | None = None) -> HealthResult:
    cfg = cfg or settings
    url = cfg.ollama_base_url.rstrip("/") + "/"
    try:
        resp = requests.get(url, timeout=cfg.health_timeout_seconds)
        if resp.status_code >= 200 and resp.status_code < 500:
            return True, "reachable"
        return False, f"Ollama responded with HTTP {resp.status_code}"
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        return False, "Ollama not running"


def _fetch_ollama_tags(cfg: Settings) -> Dict:
    url = cfg.ollama_base_url.rstrip("/") + "/api/tags"
    resp = requests.get(url, timeout=cfg.health_timeout_seconds)
    resp.raise_for_status()
    return resp.json()


def check_ollama_model_present(cfg: Settings | None = None) -> HealthResult:
    cfg = cfg or settings
    try:
        ok, msg = check_ollama_running(cfg)
        if not ok:
            return False, msg
        data = _fetch_ollama_tags(cfg)
        models = {m.get("name") for m in data.get("models", [])}
        if cfg.ollama_model in models:
            return True, cfg.ollama_model
        return (
            False,
            f"LLM model not found in Ollama. Please run: ollama pull {cfg.ollama_model}",
        )
    except Exception as exc:
        logger.warning("Ollama model check failed: %s", exc)
        return False, f"Could not verify model in Ollama: {exc}"


def check_sentence_transformer_present() -> HealthResult:
    from sentence_transformers import SentenceTransformer  # local import

    try:
        # Try a very lightweight instantiation; rely on internal caching.
        SentenceTransformer(settings.embedding_model_name)
        return True, settings.embedding_model_name
    except Exception as exc:
        logger.warning("Embedding model check failed: %s", exc)
        return False, "Embedding model missing"


def check_tesseract_present() -> HealthResult:
    try:
        import pytesseract  # type: ignore

        if getattr(pytesseract, "get_tesseract_version", None) is None:
            return False, "Tesseract not installed"
        pytesseract.get_tesseract_version()
        return True, "installed"
    except Exception as exc:
        logger.warning("Tesseract check failed: %s", exc)
        return False, "Tesseract not installed"


def check_vector_db_path_writable() -> HealthResult:
    try:
        ensure_directories()
        path: Path = settings.chroma_db_path
        test_file = path / ".writable_check"
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
        return True, str(path)
    except Exception as exc:
        logger.warning("Vector DB path check failed: %s", exc)
        return False, "Vector DB error"


def collect_health_snapshot() -> Dict[str, Dict[str, object]]:
    """
    Aggregate all health checks into a single dictionary for /health and logs.
    """
    checks = {
        "ollama": check_ollama_running(),
        "llm_model": check_ollama_model_present(),
        "embedding_model": check_sentence_transformer_present(),
        "tesseract": check_tesseract_present(),
        "vector_db": check_vector_db_path_writable(),
    }

    snapshot: Dict[str, Dict[str, object]] = {}
    for name, (ok, msg) in checks.items():
        snapshot[name] = {"ok": bool(ok), "message": msg}
    return snapshot

