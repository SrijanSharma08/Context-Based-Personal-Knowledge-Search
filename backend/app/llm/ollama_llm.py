from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

import requests

from app.config import Settings, settings
from app.health import check_ollama_model_present
from app.llm.llm_interface import LLMClient
from app.errors import (
    OllamaUnavailable,
    OllamaModelMissing,
    OllamaRequestFailed,
)

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """
    LLM client that talks to a local Ollama server over HTTP.
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self._base_url = cfg.ollama_base_url.rstrip("/")

    def _build_prompt(
        self, context: str, question: str, history: Optional[List[Dict]] = None
    ) -> str:
        history = history or []
        history_text = ""
        if history:
            conv_lines = []
            for item in history[-10:]:
                role = item.get("role", "user")
                content = item.get("content", "")
                conv_lines.append(f"{role.upper()}: {content}")
            history_text = "\n".join(conv_lines)

        prompt = (
            "You are a helpful assistant for a personal knowledge organizer.\n"
            "You must answer **only** based on the provided context.\n"
            "If the answer cannot be found in the context, reply explicitly that "
            '"the answer is not found in the provided documents.".\n\n'
        )
        if history_text:
            prompt += f"Conversation history:\n{history_text}\n\n"
        prompt += "Context:\n"
        prompt += context
        prompt += "\n\nQuestion:\n"
        prompt += question
        prompt += "\n\nAnswer:"
        return prompt

    def generate_answer(
        self, context: str, question: str, history: Optional[List[dict]] = None
    ) -> str:
        # Validate Ollama and model availability before each request
        ok, msg = check_ollama_model_present(self._cfg)
        if not ok:
            if msg == "Ollama not running":
                raise OllamaUnavailable(msg)
            if msg and msg.startswith("LLM model not found in Ollama"):
                raise OllamaModelMissing(msg)
            raise OllamaRequestFailed(msg or "Failed to validate Ollama model")

        url = f"{self._base_url}/api/generate"
        prompt = self._build_prompt(context, question, history)

        payload = {
            "model": self._cfg.ollama_model,
            "prompt": prompt,
            "temperature": self._cfg.ollama_temperature,
            "max_tokens": self._cfg.ollama_max_tokens,
            "stream": False,
        }

        timeout = self._cfg.ollama_timeout_seconds
        start = time.time()
        logger.info(
            "llm_request_start",
            extra={"model": self._cfg.ollama_model, "timeout": timeout},
        )
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            elapsed = time.time() - start
            logger.error(
                "llm_request_timeout",
                extra={"model": self._cfg.ollama_model, "elapsed": elapsed},
            )
            raise OllamaRequestFailed("Ollama request timed out") from exc
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "llm_request_failed",
                extra={"model": self._cfg.ollama_model, "elapsed": elapsed},
            )
            raise OllamaUnavailable(f"Ollama request failed: {exc}") from exc

        elapsed = time.time() - start
        logger.info(
            "llm_request_end",
            extra={
                "model": self._cfg.ollama_model,
                "elapsed": elapsed,
                "status_code": resp.status_code,
            },
        )

        if resp.status_code != 200:
            raise OllamaRequestFailed(
                f"Ollama error {resp.status_code}: {resp.text}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return resp.text

        return data.get("response") or data.get("text") or ""

