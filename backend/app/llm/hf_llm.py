from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import Settings
from app.llm.llm_interface import LLMClient

logger = logging.getLogger(__name__)


class HuggingFaceClient(LLMClient):
    """
    LLM client using a local HuggingFace Transformers causal LM.
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        logger.info("Loading HuggingFace model '%s'", cfg.hf_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_name)
        self._model = AutoModelForCausalLM.from_pretrained(cfg.hf_model_name)

        self._device = "cuda" if cfg.gpu_available else "cpu"
        self._model.to(self._device)

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
        prompt = self._build_prompt(context, question, history)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._cfg.hf_max_new_tokens,
                do_sample=True,
                temperature=self._cfg.hf_temperature,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return generated.strip()

