# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import time
import urllib.request
import urllib.error


@dataclass
class OpenAICompatConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "qwen25-7b"

    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: Optional[float] = None

    system_prompt: str = "You are a helpful assistant."
    request_timeout_s: float = 120.0
    max_retries: int = 2
    retry_sleep_s: float = 1.0


class OpenAICompatLLM:
    """
    OpenAI-compatible HTTP backend for vLLM server.

    Required interface:
        text = llm.generate(prompt)

    This class does NOT import vllm. It only sends HTTP requests to:
        {base_url}/chat/completions
    """

    def __init__(self, cfg: OpenAICompatConfig):
        self.cfg = cfg
        self.url = self.cfg.base_url.rstrip("/") + "/chat/completions"

    def generate(self, prompt: str) -> str:
        prompt = str(prompt or "")

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "max_tokens": int(self.cfg.max_new_tokens),
        }

        if self.cfg.repetition_penalty is not None:
            payload["repetition_penalty"] = float(self.cfg.repetition_penalty)

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }

        last_err = None
        for attempt in range(int(self.cfg.max_retries) + 1):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=float(self.cfg.request_timeout_s)) as resp:
                    raw = resp.read().decode("utf-8")
                obj = json.loads(raw)

                choices = obj.get("choices") or []
                if not choices:
                    return ""

                msg = choices[0].get("message") or {}
                return str(msg.get("content") or "").strip()

            except Exception as e:
                last_err = e
                if attempt < int(self.cfg.max_retries):
                    time.sleep(float(self.cfg.retry_sleep_s))

        raise RuntimeError(f"OpenAI-compatible LLM request failed: {last_err}") from last_err
