# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Dict
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _dtype_from_str(s: str):
    s = (s or "").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    # fallback
    return torch.float16


class HFCausalLLM:
    """
    A lightweight HF causal-LM wrapper with a vLLM-like interface:
      - generate(prompts: List[str]) -> List[str]
    It also supports chat_template for Instruct models like Qwen2.5.
    Extra kwargs are ignored so you can keep old llm fields in yaml.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float16",
        use_chat_template: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        # generation defaults
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stop: Optional[Sequence[str]] = None,
        # loading
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        sdp_mode: str = "math",
        **_: Any,
    ):
        self.model_path = model
        self.tok_path = tokenizer or model
        self.device = device
        self.dtype = _dtype_from_str(dtype)
        self.use_chat_template = bool(use_chat_template)
        self.system_prompt = system_prompt

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.stop = list(stop) if stop else None

        # Make SDPA stable (align with your previous “math SDP” choice)
        if device.startswith("cuda"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

        self.tok = AutoTokenizer.from_pretrained(
            self.tok_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        # Causal LM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.dtype,
        )

        if device.startswith("cuda"):
            self.model = self.model.to("cuda")
        self.model.eval()

        # padding for batched generation (causal LM prefers left padding)
        if getattr(self.tok, "pad_token_id", None) is None:
            # safe fallback
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    def _build_text(self, prompt: str) -> str:
        if self.use_chat_template and getattr(self.tok, "chat_template", None):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    @staticmethod
    def _postprocess_one(text: str) -> str:
        # common cleanup: strip + take first line
        t = (text or "").strip()
        # remove leading "Answer:" / "OUT:" patterns if present
        t = re.sub(r"^\s*(answer|out)\s*:\s*", "", t, flags=re.IGNORECASE)
        # keep first non-empty line
        lines = [x.strip() for x in t.splitlines() if x.strip()]
        return lines[0] if lines else t

    def _apply_stop(self, text: str) -> str:
        if not self.stop:
            return text
        out = text
        cut = None
        for s in self.stop:
            if not s:
                continue
            idx = out.find(s)
            if idx != -1:
                cut = idx if cut is None else min(cut, idx)
        return out[:cut] if cut is not None else out

    @torch.inference_mode()
    def generate(self, prompts: List[str]) -> List[str]:
        texts = [self._build_text(p) for p in prompts]
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc["attention_mask"].to(self.model.device)
        input_lens = attn.sum(dim=1).tolist()

        do_sample = self.temperature > 1e-6
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            top_k=self.top_k if do_sample and self.top_k > 0 else None,
            repetition_penalty=self.repetition_penalty,
            use_cache=True,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        # remove None
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        out_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            **gen_kwargs,
        )

        outs: List[str] = []
        for i in range(out_ids.size(0)):
            gen = out_ids[i, int(input_lens[i]) :]
            txt = self.tok.decode(gen, skip_special_tokens=True)
            txt = self._apply_stop(txt)
            txt = self._postprocess_one(txt)
            outs.append(txt)
        return outs
