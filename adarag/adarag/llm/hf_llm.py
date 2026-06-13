from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _build_prompt_from_question_docs(question: str, docs: Sequence[str]) -> str:
    # Keep it simple & stable; your main pipeline (CDF) already constructs prompt,
    # but this keeps compatibility for other callers.
    lines = [f"Question: {question}", "", "Evidence:"]
    for i, d in enumerate(docs, 1):
        d = (d or "").strip()
        if not d:
            continue
        lines.append(f"[{i}] {d}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def _as_dtype(dtype: str):
    d = str(dtype).lower()
    if d in ("float16", "fp16", "half"):
        return torch.float16
    if d in ("bfloat16", "bf16"):
        return torch.bfloat16
    if d in ("float32", "fp32"):
        return torch.float32
    # default
    return torch.float16


class HFTextLLM:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 64,
        max_model_len: int = 8192,
        use_chat_template: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        gen_kwargs: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = True,
        local_files_only: bool = True,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.max_model_len = int(max_model_len)
        self.use_chat_template = bool(use_chat_template)
        self.system_prompt = system_prompt
        self.gen_kwargs = dict(gen_kwargs or {})
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only

        # Load tokenizer/model
        self.tok = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )

        torch_dtype = _as_dtype(self.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            device_map="auto" if "cuda" in str(self.device) else None,
        )
        self.model.eval()

        # pad_token_id safety
        if getattr(self.tok, "pad_token_id", None) is None:
            self.tok.pad_token_id = self.tok.eos_token_id

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        prompt = prompt or ""

        # If chat template is available, use it (Qwen2.5 supports this)
        if self.use_chat_template and hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            enc = self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # Some tokenizers return Tensor directly
            if isinstance(enc, torch.Tensor):
                return {"input_ids": enc.to(self.model.device)}
            # Otherwise dict-like
            return {k: v.to(self.model.device) for k, v in enc.items()}
        else:
            enc = self.tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(self.max_model_len, getattr(self.tok, "model_max_length", self.max_model_len)),
            )
            return {k: v.to(self.model.device) for k, v in enc.items()}

    @torch.inference_mode()
    def generate(self, *args):
        if len(args) == 1:
            prompt = str(args[0])
        elif len(args) == 2:
            question = str(args[0])
            docs_raw = args[1]
            docs: List[str] = []
            if docs_raw is not None:
                for d in docs_raw:
                    if isinstance(d, str):
                        docs.append(d)
                    else:
                        t = getattr(d, "text", None)
                        if t is None and isinstance(d, dict):
                            t = d.get("text", None)
                        docs.append("" if t is None else str(t))
            prompt = _build_prompt_from_question_docs(question, docs)
        else:
            raise TypeError(f"HFTextLLM.generate expects 1 or 2 args, got {len(args)}")

        enc = self._encode_prompt(prompt)

        # 统一从 gen_kwargs 取
        temperature = float(self.gen_kwargs.get("temperature", 0.0))
        top_p = float(self.gen_kwargs.get("top_p", 1.0))
        top_k = int(self.gen_kwargs.get("top_k", 0))
        repetition_penalty = float(self.gen_kwargs.get("repetition_penalty", 1.0))

        do_sample = temperature > 1e-6

        gkw = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        if do_sample:
            # 只有采样时才传这些
            gkw.update(dict(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ))

        out = self.model.generate(**enc, **gkw)
        input_len = enc["input_ids"].shape[1]
        gen_ids = out[0][input_len:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return text
