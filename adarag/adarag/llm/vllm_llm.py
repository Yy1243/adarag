# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import inspect
import os


@dataclass
class VllmConfig:
    """
    vLLM generation config.

    该类会被 scripts/eval_light_heavy_retrieval_generation.py 中的
    _build_vllm_config() 读取并过滤参数，因此这里显式声明的字段
    才能从 yaml 配置传入 VllmLLM。
    """

    # model / tokenizer
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer: Optional[str] = None

    # generation params
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    repetition_penalty: float = 1.0

    # vLLM engine params
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    dtype: Optional[str] = None
    max_model_len: Optional[int] = None
    disable_log_stats: bool = True
    seed: int = 0

    # safety / compatibility
    trust_remote_code: bool = False
    local_files_only: bool = True
    download_dir: Optional[str] = None
    enforce_eager: bool = False

    # prompt formatting
    use_chat_template: bool = True
    system_prompt: str = "You are a helpful assistant."

    # vLLM stability switches
    # Some vLLM versions may have V1/async/chunked/prefix-caching compatibility issues.
    # These switches are optional and are only passed if supported by the installed vLLM.
    use_v1: bool = False
    async_scheduling: Optional[bool] = False
    enable_chunked_prefill: Optional[bool] = False
    enable_prefix_caching: Optional[bool] = False


class VllmLLM:
    """
    vLLM backend wrapper.

    Required interface for current evaluation script:
        text = llm.generate(prompt)

    Optional future interface:
        texts = llm.generate_batch(prompts)
    """

    def __init__(self, cfg: VllmConfig):
        self.cfg = cfg
        self._llm = None
        self._params = None
        self._tok = None

    def _lazy_init(self):
        if self._llm is not None:
            return

        # Set environment variables before importing vLLM.
        # Disable V1 engine by default for stability unless explicitly enabled.
        if self.cfg.use_v1:
            os.environ.setdefault("VLLM_USE_V1", "1")
        else:
            os.environ.setdefault("VLLM_USE_V1", "0")

        # If you want strictly offline behavior and all model files are local.
        if self.cfg.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            raise RuntimeError("vLLM is not available. Please install vllm in the current environment.") from e

        sig = inspect.signature(LLM.__init__)
        allowed = set(sig.parameters.keys())

        llm_kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "tensor_parallel_size": int(self.cfg.tensor_parallel_size),
            "gpu_memory_utilization": float(self.cfg.gpu_memory_utilization),
            "trust_remote_code": bool(self.cfg.trust_remote_code),
        }

        if self.cfg.tokenizer:
            llm_kwargs["tokenizer"] = self.cfg.tokenizer

        if self.cfg.dtype is not None:
            llm_kwargs["dtype"] = self.cfg.dtype

        if self.cfg.max_model_len is not None:
            llm_kwargs["max_model_len"] = int(self.cfg.max_model_len)

        if self.cfg.download_dir:
            llm_kwargs["download_dir"] = self.cfg.download_dir

        if "disable_log_stats" in allowed:
            llm_kwargs["disable_log_stats"] = bool(self.cfg.disable_log_stats)

        if "seed" in allowed:
            llm_kwargs["seed"] = int(self.cfg.seed)

        if "enforce_eager" in allowed:
            llm_kwargs["enforce_eager"] = bool(self.cfg.enforce_eager)

        if self.cfg.async_scheduling is not None and "async_scheduling" in allowed:
            llm_kwargs["async_scheduling"] = bool(self.cfg.async_scheduling)

        if self.cfg.enable_chunked_prefill is not None and "enable_chunked_prefill" in allowed:
            llm_kwargs["enable_chunked_prefill"] = bool(self.cfg.enable_chunked_prefill)

        if self.cfg.enable_prefix_caching is not None and "enable_prefix_caching" in allowed:
            llm_kwargs["enable_prefix_caching"] = bool(self.cfg.enable_prefix_caching)

        # Only pass parameters supported by the installed vLLM version.
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if k in allowed}

        self._llm = LLM(**llm_kwargs)

        try:
            self._tok = self._llm.get_tokenizer() if hasattr(self._llm, "get_tokenizer") else None
        except Exception:
            self._tok = None

        # SamplingParams compatibility filtering.
        sp_sig = inspect.signature(SamplingParams.__init__)
        sp_allowed = set(sp_sig.parameters.keys())

        def _sp_kwargs(**kwargs):
            return {k: v for k, v in kwargs.items() if k in sp_allowed}

        stop_token_ids: List[int] = []

        # Generic eos token.
        try:
            if self._tok is not None and getattr(self._tok, "eos_token_id", None) is not None:
                stop_token_ids.append(int(self._tok.eos_token_id))
        except Exception:
            pass

        # Llama-3 style end-of-turn token.
        try:
            if self._tok is not None and hasattr(self._tok, "convert_tokens_to_ids"):
                eot = self._tok.convert_tokens_to_ids("<|eot_id|>")
                if isinstance(eot, int) and eot >= 0:
                    stop_token_ids.append(int(eot))
        except Exception:
            pass

        # Qwen / ChatML style end token.
        try:
            if self._tok is not None and hasattr(self._tok, "convert_tokens_to_ids"):
                im_end = self._tok.convert_tokens_to_ids("<|im_end|>")
                if isinstance(im_end, int) and im_end >= 0:
                    stop_token_ids.append(int(im_end))
        except Exception:
            pass

        stop_token_ids = sorted(set(stop_token_ids))

        sampling_kwargs: Dict[str, Any] = {
            "max_tokens": int(self.cfg.max_new_tokens),
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "repetition_penalty": float(self.cfg.repetition_penalty),
            "skip_special_tokens": True,
        }

        if stop_token_ids:
            sampling_kwargs["stop_token_ids"] = stop_token_ids

        # Text stops are useful for Qwen/ChatML and Llama chat templates.
        sampling_kwargs["stop"] = ["<|eot_id|>", "<|im_end|>"]

        # vLLM versions differ in top_k handling; do not pass top_k when <= 0.
        if self.cfg.top_k is not None and int(self.cfg.top_k) > 0:
            sampling_kwargs["top_k"] = int(self.cfg.top_k)

        self._params = SamplingParams(**_sp_kwargs(**sampling_kwargs))

    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt through tokenizer chat template if enabled.

        当前 RAG prompt 本身已经包含任务说明，因此即使用 system_prompt，
        它也只是作为 chat template 的 system role，不会改动原始 prompt 内容。
        """
        prompt = str(prompt or "")

        if not self.cfg.use_chat_template:
            return prompt

        tok = self._tok
        if tok is None or not hasattr(tok, "apply_chat_template"):
            return prompt

        try:
            messages = []
            if self.cfg.system_prompt:
                messages.append({"role": "system", "content": str(self.cfg.system_prompt)})
            messages.append({"role": "user", "content": prompt})

            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def generate(self, prompt: str) -> str:
        self._lazy_init()

        final_prompt = self._format_prompt(prompt)

        out = self._llm.generate(
            [final_prompt],
            self._params,
            use_tqdm=False,
        )

        if not out:
            return ""

        o0 = out[0]
        if hasattr(o0, "outputs") and o0.outputs and hasattr(o0.outputs[0], "text"):
            return (o0.outputs[0].text or "").strip()

        return str(o0).strip()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Batch generation interface for future acceleration.

        当前 eval_light_heavy_retrieval_generation.py 仍然逐条调用 generate()；
        后续如果要进一步发挥 vLLM 吞吐优势，可以改评测脚本调用该方法。
        """
        self._lazy_init()

        final_prompts = [self._format_prompt(p) for p in prompts]

        outs = self._llm.generate(
            final_prompts,
            self._params,
            use_tqdm=False,
        )

        results: List[str] = []
        for out in outs:
            if hasattr(out, "outputs") and out.outputs and hasattr(out.outputs[0], "text"):
                results.append((out.outputs[0].text or "").strip())
            else:
                results.append("")

        return results