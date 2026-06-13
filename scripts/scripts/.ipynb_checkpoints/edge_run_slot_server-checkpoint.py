# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from adarag.data import QAItem
from adarag.utils import load_yaml
from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever
from adarag.retrievers.heavy_bm25 import HeavyBM25Retriever
from adarag.retrievers.heavy_es_bm25_rerank import ElasticBM25RerankRetriever
from adarag.llm.vllm_llm import VllmLLM, VllmConfig
from adarag.llm.hf_llm import HFTextLLM
from adarag.pipeline.adarag_system_cdf import AdaRAGSystemCDF


# --------- helpers copied from run_adarag_real.py style ---------
def _get_field(x: Any, names: List[str], default=None):
    if x is None:
        return default
    if isinstance(x, dict):
        for n in names:
            if n in x and x[n] is not None:
                return x[n]
        return default
    for n in names:
        if hasattr(x, n):
            v = getattr(x, n)
            if v is not None:
                return v
    return default


def _as_answers(a: Any) -> List[str]:
    if a is None:
        return []
    if isinstance(a, str):
        return [a]
    if isinstance(a, (list, tuple)):
        return [str(z) for z in a if z is not None]
    return [str(a)]


def _make_qa(x: Any) -> QAItem:
    q = _get_field(x, ["q", "question", "query"], default="") or ""
    a = _get_field(x, ["a", "answers", "answer", "gold", "ground_truth"], default=None)
    answers = _as_answers(a)
    qid = _get_field(x, ["qid", "id", "example_id"], default="") or ""
    return QAItem(q=str(q), a=answers, qid=str(qid))


def _build_vllm_config(llm_cfg: dict) -> VllmConfig:
    llm_cfg = llm_cfg or {}
    candidate = {
        "model": llm_cfg.get("model", "meta-llama/Meta-Llama-3-8B-Instruct"),
        "max_new_tokens": int(llm_cfg.get("max_new_tokens", 64)),
        "temperature": float(llm_cfg.get("temperature", 0.0)),
        "top_p": float(llm_cfg.get("top_p", 1.0)),
        "tensor_parallel_size": int(llm_cfg.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(llm_cfg.get("gpu_memory_utilization", 0.85)),
        "max_model_len": llm_cfg.get("max_model_len", None),
        "dtype": llm_cfg.get("dtype", None),
    }
    candidate = {k: v for k, v in candidate.items() if v is not None}
    sig = inspect.signature(VllmConfig.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in candidate.items() if k in allowed}
    for k, v in llm_cfg.items():
        if k in allowed and k not in filtered and v is not None:
            filtered[k] = v
    return VllmConfig(**filtered)


def _onehot_k(n_docs: int, k: int) -> np.ndarray:
    k = max(1, min(n_docs, int(k)))
    p = np.zeros((n_docs,), dtype=float)
    p[k - 1] = 1.0
    return p


def _to_jsonable(x: Any) -> Any:
    # ensure numpy types are JSON-friendly
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    return x


# --------- request schema ---------
class RunSlotReq(BaseModel):
    # either provide items OR question
    items: Optional[List[Dict[str, Any]]] = None
    question: Optional[str] = None
    z: Optional[List[float]] = None
    latency_target_s: Optional[float] = None


def build_system_from_config(config_path: str) -> tuple[AdaRAGSystemCDF, Dict[str, Any]]:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    n_docs = int(cfg.get("n_docs", 10))

    # --- retrievers ---
    light_cfg = cfg.get("light_retriever", {}) or {}
    light_top_k = int(light_cfg.get("top_n", n_docs))
    light = FaissHNSWRetriever(
        corpus_path=light_cfg["corpus_path"],
        index_path=light_cfg["index_path"],
        embedding_model=light_cfg.get("embedding_model", "jinaai/jina-embeddings-v2-base-en"),
        top_n=light_top_k,
        device=light_cfg.get("device", "cuda"),
    )
    light.load_or_build(
        rebuild=bool(light_cfg.get("rebuild", False)),
        max_passages=light_cfg.get("max_passages", None),
    )

    heavy_cfg = cfg["heavy_retriever"]
    heavy_top_k = int(heavy_cfg.get("top_n", n_docs))
    heavy_type = heavy_cfg.get("type", "bm25_local")

    if heavy_type == "bm25_es":
        heavy = ElasticBM25RerankRetriever(
            es_url=heavy_cfg.get("es_url", "http://localhost:9200"),
            index_name=heavy_cfg["index_name"],
            top_n=heavy_top_k,
            bm25_k=int(heavy_cfg.get("bm25_k", 50)),
            reranker_model=heavy_cfg.get("reranker_model", "BAAI/bge-reranker-base"),
            device=heavy_cfg.get("device", "cuda"),
            collapse_field=heavy_cfg.get("collapse_field", None),
            max_per_title=int(heavy_cfg.get("max_per_title", 5)),
            minimum_should_match=heavy_cfg.get("minimum_should_match", None),
            request_timeout=float(heavy_cfg.get("request_timeout", 30.0)),
        )
    elif heavy_type == "bm25_local":
        heavy = HeavyBM25Retriever(
            corpus_path=heavy_cfg["corpus_path"],
            top_n=heavy_cfg.get("top_n", 10),
            max_passages=heavy_cfg.get("max_passages", None),
            candidate_k=heavy_cfg.get("candidate_k", 200),
            use_rerank=heavy_cfg.get("use_rerank", True),
        )
    else:
        raise ValueError(f"Unknown heavy_retriever.type={heavy_type}")

    # --- llm ---
    llm_cfg = cfg.get("llm", {}) or {}
    backend = (llm_cfg.get("backend", "vllm") or "vllm").lower()
    if backend == "vllm":
        llm = VllmLLM(_build_vllm_config(llm_cfg))
    elif backend == "hf":
        gen_kwargs = {
            "temperature": float(llm_cfg.get("temperature", 0.0)),
            "top_p": float(llm_cfg.get("top_p", 1.0)),
            "top_k": int(llm_cfg.get("top_k", 0)),
            "repetition_penalty": float(llm_cfg.get("repetition_penalty", 1.0)),
            "max_new_tokens": int(llm_cfg.get("max_new_tokens", 64)),
        }
        llm = HFTextLLM(
            model_path=llm_cfg["model"],
            tokenizer_path=llm_cfg.get("tokenizer", llm_cfg["model"]),
            device=llm_cfg.get("device", "cuda"),
            dtype=llm_cfg.get("dtype", "float16"),
            max_new_tokens=int(llm_cfg.get("max_new_tokens", 64)),
            max_model_len=int(llm_cfg.get("max_model_len", 8192)),
            use_chat_template=bool(llm_cfg.get("use_chat_template", True)),
            system_prompt=llm_cfg.get("system_prompt", "You are a helpful assistant."),
            gen_kwargs=gen_kwargs,
            trust_remote_code=bool(llm_cfg.get("trust_remote_code", False)),
            local_files_only=bool(llm_cfg.get("local_files_only", True)),
        )
    else:
        raise ValueError(f"Unknown llm.backend={backend}, expected vllm|hf")

    # --- judge llm (as in baseline) ---
    judge_cfg = cfg.get("judge_llm", None)
    judge_llm = None
    if judge_cfg:
        j = judge_cfg
        j_gen_kwargs = {
            "temperature": float(j.get("temperature", 0.0)),
            "top_p": float(j.get("top_p", 1.0)),
            "top_k": int(j.get("top_k", 0)),
            "repetition_penalty": float(j.get("repetition_penalty", 1.0)),
            "max_new_tokens": int(j.get("max_new_tokens", 8)),
        }
        judge_llm = HFTextLLM(
            model_path=j["model"],
            tokenizer_path=j.get("tokenizer", j["model"]),
            device=j.get("device", "cuda"),
            dtype=j.get("dtype", "float16"),
            max_new_tokens=int(j.get("max_new_tokens", 8)),
            max_model_len=int(j.get("max_model_len", 8192)),
            use_chat_template=bool(j.get("use_chat_template", True)),
            system_prompt=j.get("system_prompt", "You are a strict evaluator."),
            gen_kwargs=j_gen_kwargs,
            trust_remote_code=bool(j.get("trust_remote_code", False)),
            local_files_only=bool(j.get("local_files_only", True)),
        )

    sys_cfg = cfg.get("system", {}) or {}
    system = AdaRAGSystemCDF(
        light_retriever=light,
        heavy_retriever=heavy,
        llm=llm,
        n_docs=n_docs,
        acc_mode=cfg.get("acc_mode", "contains"),
        seed=seed,
        force_heavy=bool(sys_cfg.get("force_heavy", False)),
        prompt_max_doc_chars=int(sys_cfg.get("prompt_max_doc_chars", 1600)),
        judge_llm=judge_llm,
    )
    return system, cfg


def build_app(server_id: str, system: AdaRAGSystemCDF, cfg: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title=f"EdgeRunSlot-{server_id}")

    n_docs = int(cfg.get("n_docs", 10))
    # default z if client doesn't provide one (safe for quick curl tests):
    # take k=n_docs in light & heavy, and p=1.0 (force heavy) => stable quality
    default_z = np.concatenate([_onehot_k(n_docs, n_docs), _onehot_k(n_docs, n_docs), [1.0]]).astype(float)

    @app.get("/health")
    def health():
        return {"ok": True, "server_id": server_id}

    @app.post("/run_slot")
    def run_slot(req: RunSlotReq):
        if req.items is None and req.question is None:
            return {"error": "provide either items or question"}

        if req.items is not None:
            batch = [_make_qa(x) for x in req.items]
        else:
            batch = [_make_qa({"q": req.question or "", "a": []})]

        z_arr = np.asarray(req.z, dtype=float) if req.z is not None else default_z
        latency_target_s = float(req.latency_target_s if req.latency_target_s is not None else 1e9)

        res = system.run_slot(batch=batch, z=z_arr, latency_target_s=latency_target_s)

        # make sure JSON serializable
        res = _to_jsonable(res)
        return jsonable_encoder(res)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--server-id", type=str, required=True)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    system, cfg = build_system_from_config(args.config)
    app = build_app(args.server_id, system, cfg)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()