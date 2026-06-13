# /T20050013/adarag_repro/scripts/eval_light_heavy_retrieval_generation.py
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from adarag.data import QAItem
from adarag.data_hf import load_nq_open_stream
from adarag.eval.evaluator import batch_accuracy
from adarag.pipeline.prompt_builder import build_prompt
from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever
from adarag.retrievers.heavy_bm25 import HeavyBM25Retriever
from adarag.retrievers.heavy_es_bm25_rerank import ElasticBM25RerankRetriever
from adarag.retrievers.heavy_hybrid_dense_bm25 import HybridDenseBM25Retriever
from adarag.llm.hf_llm import HFTextLLM
from adarag.llm.vllm_llm import VllmLLM, VllmConfig
from adarag.utils import load_yaml
from adarag.retrievers.heavy_dense_rerank import DenseRerankHeavyRetriever
from adarag.retrievers.heavy_hybrid_dense_bm25_cross_rerank import HybridDenseBM25CrossRerankRetriever

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def _doc_to_text(d: Any) -> str:
    """
    评测统计专用文本：
    - Doc 对象：使用 d.full_text，即 title + "\n" + text；
    - dict：显式拼 title + text；
    - 这里用于 oracle recall / contains / prompt_doc_chars 统计。
    """
    if d is None:
        return ""

    if isinstance(d, str):
        return d

    if isinstance(d, dict):
        title = str(d.get("title") or "").strip()
        text = str(
            d.get("text")
            or d.get("contents")
            or d.get("passage")
            or d.get("document")
            or ""
        ).strip()

        if title and text:
            return f"{title}\n{text}"
        return title or text

    if hasattr(d, "full_text"):
        return str(d.full_text or "")

    title = str(getattr(d, "title", "") or "").strip()
    text = str(getattr(d, "text", "") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("text", "output", "answer", "generated_text"):
            if key in x:
                return str(x[key])
        return str(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _to_text(x[0])
    return str(x)


def _postprocess_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    m = re.search(r"final\s*answer\s*[:：]\s*(.*)$", s, flags=re.I | re.S)
    if m:
        s = m.group(1)
    s = " ".join(s.strip().split())
    return s


def _contains_any(text: str, gold: Any) -> bool:
    if text is None:
        return False
    t = _norm(text)
    if isinstance(gold, str):
        g = _norm(gold)
        return bool(g) and (g in t)
    if isinstance(gold, list):
        for a in gold:
            g = _norm(a)
            if g and (g in t):
                return True
    return False


def _oracle_hit(docs: List[Any], gold: Any) -> bool:
    if not docs:
        return False
    texts = [_norm(_doc_to_text(d)) for d in docs]
    if isinstance(gold, str):
        g = _norm(gold)
        return any(g and (g in t) for t in texts)
    if isinstance(gold, list):
        for a in gold:
            g = _norm(a)
            if any(g and (g in t) for t in texts):
                return True
    return False


def _judge_correct(judge_llm, question: str, pred: str, gold: Any) -> Optional[bool]:
    if judge_llm is None:
        return None

    gold_list = gold if isinstance(gold, list) else [gold]
    gold_list = [str(x) for x in gold_list if x is not None]

    prompt = (
        "You are a strict QA evaluator.\n"
        "Decide whether the model answer is semantically equivalent to ANY gold answer.\n"
        "Return ONLY 1 or 0.\n\n"
        f"Question: {question}\n"
        f"Gold answers: {gold_list}\n"
        f"Model answer: {pred}\n"
    )

    out = judge_llm.generate(prompt)
    out = _to_text(out).strip()
    m = re.search(r"[01]", out)
    return (m is not None) and (m.group(0) == "1")


def _safe_mean(g: pd.DataFrame, col: str):
    if col not in g.columns:
        return None
    s = pd.to_numeric(g[col], errors="coerce")
    if s.notna().sum() == 0:
        return None
    return float(s.mean())


def _build_vllm_config(llm_cfg: dict) -> VllmConfig:
    import inspect

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


def build_llm(llm_cfg: dict):
    backend = (llm_cfg.get("backend", "vllm") or "vllm").lower()

    if backend in ("openai_compat", "openai-compatible", "openai"):
        # Lazy import: only required when using OpenAI-compatible HTTP backend.
        # This keeps the original hf/vllm branches unaffected.
        from adarag.llm.openai_compat_llm import OpenAICompatLLM, OpenAICompatConfig

        return OpenAICompatLLM(OpenAICompatConfig(
            base_url=str(llm_cfg.get("base_url", "http://127.0.0.1:8000/v1")),
            api_key=str(llm_cfg.get("api_key", "EMPTY")),
            model=str(llm_cfg.get("model", "qwen25-7b")),
            max_new_tokens=int(llm_cfg.get("max_new_tokens", llm_cfg.get("max_tokens", 64))),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            top_p=float(llm_cfg.get("top_p", 1.0)),
            repetition_penalty=(
                None if llm_cfg.get("repetition_penalty", None) is None
                else float(llm_cfg.get("repetition_penalty"))
            ),
            system_prompt=str(llm_cfg.get("system_prompt", "You are a helpful assistant.")),
            request_timeout_s=float(llm_cfg.get("request_timeout_s", 120.0)),
            max_retries=int(llm_cfg.get("max_retries", 2)),
            retry_sleep_s=float(llm_cfg.get("retry_sleep_s", 1.0)),
        ))

    if backend == "vllm":
        return VllmLLM(_build_vllm_config(llm_cfg))

    if backend == "hf":
        gen_kwargs = {
            "temperature": float(llm_cfg.get("temperature", 0.0)),
            "top_p": float(llm_cfg.get("top_p", 1.0)),
            "top_k": int(llm_cfg.get("top_k", 0)),
            "repetition_penalty": float(llm_cfg.get("repetition_penalty", 1.0)),
            "max_new_tokens": int(llm_cfg.get("max_new_tokens", 64)),
        }
        return HFTextLLM(
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

    raise ValueError(f"Unknown llm.backend={backend}")


def build_light_retriever(cfg: dict, topn: int):
    light_cfg = cfg.get("light_retriever", {}) or {}
    retriever = FaissHNSWRetriever(
        corpus_path=light_cfg["corpus_path"],
        index_path=light_cfg["index_path"],
        embedding_model=light_cfg.get("embedding_model", "jinaai/jina-embeddings-v2-base-en"),
        top_n=topn,
        device=light_cfg.get("device", "cuda"),
    )
    retriever.load_or_build(
        rebuild=bool(light_cfg.get("rebuild", False)),
        max_passages=light_cfg.get("max_passages", None),
    )
    return retriever


def build_heavy_retriever(cfg: dict, topn: int, light=None, llm_rewriter=None):
    heavy_cfg = cfg["heavy_retriever"]
    heavy_type = heavy_cfg.get("type", "bm25_local")

    if heavy_type == "bm25_es":
        return ElasticBM25RerankRetriever(
            es_url=heavy_cfg.get("es_url", "http://localhost:9200"),
            index_name=heavy_cfg["index_name"],
            top_n=topn,
            bm25_k=int(heavy_cfg.get("bm25_k", 50)),
            reranker_model=heavy_cfg.get("reranker_model", "BAAI/bge-reranker-base"),
            device=heavy_cfg.get("device", "cuda"),
            collapse_field=heavy_cfg.get("collapse_field", None),
            max_per_title=int(heavy_cfg.get("max_per_title", 5)),
            minimum_should_match=heavy_cfg.get("minimum_should_match", None),
            request_timeout=float(heavy_cfg.get("request_timeout", 30.0)),
            rerank_k=int(heavy_cfg.get("rerank_k", min(topn, 50))),
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 32)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 2000)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

    if heavy_type == "bm25_local":
        return HeavyBM25Retriever(
            corpus_path=heavy_cfg["corpus_path"],
            top_n=topn,
            max_passages=heavy_cfg.get("max_passages", None),
            candidate_k=heavy_cfg.get("candidate_k", 200),
            use_rerank=heavy_cfg.get("use_rerank", True),
        )

    if heavy_type == "hybrid_dense_bm25_cross":
        if light is None:
            raise ValueError("hybrid_dense_bm25_cross requires light retriever")

        bm25 = ElasticBM25RerankRetriever(
            es_url=heavy_cfg.get("es_url", "http://localhost:9200"),
            index_name=heavy_cfg["index_name"],
            top_n=topn,
            bm25_k=int(heavy_cfg.get("bm25_k", 50)),
            reranker_model=None,
            device=heavy_cfg.get("device", "cpu"),
            collapse_field=heavy_cfg.get("collapse_field", None),
            max_per_title=int(heavy_cfg.get("max_per_title", 5)),
            minimum_should_match=heavy_cfg.get("minimum_should_match", None),
            request_timeout=float(heavy_cfg.get("request_timeout", 30.0)),
            rerank_k=0,
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 32)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 2000)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

        return HybridDenseBM25CrossRerankRetriever(
            dense_retriever=light,
            bm25_retriever=bm25,
            reranker_model=heavy_cfg["reranker_model"],
            dense_top_n=int(heavy_cfg.get("dense_top_n", 80)),
            bm25_top_n=int(heavy_cfg.get("bm25_top_n", 80)),
            out_top_n=topn,
            rrf_k=int(heavy_cfg.get("rrf_k", 60)),
            dense_weight=float(heavy_cfg.get("dense_weight", 1.0)),
            bm25_weight=float(heavy_cfg.get("bm25_weight", 0.20)),
            candidate_pool_n=int(heavy_cfg.get("candidate_pool_n", 60)),
            rerank_k=int(heavy_cfg.get("rerank_k", 40)),
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 8)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 256)),
            rerank_device=str(heavy_cfg.get("rerank_device", "cpu")).strip(),
            max_per_title=int(heavy_cfg.get("max_per_title", 2)),
            dedup_key_mode=str(heavy_cfg.get("dedup_key_mode", "title_text")).strip().lower(),
            simplify_bm25=bool(heavy_cfg.get("simplify_bm25", True)),
            lexical_bonus_weight=float(heavy_cfg.get("lexical_bonus_weight", 0.06)),
            title_bonus=float(heavy_cfg.get("title_bonus", 0.05)),
            phrase_bonus=float(heavy_cfg.get("phrase_bonus", 0.08)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

    if heavy_type == "dense_rerank_heavy":
        if light is None:
            raise ValueError("dense_rerank_heavy requires light retriever")

        return DenseRerankHeavyRetriever(
            dense_retriever=light,
            reranker_model=heavy_cfg["reranker_model"],
            candidate_pool_n=int(heavy_cfg.get("candidate_pool_n", 120)),
            rerank_head_n=int(heavy_cfg.get("rerank_head_n", 30)),
            out_top_n=topn,
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 8)),
            rerank_max_length=int(heavy_cfg.get("rerank_max_length", 512)),
            rerank_device=str(heavy_cfg.get("rerank_device", "cpu")).strip(),
            dense_prior_weight=float(heavy_cfg.get("dense_prior_weight", 0.75)),
            max_per_title=int(heavy_cfg.get("max_per_title", 999)),
            dedup_key_mode=str(heavy_cfg.get("dedup_key_mode", "title_text")).strip().lower(),
            lexical_bonus_weight=float(heavy_cfg.get("lexical_bonus_weight", 0.00)),
            title_bonus=float(heavy_cfg.get("title_bonus", 0.00)),
            phrase_bonus=float(heavy_cfg.get("phrase_bonus", 0.00)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

    if heavy_type == "hybrid_dense_bm25_cross_rerank":
        if light is None:
            raise ValueError("hybrid_dense_bm25_cross_rerank requires light retriever")

        bm25 = ElasticBM25RerankRetriever(
            es_url=heavy_cfg.get("es_url", "http://127.0.0.1:9200"),
            index_name=heavy_cfg["index_name"],
            top_n=int(heavy_cfg.get("bm25_top_n", 100)),
            bm25_k=int(heavy_cfg.get("bm25_k", 300)),
            reranker_model=None,
            device=str(heavy_cfg.get("device", "cpu")).strip(),
            collapse_field=heavy_cfg.get("collapse_field", None),
            max_per_title=int(heavy_cfg.get("bm25_max_per_title", 5)),
            minimum_should_match=heavy_cfg.get("minimum_should_match", None),
            request_timeout=float(heavy_cfg.get("request_timeout", 120)),
            rerank_k=0,
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 8)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_length", 512)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

        return HybridDenseBM25CrossRerankRetriever(
            dense_retriever=light,
            bm25_retriever=bm25,
            reranker_model=heavy_cfg["reranker_model"],
            dense_top_n=int(heavy_cfg.get("dense_top_n", 60)),
            bm25_top_n=int(heavy_cfg.get("bm25_top_n", 100)),
            out_top_n=topn,
            candidate_pool_n=int(heavy_cfg.get("candidate_pool_n", 80)),
            rrf_k=int(heavy_cfg.get("rrf_k", 60)),
            dense_weight=float(heavy_cfg.get("dense_weight", 1.0)),
            bm25_weight=float(heavy_cfg.get("bm25_weight", 0.15)),
            rerank_k=int(heavy_cfg.get("rerank_k", 50)),
            rerank_weight=float(heavy_cfg.get("rerank_weight", 0.15)),
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 8)),
            rerank_max_length=int(heavy_cfg.get("rerank_max_length", 512)),
            rerank_device=str(heavy_cfg.get("rerank_device", "cuda")).strip(),
            max_per_title=int(heavy_cfg.get("max_per_title", 2)),
            dedup_key_mode=str(heavy_cfg.get("dedup_key_mode", "title_text")).strip().lower(),
            dense_prompt_n=int(heavy_cfg.get("dense_prompt_n", 8)),
            rescue_top_n=int(heavy_cfg.get("rescue_top_n", 2)),
            rescue_rerank_k=int(heavy_cfg.get("rescue_rerank_k", 100)),
            mix_top_n=int(heavy_cfg.get("mix_top_n", 10)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

    if heavy_type == "hybrid_dense_bm25":
        if light is None:
            raise ValueError("hybrid_dense_bm25 requires light retriever")

        bm25 = ElasticBM25RerankRetriever(
            es_url=heavy_cfg.get("es_url", "http://localhost:9200"),
            index_name=heavy_cfg["index_name"],

            # 这里建议用 bm25_top_n 作为 BM25 最终返回上限；
            # bm25_k 是 ES 候选池大小。
            top_n=int(heavy_cfg.get("bm25_top_n", topn)),
            bm25_k=int(heavy_cfg.get("bm25_k", heavy_cfg.get("bm25_top_n", 300))),

            reranker_model=None,
            device=str(heavy_cfg.get("device", "cpu")).strip(),
            collapse_field=heavy_cfg.get("collapse_field", None),

            # 候选池阶段建议优先读 bm25_max_per_title；
            # 设为 0 表示不限制同 title 的 passage 数量。
            max_per_title=int(heavy_cfg.get("bm25_max_per_title", heavy_cfg.get("max_per_title", 0))),

            minimum_should_match=heavy_cfg.get("minimum_should_match", None),
            request_timeout=float(heavy_cfg.get("request_timeout", 120.0)),

            # 关键：把 text_only_simple 传进去
            query_mode=str(
                heavy_cfg.get("bm25_query_mode", heavy_cfg.get("query_mode", "text_only_simple"))
            ).strip().lower(),

            rerank_k=0,
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 32)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 2000)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

        return HybridDenseBM25Retriever(
            dense_retriever=light,
            bm25_retriever=bm25,
            dense_top_n=int(heavy_cfg.get("dense_top_n", 50)),
            bm25_top_n=int(heavy_cfg.get("bm25_top_n", topn)),
            out_top_n=topn,
            merge_mode=str(heavy_cfg.get("merge_mode", "rrf")).strip().lower(),
            dense_keep_n=int(heavy_cfg.get("dense_keep_n", 10)),
            bm25_keep_n=int(heavy_cfg.get("bm25_keep_n", 3)),
            rrf_k=int(heavy_cfg.get("rrf_k", 60)),
            dense_weight=float(heavy_cfg.get("dense_weight", 1.0)),
            bm25_weight=float(heavy_cfg.get("bm25_weight", 0.2)),
            simplify_bm25=bool(heavy_cfg.get("simplify_bm25", True)),
            dedup_key_mode=str(heavy_cfg.get("dedup_key_mode", "title_text")).strip().lower(),
            score_fuse_norm=str(heavy_cfg.get("score_fuse_norm", "minmax")).strip().lower(),
            score_fuse_w_dense=float(heavy_cfg.get("score_fuse_w_dense", 0.8)),
            score_fuse_w_bm25=float(heavy_cfg.get("score_fuse_w_bm25", 0.2)),
            rewrite_mode=str(heavy_cfg.get("rewrite_mode", "none")).strip().lower(),
            rewrite_prf_docs=int(heavy_cfg.get("rewrite_prf_docs", 5)),
            rewrite_prf_terms=int(heavy_cfg.get("rewrite_prf_terms", 8)),
            rewrite_doc_max_chars=int(heavy_cfg.get("rewrite_doc_max_chars", 400)),
            rewrite_min_df=int(heavy_cfg.get("rewrite_min_df", 2)),
            llm_rewriter=llm_rewriter,
            reranker_model=heavy_cfg.get("reranker_model", None),
            rerank_k=int(heavy_cfg.get("rerank_k", 30)),
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 8)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 256)),
            profile=bool(heavy_cfg.get("profile", False)),
        )

    raise ValueError(f"Unknown heavy_retriever.type={heavy_type}")


def load_questions(cfg: dict, max_questions: int) -> List[QAItem]:
    ds_cfg = cfg["dataset"]
    it = load_nq_open_stream(
        split=ds_cfg.get("split", "validation"),
        seed=int(ds_cfg.get("seed", cfg.get("seed", 42))),
        local_path=ds_cfg["local_path"],
        max_examples=ds_cfg.get("max_examples", None),
    )

    out = []
    for x in it:
        out.append(_make_qa(x))
        if len(out) >= max_questions:
            break
    return out


def evaluate_path(
    qa: QAItem,
    path_name: str,
    docs: List[Any],
    retrieve_time_s: float,
    llm,
    judge_llm,
    prompt_max_doc_chars: int,
    acc_mode: str,
    gen_topk: int,
) -> Dict[str, Any]:
    docs_k = docs[:gen_topk]

    t0 = time.time()
    prompt = build_prompt(qa.q, docs_k, max_doc_chars=prompt_max_doc_chars)
    prompt_build_time_s = time.time() - t0

    t1 = time.time()
    pred_raw = llm.generate(prompt)
    generate_time_s = time.time() - t1
    pred = _postprocess_answer(_to_text(pred_raw))

    acc_value = float(batch_accuracy([pred], [qa.a], mode=acc_mode)) if qa.a else None
    contains_correct = _contains_any(pred, qa.a) if qa.a else None
    judge_correct = _judge_correct(judge_llm, qa.q, pred, qa.a) if qa.a else None
    prompt_has_gold = _contains_any(prompt, qa.a) if qa.a else None

    prompt_chars = len(prompt)
    prompt_doc_chars = sum(len(_doc_to_text(d)) for d in docs_k)

    return {
        "qid": qa.qid,
        "question": qa.q,
        "path": path_name,
        "retrieve_time_s": retrieve_time_s,

        "recall_at_gen_topk": int(_oracle_hit(docs_k, qa.a)),
        "recall_at_1": int(_oracle_hit(docs[:1], qa.a)),
        "recall_at_2": int(_oracle_hit(docs[:2], qa.a)),
        "recall_at_3": int(_oracle_hit(docs[:3], qa.a)),
        "recall_at_4": int(_oracle_hit(docs[:4], qa.a)),
        "recall_at_5": int(_oracle_hit(docs[:5], qa.a)),
        "recall_at_10": int(_oracle_hit(docs[:10], qa.a)),
        "recall_at_50": int(_oracle_hit(docs[:50], qa.a)),
        "recall_at_200": int(_oracle_hit(docs[:200], qa.a)),

        "gen_topk": gen_topk,
        "prompt_chars": prompt_chars,
        "prompt_doc_chars": prompt_doc_chars,
        "prompt_build_time_s": prompt_build_time_s,
        "generate_time_s": generate_time_s,
        "total_time_s": retrieve_time_s + prompt_build_time_s + generate_time_s,

        "acc_mode": acc_mode,
        "acc_value": acc_value,
        "contains_correct": contains_correct,
        "judge_correct": judge_correct,
        "prompt_has_gold": prompt_has_gold,

        "pred": pred,
        "gold_answers": json.dumps(qa.a, ensure_ascii=False),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for path_name, g in df.groupby("path"):
        rows.append({
            "path": path_name,
            "n_questions": int(len(g)),
            "gen_topk": int(g["gen_topk"].iloc[0]) if "gen_topk" in g else None,

            "mean_retrieve_time_s": _safe_mean(g, "retrieve_time_s"),
            "mean_prompt_build_time_s": _safe_mean(g, "prompt_build_time_s"),
            "mean_generate_time_s": _safe_mean(g, "generate_time_s"),
            "mean_total_time_s": _safe_mean(g, "total_time_s"),

            "mean_prompt_chars": _safe_mean(g, "prompt_chars"),
            "mean_prompt_doc_chars": _safe_mean(g, "prompt_doc_chars"),

            "recall_at_gen_topk": _safe_mean(g, "recall_at_gen_topk"),
            "recall_at_1": _safe_mean(g, "recall_at_1"),
            "recall_at_2": _safe_mean(g, "recall_at_2"),
            "recall_at_3": _safe_mean(g, "recall_at_3"),
            "recall_at_4": _safe_mean(g, "recall_at_4"),
            "recall_at_5": _safe_mean(g, "recall_at_5"),
            "recall_at_10": _safe_mean(g, "recall_at_10"),
            "recall_at_50": _safe_mean(g, "recall_at_50"),
            "recall_at_200": _safe_mean(g, "recall_at_200"),

            "mean_acc_value": _safe_mean(g, "acc_value"),
            "contains_accuracy": _safe_mean(g, "contains_correct"),
            "judge_accuracy": _safe_mean(g, "judge_correct"),
            "prompt_gold_rate": _safe_mean(g, "prompt_has_gold"),
        })

    out = pd.DataFrame(rows)

    if "gen_topk" in out.columns:
        out = out.sort_values(["gen_topk", "path"])

    return out


def _parse_topks(s: str, fallback: int) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return [int(fallback)]

    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if k <= 0:
            raise ValueError(f"top-k must be positive, got {k}")
        out.append(k)

    if not out:
        out = [int(fallback)]

    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="Evaluate light/heavy retrieval recall and top-k generation quality.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--max-questions", type=int, default=100)
    ap.add_argument("--fetch-topn", type=int, default=200)
    ap.add_argument("--gen-topk", type=int, default=10)
    ap.add_argument("--output-dir", type=str, default="outputs_light_heavy_eval")
    ap.add_argument("--disable-judge", action="store_true")

    ap.add_argument(
        "--light-gen-topks",
        type=str,
        default="",
        help="Comma-separated light gen top-k list, e.g. 1,2,3,4,5. If set, sweep these top-k values for light retrieval.",
    )
    ap.add_argument(
        "--light-only",
        action="store_true",
        help="Only evaluate dense/light retrieval; skip heavy retrieval.",
    )

    args = ap.parse_args()

    cfg = load_yaml(args.config)
    _ensure_dir(args.output_dir)

    prompt_max_doc_chars = int((cfg.get("system", {}) or {}).get("prompt_max_doc_chars", 1600))
    acc_mode = str(cfg.get("acc_mode", "token"))

    light_gen_topks = _parse_topks(args.light_gen_topks, fallback=args.gen_topk)

    questions = load_questions(cfg, args.max_questions)
    if not questions:
        raise RuntimeError("No dataset questions loaded.")

    print(f"[init] loaded questions: {len(questions)}")
    print(f"[init] light_gen_topks: {light_gen_topks}")
    print(f"[init] light_only: {args.light_only}")
    print(f"[init] fetch_topn: {args.fetch_topn}")
    print(f"[init] prompt_max_doc_chars: {prompt_max_doc_chars}")
    print(f"[init] acc_mode: {acc_mode}")

    print("[init] loading light retriever...")
    light = build_light_retriever(cfg, topn=args.fetch_topn)
    print("[init] light retriever loaded.")

    print("[init] loading generator llm...")
    llm = build_llm(cfg.get("llm", {}) or {})
    print("[init] generator llm loaded.")

    heavy = None
    rewriter_llm = None

    if not args.light_only:
        rewriter_cfg = cfg.get("rewriter_llm", None)
        rewriter_llm = build_llm(rewriter_cfg) if rewriter_cfg else llm

        print("[init] loading heavy retriever...")
        heavy = build_heavy_retriever(cfg, topn=args.fetch_topn, light=light, llm_rewriter=rewriter_llm)
        print("[init] heavy retriever loaded.")

    judge_llm = None
    if (not args.disable_judge) and cfg.get("judge_llm", None):
        print("[init] loading judge llm...")
        judge_llm = build_llm(cfg.get("judge_llm", {}) or {})
        print("[init] judge llm loaded.")

    records = []
    t_all = time.time()

    for idx, qa in enumerate(questions, start=1):
        t0 = time.time()
        light_docs, _ = light.retrieve(qa.q)
        light_retrieve_time_s = time.time() - t0
        light_docs = light_docs[:args.fetch_topn]

        for k in light_gen_topks:
            records.append(
                evaluate_path(
                    qa=qa,
                    path_name=f"light_top{k}",
                    docs=light_docs,
                    retrieve_time_s=light_retrieve_time_s,
                    llm=llm,
                    judge_llm=judge_llm,
                    prompt_max_doc_chars=prompt_max_doc_chars,
                    acc_mode=acc_mode,
                    gen_topk=k,
                )
            )

        if not args.light_only:
            t1 = time.time()
            heavy_docs, _ = heavy.retrieve(qa.q)
            heavy_retrieve_time_s = time.time() - t1
            heavy_docs = heavy_docs[:args.fetch_topn]

            records.append(
                evaluate_path(
                    qa=qa,
                    path_name=f"heavy_top{args.gen_topk}",
                    docs=heavy_docs,
                    retrieve_time_s=heavy_retrieve_time_s,
                    llm=llm,
                    judge_llm=judge_llm,
                    prompt_max_doc_chars=prompt_max_doc_chars,
                    acc_mode=acc_mode,
                    gen_topk=args.gen_topk,
                )
            )

        if idx % 10 == 0:
            tmp_df = pd.DataFrame(records)
            tmp_summary = summarize(tmp_df)

            cols = [
                "path",
                "gen_topk",
                "judge_accuracy",
                "contains_accuracy",
                "prompt_gold_rate",
                "mean_prompt_build_time_s",
                "mean_generate_time_s",
                "mean_total_time_s",
            ]
            cols = [c for c in cols if c in tmp_summary.columns]

            print(f"\n[progress] Processed {idx}/{len(questions)} questions elapsed={time.time() - t_all:.1f}s")
            print(tmp_summary[cols].to_string(index=False))

    detailed_df = pd.DataFrame(records)
    summary_df = summarize(detailed_df)

    detailed_csv = os.path.join(args.output_dir, "light_heavy_detailed.csv")
    summary_csv = os.path.join(args.output_dir, "light_heavy_summary.csv")
    detailed_df.to_csv(detailed_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "max_questions": args.max_questions,
                "fetch_topn": args.fetch_topn,
                "gen_topk": args.gen_topk,
                "light_gen_topks": light_gen_topks,
                "light_only": args.light_only,
                "disable_judge": args.disable_judge,
                "output_dir": args.output_dir,
                "elapsed_s": time.time() - t_all,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Done ===")
    print(summary_df)
    print(f"\nSaved summary : {summary_csv}")
    print(f"Saved detailed: {detailed_csv}")
    print(f"Saved meta    : {os.path.join(args.output_dir, 'run_meta.json')}")


if __name__ == "__main__":
    main()