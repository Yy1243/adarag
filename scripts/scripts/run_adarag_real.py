# -*- coding: utf-8 -*-
"""
Run AdaRAG (real-min) in offline-friendly way.

This version is aligned with the paper's online optimization logic:
- executed action is z_t = z_hat_t + delta * u
- z_hat is updated by projection onto (1-gamma)Z using the correct scaled projection
- gamma can be tied to radius_r through gamma = delta / r
- the legacy non-paper BanditOptimizer is removed from the optimizer module
"""

from __future__ import annotations

import os
import json
import argparse
import inspect
from typing import Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from adarag.data import QAItem
from adarag.utils import load_yaml
from adarag.optimizer.bandit_optimizer import AdaRAGBanditOptimizer
from adarag.data_hf import load_nq_open_stream
from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever
from adarag.retrievers.heavy_bm25 import HeavyBM25Retriever
from adarag.retrievers.heavy_es_bm25_rerank import ElasticBM25RerankRetriever
from adarag.retrievers.heavy_hybrid_dense_bm25 import HybridDenseBM25Retriever
from adarag.llm.vllm_llm import VllmLLM, VllmConfig
from adarag.llm.hf_llm import HFTextLLM
from adarag.pipeline.adarag_system_addfold import AdaRAGSystemCDF


def _ensure_dir(p: str) -> None:        #确保目录存在
    os.makedirs(p, exist_ok=True)


def _get_field(x: Any, names: List[str], default=None):         #从x中是找否有name列表中的属性取得相应的值，找到第一个非 None 的值返回，都找不到就返回 default
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


def _as_answers(a: Any) -> List[str]:       #确保答案是字符串列表，数据清理
    if a is None:
        return []
    if isinstance(a, str):
        return [a]
    if isinstance(a, (list, tuple)):
        return [str(z) for z in a if z is not None]
    return [str(a)]


def _make_qa(x: Any) -> QAItem:             #确保生成标准的QAItem对象，数据清理
    q = _get_field(x, ["q", "question", "query"], default="") or ""
    a = _get_field(x, ["a", "answers", "answer", "gold", "ground_truth"], default=None)
    answers = _as_answers(a)
    qid = _get_field(x, ["qid", "id", "example_id"], default="") or ""
    return QAItem(q=str(q), a=answers, qid=str(qid))


def _build_vllm_config(llm_cfg: dict) -> VllmConfig:        #传入的是config.yaml配置文件，字典类型
    llm_cfg = llm_cfg or {}

    candidate = {
        "model": llm_cfg.get("model", "meta-llama/Meta-Llama-3-8B-Instruct"),           #字典的用法.get,获取model的属性值，否则返回后面的默认值
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
    allowed = set(sig.parameters.keys()) - {"self"}              #将参数名减去self后得到的参数名集合  
    filtered = {k: v for k, v in candidate.items() if k in allowed}         #过滤不支持的参数，保留可支持的参数

    for k, v in llm_cfg.items():             #允许用户传入不在默认列表中但当前 vLLM 支持的额外参数。
        if k in allowed and k not in filtered and v is not None:
            filtered[k] = v

    return VllmConfig(**filtered)        #只传当前版本支持的参数，返回一个实例对象，**filtered代表字典解包，如data = {"name": "Bob", "age": 30}，p2 = Person(**data)等价于 Person(name="Bob", age=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true", help="truncate outputs before run")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    out_dir = cfg.get("output_dir", "outputs_real_min")
    _ensure_dir(out_dir)
    seed = int(cfg.get("seed", 42))

    # --------------------------
    # data stream (offline)
    # --------------------------
    ds_cfg = cfg["dataset"]         #ds_cfg为分化出来的字典，dataset部分的配置，cfg是总配置字典。
    it = load_nq_open_stream(           #load_nq_open_stream函数是用来加载Natural Questions Open数据集,it是一个迭代器，返回的数据类型可能是QAItem/dataclass/dict等不同类型，函数内部设计了数据清理和标准化的步骤，确保输出的每个元素都具有.q和.a属性，分别代表问题和答案,惰性读回
        split=ds_cfg.get("split", "validation"),
        seed=int(ds_cfg.get("seed", seed)),
        local_path=ds_cfg["local_path"],
        max_examples=ds_cfg.get("max_examples", None),      #None表示全部加载，函数里面的设计
    )
    raw_items = list(it)
    if not raw_items:
        raise RuntimeError("Dataset stream is empty. Check dataset.local_path and max_examples.")       #raise关键字抛出异常，中断程序执行，RuntimeError是Python内置的异常类型，表示在运行时发生了错误

    slot_size = int(cfg.get("slot_size", 10))
    n_slots_cfg = cfg.get("n_slots", None)
    if n_slots_cfg is None:
        n_slots = max(1, len(raw_items) // slot_size)
    else:
        n_slots = int(n_slots_cfg)

    total_needed = min(len(raw_items), n_slots * slot_size)
    n_slots = max(1, total_needed // slot_size)
    qa_used = raw_items[: n_slots * slot_size]

    # --------------------------
    # retrievers
    # --------------------------
    n_docs = int(cfg.get("n_docs", 10))
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
            rerank_k=heavy_cfg.get("rerank_k", 50),
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 32)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 2000)),
            profile=bool(heavy_cfg.get("profile", False)),
        )
    elif heavy_type == "bm25_local":
        heavy = HeavyBM25Retriever(
            corpus_path=heavy_cfg["corpus_path"],
            top_n=heavy_cfg.get("top_n", 10),
            max_passages=heavy_cfg.get("max_passages", None),
            candidate_k=heavy_cfg.get("candidate_k", 200),
            use_rerank=heavy_cfg.get("use_rerank", True),
        )
    elif heavy_type == "hybrid_dense_bm25":
        bm25 = ElasticBM25RerankRetriever(
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
            rerank_k=heavy_cfg.get("rerank_k", 50),
            rerank_batch_size=int(heavy_cfg.get("rerank_batch_size", 32)),
            rerank_max_doc_chars=int(heavy_cfg.get("rerank_max_doc_chars", 2000)),
            profile=bool(heavy_cfg.get("profile", False)),
        )
        heavy = HybridDenseBM25Retriever(
            dense_retriever=light,
            bm25_retriever=bm25,
            dense_top_n=int(heavy_cfg.get("dense_top_n", 50)),
            bm25_top_n=int(heavy_cfg.get("bm25_top_n", heavy_cfg.get("top_n", 50))),
            out_top_n=int(heavy_cfg.get("top_n", 10)),
            merge_mode=str(heavy_cfg.get("merge_mode", "rrf")).strip().lower(),
            dense_keep_n=int(heavy_cfg.get("dense_keep_n", 10)),
            bm25_keep_n=int(heavy_cfg.get("bm25_keep_n", 3)),
            rrf_k=int(heavy_cfg.get("rrf_k", 60)),
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
        )
    else:
        raise ValueError(f"Unknown heavy_retriever.type={heavy_type}")

    try:
        heavy.retrieve("warmup query")
        print("[Warmup] heavy retriever initialized in main thread.")
    except Exception as e:
        print(f"[Warmup] heavy init failed: {repr(e)}")
        raise

    llm_cfg = cfg.get("llm", {}) or {}
    backend = (llm_cfg.get("backend", "vllm") or "vllm").lower()

    if backend == "vllm":
        vcfg = _build_vllm_config(llm_cfg)
        llm = VllmLLM(vcfg)
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

    # --------------------------
    # system + optimizer
    # --------------------------
    judge_cfg = cfg.get("judge_llm", None)
    judge_llm = None
    if judge_cfg:
        j = judge_cfg
        j_backend = (j.get("backend", "hf") or "hf").lower()
        if j_backend != "hf":
            raise ValueError("judge_llm currently supports backend=hf only (recommended for Qwen1.5-14B-Chat).")

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
    exec_mode = str(sys_cfg.get("exec_mode", "serial")).strip().lower()
    heavy_max_workers = int(sys_cfg.get("heavy_max_workers", 8))
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
        exec_mode=exec_mode,
        heavy_max_workers=heavy_max_workers,
    )

    opt_cfg = cfg.get("optimizer", {}) or {}
    delta = float(opt_cfg.get("delta", 0.05))
    radius_r = opt_cfg.get("radius_r", opt_cfg.get("r", None))
    gamma_cfg = opt_cfg.get("gamma", None)
    strict_perturbation_check = bool(opt_cfg.get("strict_perturbation_check", True))
    feasibility_tol = float(opt_cfg.get("feasibility_tol", 1e-8))

    optimizer = AdaRAGBanditOptimizer(
        n_docs=n_docs,
        Q_target=float(cfg.get("quality_target", 0.0)),
        alpha=float(opt_cfg.get("alpha", 0.08)),
        mu=float(opt_cfg.get("mu", 0.3)),
        delta=delta,
        gamma=None if gamma_cfg is None else float(gamma_cfg),
        radius_r=None if radius_r is None else float(radius_r),
        seed=seed,
        strict_perturbation_check=strict_perturbation_check,
        feasibility_tol=feasibility_tol,
    )

    print(
        "[Optimizer] "
        f"alpha={optimizer.alpha}, mu={optimizer.mu}, delta={optimizer.delta}, "
        f"gamma={optimizer.gamma}, radius_r={optimizer.radius_r}, "
        f"strict_perturbation_check={optimizer.strict_perturbation_check}"
    )

    decisions_path = os.path.join(out_dir, "decisions.jsonl")
    metrics_path = os.path.join(out_dir, "slot_metrics.csv")
    examples_path = os.path.join(out_dir, "slot_examples.jsonl")

    if args.overwrite:
        if os.path.exists(decisions_path):
            open(decisions_path, "w", encoding="utf-8").close()
        if os.path.exists(examples_path):
            open(examples_path, "w", encoding="utf-8").close()

    rows = []
    lam = float(optimizer.state.lambda_)

    for t in tqdm(range(n_slots), desc="Slots"):
        raw_batch = qa_used[t * slot_size: (t + 1) * slot_size]
        batch = [_make_qa(x) for x in raw_batch]

        z_perturbed, u, z_hat = optimizer.propose()
        res = system.run_slot(batch=batch, z=z_perturbed, latency_target_s=1e9)

        judge_acc = res.get("judge_accuracy", None)
        if judge_acc is not None:
            q_feedback = float(judge_acc)
        else:
            q_feedback = float(res["accuracy"])

        d_feedback_raw = float(res["latency_s"])
        d_feedback_out = float(f"{d_feedback_raw:.2f}")

        dbg = optimizer.update(
            z_perturbed=z_perturbed,
            u=u,
            d_feedback=d_feedback_raw,
            q_feedback=q_feedback,
        )
        lam = float(optimizer.state.lambda_)

        with open(examples_path, "a", encoding="utf-8") as fex:
            for ex in res.get("examples", []):
                ex = dict(ex)
                ex["slot"] = int(t)
                ex["lambda"] = float(lam)
                ex["p"] = float(res.get("p", 0.0))
                fex.write(json.dumps(ex, ensure_ascii=False) + "\n")

        row = {
            "slot": t,
            "latency_s": d_feedback_out,
            "accuracy": q_feedback,
            "p": float(res.get("p", 0.0)),
            "heavy_frac_real": float(res.get("heavy_frac_real", 0.0)),
            "avg_docs_light": float(res.get("avg_docs_light", 0.0)),
            "avg_docs_heavy": float(res.get("avg_docs_heavy", 0.0)),
            "oracle_recall_any": float(res.get("oracle_recall_any", 0.0)),
            "prompt_gold_rate": float(res.get("prompt_gold_rate", 0.0)),
            "lambda": float(lam),
            "gamma": float(optimizer.gamma),
            "radius_r": float(optimizer.radius_r),
            "perturbation_norm": float(dbg.get("perturbation_norm", np.nan)),
            "projection_gap_perturbed": float(dbg.get("projection_gap_perturbed", np.nan)),
            "accuracy_contains": float(res.get("accuracy", 0.0)),
            "oracle_recall_any_full": float(res.get("oracle_recall_any_full", 0.0)),
            "oracle_recall_light_full": float(res.get("oracle_recall_light_full", 0.0)),
            "oracle_recall_heavy_full": float(res.get("oracle_recall_heavy_full", 0.0)),
        }
        rows.append(row)

        with open(decisions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "t": t + 1,
                "q_feedback": q_feedback,
                "d_feedback": d_feedback_out,
                "z_hat_prev": dbg.get("z_hat_prev", None),
                "z_hat": dbg.get("z_hat", None),
                "z_perturbed": dbg.get("z_perturbed", None),
                "lambda": float(lam),
                "gamma": float(optimizer.gamma),
                "radius_r": float(optimizer.radius_r),
                "perturbation_norm": float(dbg.get("perturbation_norm", np.nan)),
                "projection_gap_perturbed": float(dbg.get("projection_gap_perturbed", np.nan)),
                "corr": float(dbg.get("corr", np.nan)),
            }, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False)

    print("\n=== Done ===")
    print(df.tail().round(4))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {decisions_path}")
    if os.path.exists(examples_path):
        print(f"Saved: {examples_path}")


if __name__ == "__main__":
    main()
