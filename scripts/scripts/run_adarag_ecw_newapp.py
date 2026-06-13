# -*- coding: utf-8 -*-
"""
Run AdaRAG on ECW_newapp trace in an offline-friendly, robust way.

Aligned with run_adarag_real.py style:
- VllmConfig kwargs filtered by signature
- Dataset items normalized to QAItem
- Robust compat wrappers for propose()/update()/run_slot() signature drift
- Hourly-slot semantics: 1 hour -> 1 slot, propose once per hour, update once per hour
- ECW load trace -> N queries per hour via: N_h = arrival_scale * (100*load + 50)

IMPORTANT (your requirement):
- Only use bw_upload time series from ECW_newapp.pkl (ignore all other 15 fields).
"""

from __future__ import annotations

import os
import json
import time
import pickle
import argparse
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adarag.utils import load_yaml
from adarag.data import QAItem
from adarag.data_hf import load_nq_open_stream
from adarag.optimizer.bandit_optimizer import AdaRAGBanditOptimizer
from adarag.retrievers.faiss_hnsw import FaissHNSWRetriever
from adarag.retrievers.heavy_bm25 import HeavyBM25Retriever
from adarag.retrievers.heavy_es_bm25_rerank import ElasticBM25RerankRetriever
from adarag.llm.vllm_llm import VllmLLM, VllmConfig
from adarag.llm.hf_llm import HFTextLLM

# 兼容不同文件名/路径的系统实现
try:
    from adarag.pipeline.adarag_system_addfold import AdaRAGSystemCDF
except Exception:
    try:
        from adarag.pipeline.adarag_system_cdf import AdaRAGSystemCDF
    except Exception:
        from adarag.pipeline.adarag_system import AdaRAGSystemCDF


# ----------------------------
# utils (aligned with real-min)
# ----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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


def _to_jsonable(x: Any):
    """numpy / 标量 转成 json 可写类型"""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    return x


def _build_vllm_config(llm_cfg: dict) -> VllmConfig:
    llm_cfg = llm_cfg or {}
    candidate = {
        "model": llm_cfg.get("model"),
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


def _build_llm(llm_cfg: dict):
    llm_cfg = llm_cfg or {}
    backend = (llm_cfg.get("backend", "hf") or "hf").lower()

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


def _build_judge(judge_cfg: Optional[dict]):
    if not judge_cfg:
        return None
    j = judge_cfg
    j_backend = (j.get("backend", "hf") or "hf").lower()
    if j_backend != "hf":
        raise ValueError("judge_llm currently supports backend=hf only.")

    gen_kwargs = {
        "temperature": float(j.get("temperature", 0.0)),
        "top_p": float(j.get("top_p", 1.0)),
        "top_k": int(j.get("top_k", 0)),
        "repetition_penalty": float(j.get("repetition_penalty", 1.0)),
        "max_new_tokens": int(j.get("max_new_tokens", 8)),
    }
    return HFTextLLM(
        model_path=j["model"],
        tokenizer_path=j.get("tokenizer", j["model"]),
        device=j.get("device", "cuda"),
        dtype=j.get("dtype", "float16"),
        max_new_tokens=int(j.get("max_new_tokens", 8)),
        max_model_len=int(j.get("max_model_len", 8192)),
        use_chat_template=bool(j.get("use_chat_template", True)),
        system_prompt=j.get("system_prompt", "You are a strict evaluator."),
        gen_kwargs=gen_kwargs,
        trust_remote_code=bool(j.get("trust_remote_code", False)),
        local_files_only=bool(j.get("local_files_only", True)),
    )


# ----------------------------
# compat wrappers
# ----------------------------
def _call_compat(fn, **kwargs):
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filtered)


def _propose_compat(opt):
    """兼容 propose() 可能返回 (z,u,dbg) / (z,u) / dict / z"""
    out = opt.propose()
    if isinstance(out, dict):
        z = out.get("z_perturbed", out.get("z", None))
        u = out.get("u", None)
        return z, u, out
    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            return out[0], out[1], out[2]
        if len(out) == 2:
            return out[0], out[1], {}
        if len(out) == 1:
            return out[0], None, {}
    return out, None, {}


def _update_compat(opt, **kwargs):
    return _call_compat(opt.update, **kwargs)


def _get_metric(res: dict, keys, default=0.0):
    if not isinstance(res, dict):
        return default
    for k in keys:
        if k in res and res[k] is not None:
            return res[k]
    m = res.get("metrics", None)
    if isinstance(m, dict):
        for k in keys:
            if k in m and m[k] is not None:
                return m[k]
    return default


# ----------------------------
# data loading
# ----------------------------
def _load_qa_pool(ds_cfg: dict, seed: int) -> List[QAItem]:
    it = load_nq_open_stream(
        split=ds_cfg.get("split", "validation"),
        seed=int(ds_cfg.get("seed", seed)),
        local_path=ds_cfg["local_path"],
        max_examples=ds_cfg.get("max_examples", None),
    )
    raw = list(it)
    if not raw:
        raise RuntimeError("QA pool is empty. Check dataset.local_path / max_examples.")
    return [_make_qa(x) for x in raw]


# ----------------------------
# ECW trace loading (ONLY bw_upload)
# ----------------------------
def _extract_bw_upload(arr: np.ndarray) -> List[np.ndarray]:
    """
    Only keep bw_upload series.

    ECW_newapp.pkl (common):
      shape = (N, T, F) e.g. (11, 720, 16), bw_upload is feature-0.

    Also supports:
      (T, F) -> take [:,0]
      (N, T) -> each row is one trace
      (T,)   -> already a trace
      (N, F, T) -> take [ :,0,: ] if detected
    """
    a = np.asarray(arr, dtype=float)

    if a.ndim == 1:
        return [a.reshape(-1)]

    if a.ndim == 2:
        # (T,F): usually T is big (>=100) and F is small (<=64)
        T, F = a.shape[0], a.shape[1]
        if (T >= 100) and (F <= 64):
            return [a[:, 0].reshape(-1)]
        # (N,T): each row a trace
        return [a[i, :].reshape(-1) for i in range(a.shape[0])]

    if a.ndim == 3:
        N, A, B = a.shape[0], a.shape[1], a.shape[2]
        # (N,T,F)
        if (A >= 100) and (B <= 64):
            return [a[i, :, 0].reshape(-1) for i in range(N)]
        # (N,F,T)
        if (A <= 64) and (B >= 100):
            return [a[i, 0, :].reshape(-1) for i in range(N)]
        # fallback: assume last dim is feature
        return [a[i, :, 0].reshape(-1) for i in range(N)]

    # avoid flattening fields; try squeeze then recurse
    a2 = np.squeeze(a)
    if a2.ndim == a.ndim:
        # last resort: treat first axis as trace axis and keep 1D
        a3 = a.reshape(a.shape[0], -1)
        return [a3[i, :].reshape(-1) for i in range(a3.shape[0])]
    return _extract_bw_upload(a2)


def _load_ecw_traces_from_any(x: Any) -> List[np.ndarray]:
    if isinstance(x, np.ndarray):
        return _extract_bw_upload(x)

    if isinstance(x, (list, tuple)):
        out: List[np.ndarray] = []
        for z in x:
            if isinstance(z, np.ndarray):
                out.extend(_extract_bw_upload(z))
            elif isinstance(z, (list, tuple)):
                out.append(np.asarray(z, dtype=float).reshape(-1))
            elif isinstance(z, dict):
                for k in ("load", "loads", "trace", "series", "y"):
                    if k in z:
                        out.append(np.asarray(z[k], dtype=float).reshape(-1))
                        break
        out = [np.asarray(a, dtype=float).reshape(-1) for a in out if a is not None]
        return [a for a in out if a.size > 1] or out

    if isinstance(x, dict):
        for k in ("data", "traces", "loads", "series"):
            if k in x:
                return _load_ecw_traces_from_any(x[k])
        for k in ("load", "loads", "trace", "series", "y"):
            if k in x:
                return [np.asarray(x[k], dtype=float).reshape(-1)]
        if x:
            return _load_ecw_traces_from_any(next(iter(x.values())))

    return []


def _load_ecw_traces(pkl_path: str) -> List[np.ndarray]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    traces = _load_ecw_traces_from_any(obj)
    if not traces:
        raise ValueError(f"Unrecognized ECW_newapp.pkl structure: {type(obj)}")
    return traces


# ----------------------------
# hourly mapping
# ----------------------------
def _n_queries_for_hour(load_val: float, arrival_scale: float, rounding: str = "round") -> int:
    """
    N_h = arrival_scale * (100*load + 50)
    rounding: "round" | "floor" | "ceil"
    """
    lv = float(load_val)
    if not np.isfinite(lv):
        lv = 0.0
    raw = float(arrival_scale) * (100.0 * lv + 50.0)
    raw = max(0.0, raw)

    r = (rounding or "round").lower().strip()
    if r == "floor":
        return int(np.floor(raw))
    if r == "ceil":
        return int(np.ceil(raw))
    return int(np.round(raw))


# ----------------------------
# retrievers builder
# ----------------------------
def _build_retrievers(cfg: dict, n_docs: int):
    light_cfg = cfg.get("light_retriever", {}) or {}
    heavy_cfg = cfg.get("heavy_retriever", {}) or {}

    light = FaissHNSWRetriever(
        corpus_path=light_cfg["corpus_path"],
        index_path=light_cfg["index_path"],
        embedding_model=light_cfg.get("embedding_model", None),
        top_n=int(light_cfg.get("top_n", n_docs)),
        device=light_cfg.get("device", "cuda"),
    )
    light.load_or_build(
        rebuild=bool(light_cfg.get("rebuild", False)),
        max_passages=light_cfg.get("max_passages", None),
    )

    heavy_type = heavy_cfg.get("type", "bm25_es")
    if heavy_type == "bm25_es":
        heavy = ElasticBM25RerankRetriever(
            es_url=heavy_cfg.get("es_url", "http://localhost:9200"),
            index_name=heavy_cfg["index_name"],
            top_n=int(heavy_cfg.get("top_n", n_docs)),
            bm25_k=int(heavy_cfg.get("bm25_k", 50)),
            reranker_model=heavy_cfg.get("reranker_model"),
            device=heavy_cfg.get("device", "cpu"),
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
            top_n=int(heavy_cfg.get("top_n", n_docs)),
            max_passages=heavy_cfg.get("max_passages", None),
            candidate_k=heavy_cfg.get("candidate_k", 200),
            use_rerank=heavy_cfg.get("use_rerank", True),
        )
    else:
        raise ValueError(f"Unknown heavy_retriever.type={heavy_type}")

    return light, heavy


# ----------------------------
# core runner
# ----------------------------
def run_one_trace_hourly(
    trace_id: int,
    load_trace: np.ndarray,
    qa_pool: List[QAItem],
    system: Any,
    optimizer: AdaRAGBanditOptimizer,
    out_dir: str,
    *,
    arrival_scale: float,
    micro_batch: int,
    max_hours: Optional[int],
    seed: int,
    rounding: str = "round",
    time_round_digits: int = 2,
    overwrite: bool = False,
    save_examples: bool = False,
    examples_max_per_hour: int = 20,
) -> None:
    _ensure_dir(out_dir)

    decisions_path = os.path.join(out_dir, f"decisions_ecw_trace{trace_id}.jsonl")
    metrics_path = os.path.join(out_dir, f"metrics_ecw_trace{trace_id}.csv")
    examples_path = os.path.join(out_dir, f"examples_ecw_trace{trace_id}.jsonl")

    if overwrite:
        if os.path.exists(decisions_path):
            open(decisions_path, "w", encoding="utf-8").close()
        if os.path.exists(metrics_path):
            try:
                os.remove(metrics_path)
            except Exception:
                pass
        if save_examples and os.path.exists(examples_path):
            open(examples_path, "w", encoding="utf-8").close()

    H = int(len(load_trace))
    if max_hours is not None:
        H = min(H, int(max_hours))

    qa_ptr = 0
    rows: List[Dict[str, Any]] = []

    for hour_idx in range(H):
        load_val = float(load_trace[hour_idx])
        n_hour = _n_queries_for_hour(load_val, arrival_scale=float(arrival_scale), rounding=rounding)

        sim_time_s = float((hour_idx + 1) * 3600)

        # N=0：跳过本小时，不更新 bandit（保持 lambda 不变）
        if n_hour <= 0:
            rows.append(
                {
                    "trace_id": int(trace_id),
                    "hour_idx": int(hour_idx),
                    "sim_time_s": round(sim_time_s, time_round_digits),
                    "load_value": float(load_val),
                    "arrival_scale": float(arrival_scale),
                    "n_queries_hour": int(n_hour),
                    "micro_batch": int(micro_batch),
                    "wall_total_s": 0.0,
                    "avg_latency_per_q_s": 0.0,
                    "throughput_qps": 0.0,
                    "accuracy_feedback": 0.0,
                    "p": 0.0,
                    "heavy_frac_real": 0.0,
                    "avg_docs_light": 0.0,
                    "avg_docs_heavy": 0.0,
                    "oracle_recall_any": 0.0,
                    "prompt_gold_rate": 0.0,
                    "lambda": float(optimizer.state.lambda_),
                }
            )
            continue

        # 每小时：propose 1 次 z，整小时复用
        z_perturbed, u, dbg0 = _propose_compat(optimizer)
        z_hat_guess = dbg0.get("z_hat", None) if isinstance(dbg0, dict) else None

        q_feedback_ac=0.3      #1加权计数准确率
        wall_total = 0.0
        w_acc_sum = 0.0
        w_p_sum = 0.0
        w_hfrac_sum = 0.0
        w_light_sum = 0.0
        w_heavy_sum = 0.0
        w_oracle_sum = 0.0
        w_gold_sum = 0.0
        processed = 0
        saved_examples_cnt = 0

        while processed < n_hour:
            cur = min(int(micro_batch), int(n_hour - processed))
            batch = [qa_pool[(qa_ptr + i) % len(qa_pool)] for i in range(cur)]
            qa_ptr += cur
            processed += cur

            t0 = time.perf_counter()
            res = _call_compat(
                system.run_slot,
                batch=batch,
                z=z_perturbed,
                latency_target_s=1e9,
                return_timings=False,
                verbose=False,
            )
            t1 = time.perf_counter()

            wall_s = max(t1 - t0, 1e-9)
            wall_total += wall_s

            q_feedback = float(_get_metric(res, ["judge_accuracy", "accuracy"], default=0.0))
            w_acc_sum += q_feedback * cur

            w_p_sum += float(_get_metric(res, ["p"], 0.0)) * cur
            w_hfrac_sum += float(_get_metric(res, ["heavy_frac_real", "heavy_frac"], 0.0)) * cur
            w_light_sum += float(_get_metric(res, ["avg_docs_light"], 0.0)) * cur
            w_heavy_sum += float(_get_metric(res, ["avg_docs_heavy"], 0.0)) * cur
            w_oracle_sum += float(_get_metric(res, ["oracle_recall_any"], 0.0)) * cur
            w_gold_sum += float(_get_metric(res, ["prompt_gold_rate"], 0.0)) * cur

            if save_examples and isinstance(res, dict) and "examples" in res and res["examples"]:
                for ex in res["examples"]:
                    if saved_examples_cnt >= int(examples_max_per_hour):
                        break
                    exd = dict(ex)
                    exd["trace_id"] = int(trace_id)
                    exd["hour_idx"] = int(hour_idx)
                    exd["sim_time_s"] = round(sim_time_s, time_round_digits)
                    exd["lambda"] = float(optimizer.state.lambda_)
                    exd["p"] = float(_get_metric(res, ["p"], 0.0))
                    with open(examples_path, "a", encoding="utf-8") as fex:
                        fex.write(json.dumps(exd, ensure_ascii=False) + "\n")
                    saved_examples_cnt += 1

        acc_hour_raw= w_acc_sum / max(1,n_hour)  #2准确值的计算和存储
        acc_hour = float(acc_hour_raw+q_feedback_ac)  #3更新准确率
        avg_latency_per_q = wall_total / max(1, n_hour)
        throughput_qps = n_hour / max(wall_total, 1e-9)

        # bandit：每小时更新 1 次
        d_feedback = float(avg_latency_per_q)
        dbg = _update_compat(
            optimizer,
            z_perturbed=z_perturbed,
            u=u,
            d_feedback=d_feedback,
            q_feedback=float(acc_hour_raw),     #4更新bandit值
        )
        if not isinstance(dbg, dict):
            dbg = {}

        row = {
            "trace_id": int(trace_id),
            "hour_idx": int(hour_idx),
            "sim_time_s": round(sim_time_s, time_round_digits),
            "load_value": float(load_val),
            "arrival_scale": float(arrival_scale),
            "n_queries_hour": int(n_hour),
            "micro_batch": int(micro_batch),
            "wall_total_s": round(float(wall_total), time_round_digits),
            "avg_latency_per_q_s": round(float(avg_latency_per_q), time_round_digits),
            "throughput_qps": round(float(throughput_qps), time_round_digits),
            "accuracy_feedback": round(float(acc_hour), 4),
            "p": round(float(w_p_sum / n_hour), 6),
            "heavy_frac_real": round(float(w_hfrac_sum / n_hour), 6),
            "avg_docs_light": round(float(w_light_sum / n_hour), 6),
            "avg_docs_heavy": round(float(w_heavy_sum / n_hour), 6),
            "oracle_recall_any": round(float(w_oracle_sum / n_hour), 6),
            "prompt_gold_rate": round(float(w_gold_sum / n_hour), 6),
            "lambda": round(float(optimizer.state.lambda_), 6),
        }
        rows.append(row)

        z_hat_val = dbg.get("z_hat", z_hat_guess)
        z_pert_val = dbg.get("z_perturbed", z_perturbed)

        with open(decisions_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "trace_id": int(trace_id),
                        "hour_idx": int(hour_idx),
                        "sim_time_s": round(sim_time_s, time_round_digits),
                        "load_value": float(load_val),
                        "n_queries_hour": int(n_hour),
                        "micro_batch": int(micro_batch),
                        "q_feedback": float(acc_hour),
                        "d_feedback": float(avg_latency_per_q),
                        "wall_total_s": float(wall_total),
                        "z_hat": _to_jsonable(z_hat_val),
                        "z_perturbed": _to_jsonable(z_pert_val),
                        "lambda": float(optimizer.state.lambda_),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            json.dumps(
                {
                    "trace_id": int(trace_id),
                    "hour": int(hour_idx),
                    "sim_time_s": round(sim_time_s, time_round_digits),
                    "load": round(load_val, 6),
                    "N": int(n_hour),
                    "acc": round(float(acc_hour), 3),
                    "avg_lat_s": round(float(avg_latency_per_q), time_round_digits),
                    "wall_s": round(float(wall_total), time_round_digits),
                    "lambda": round(float(optimizer.state.lambda_), 3),
                },
                ensure_ascii=False,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False)

    print(f"\n[ECW trace {trace_id}] Done. hours={len(df)}")
    print(df.tail(5))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {decisions_path}")
    if save_examples:
        print(f"Saved: {examples_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--trace_ids", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))

    out_root = cfg.get("output_dir", "outputs_ecw_newapp")
    _ensure_dir(out_root)

    # QA pool
    qa_pool = _load_qa_pool(cfg["dataset"], seed=seed)

    # retrievers
    n_docs = int(cfg.get("n_docs", 10))
    light, heavy = _build_retrievers(cfg, n_docs=n_docs)

    # llm + judge
    llm = _build_llm(cfg.get("llm", {}) or {})
    judge_llm = _build_judge(cfg.get("judge_llm", None))

    # system
    sys_cfg = cfg.get("system", {}) or {}
    system = AdaRAGSystemCDF(
        light_retriever=light,
        heavy_retriever=heavy,
        llm=llm,
        n_docs=n_docs,
        acc_mode=cfg.get("acc_mode", "token"),
        seed=seed,
        force_heavy=bool(sys_cfg.get("force_heavy", False)),
        prompt_max_doc_chars=int(sys_cfg.get("prompt_max_doc_chars", 6000)),
        judge_llm=judge_llm,
        exec_mode=str(sys_cfg.get("exec_mode", "overlap")).strip().lower(),
        heavy_max_workers=int(sys_cfg.get("heavy_max_workers", 8)),
    )

    # optimizer factory（每条 trace 单独一个 optimizer，避免相互污染）
    opt_cfg = cfg.get("optimizer", {}) or {}

    def _new_optimizer():
        return AdaRAGBanditOptimizer(
            n_docs=n_docs,
            Q_target=float(cfg.get("quality_target", 0.0)),
            alpha=float(opt_cfg.get("alpha", 0.08)),
            mu=float(opt_cfg.get("mu", 0.3)),
            delta=float(opt_cfg.get("delta", 0.05)),
            gamma=float(opt_cfg.get("gamma", 0.05)),
            seed=seed,
        )

    # ECW config
    ecw_cfg = cfg.get("ecw", {}) or {}
    pkl_path = ecw_cfg.get(
        "pkl_path", "/T20050013/adarag_repro/ECWDataset-main/ECWDataset-main/ECW_newapp.pkl"
    )
    arrival_scale = float(ecw_cfg.get("arrival_scale", 1.0))
    micro_batch = int(ecw_cfg.get("micro_batch", ecw_cfg.get("max_batch", 32)))

    # max_steps：解释为最多跑多少小时
    max_hours = ecw_cfg.get("max_steps", None)
    max_hours = int(max_hours) if max_hours is not None else None

    rounding = str(ecw_cfg.get("rounding", "round"))
    time_round_digits = int(ecw_cfg.get("time_round_digits", 2))

    save_examples = bool(ecw_cfg.get("save_examples", False))
    examples_max_per_hour = int(ecw_cfg.get("examples_max_per_hour", 20))

    traces = _load_ecw_traces(pkl_path)

    for tid in args.trace_ids:
        if tid < 0 or tid >= len(traces):
            continue
        out_dir = os.path.join(out_root, f"trace_{tid}")

        run_one_trace_hourly(
            trace_id=tid,
            load_trace=traces[tid],
            qa_pool=qa_pool,
            system=system,
            optimizer=_new_optimizer(),
            out_dir=out_dir,
            arrival_scale=arrival_scale,
            micro_batch=micro_batch,
            max_hours=max_hours,
            seed=seed,
            rounding=rounding,
            time_round_digits=time_round_digits,
            overwrite=bool(args.overwrite),
            save_examples=save_examples,
            examples_max_per_hour=examples_max_per_hour,
        )


if __name__ == "__main__":
    main()
