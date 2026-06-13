from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import concurrent.futures as cf
import json
import shutil
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DEFAULT = "/home/yy/adarag_repro"
if ROOT_DEFAULT not in sys.path:
    sys.path.insert(0, ROOT_DEFAULT)

from adarag.data import QAItem
from adarag.data_hf import load_nq_open_stream
from metis_hnsw.config_selector import ResourceAwareConfigSelector, SelectorConfig
from metis_hnsw.evaluator import score_answer
from metis_hnsw.judge import JudgeConfig, StrictQAJudge
from metis_hnsw.profiler import OnlineAPIProfiler, ProfilerConfig
from metis_hnsw.retrievers import build_hnsw_retriever_from_cfg
from metis_hnsw.synthesis import MetisSynthesizer, SynthesisConfig
from metis_hnsw.token_counter import TokenCounter
from metis_hnsw.utils import append_jsonl, doc_to_record, ensure_dir, load_metis_config, percentile, save_yaml
from metis_hnsw.vllm_client import VLLMChatClient, VLLMClientConfig
from metis_hnsw.vllm_state import VLLMStateConfig, VLLMStateObserver

DEFAULT_CONFIG = "/home/yy/adarag_repro/scripts/config_metis_hnsw_online.yaml"


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


def _make_qa(x: Any, idx: int) -> QAItem:
    q = _get_field(x, ["q", "question", "query"], default="") or ""
    a = _get_field(x, ["a", "answers", "answer", "gold", "ground_truth"], default=None)
    qid = _get_field(x, ["qid", "id", "example_id"], default=None)
    return QAItem(q=str(q), a=_as_answers(a), qid=str(qid if qid is not None else idx))


def _load_questions(cfg: Dict[str, Any], n_needed: int) -> List[QAItem]:
    ds_cfg = cfg["dataset"]
    it = load_nq_open_stream(
        split=ds_cfg.get("split", "validation"),
        seed=int(ds_cfg.get("seed", cfg.get("seed", 42))),
        local_path=ds_cfg["local_path"],
        max_examples=ds_cfg.get("max_examples", None),
    )
    out: List[QAItem] = []
    for idx, x in enumerate(it):
        out.append(_make_qa(x, idx))
        if len(out) >= n_needed:
            break
    if not out:
        raise RuntimeError("Dataset stream is empty. Check dataset.local_path and max_examples.")
    return out


def _make_slot_sizes(cfg: Dict[str, Any], out_dir: str) -> List[int]:
    n_slots = int(cfg.get("n_slots", 10))
    default_slot_size = int(cfg.get("slot_size", 50))
    ss_cfg = cfg.get("slot_sizing", {}) or {}
    mode = str(ss_cfg.get("mode", "fixed")).strip().lower()
    if mode == "fixed":
        return [default_slot_size for _ in range(n_slots)]
    if mode != "random":
        raise ValueError(f"slot_sizing.mode must be fixed or random, got {mode}")
    min_size = int(ss_cfg.get("min_size", default_slot_size))
    max_size = int(ss_cfg.get("max_size", default_slot_size))
    if min_size <= 0 or max_size < min_size:
        raise ValueError(f"Invalid slot_sizing range: {min_size}, {max_size}")
    path = ss_cfg.get("path")
    if path:
        path = os.path.abspath(os.path.expanduser(str(path)))
    else:
        path = os.path.join(out_dir, "slot_sizes.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            sizes = [int(x) for x in json.load(f)]
        if len(sizes) != n_slots:
            raise ValueError(f"slot_sizes length={len(sizes)} != n_slots={n_slots}: {path}")
        return sizes
    rng = np.random.RandomState(int(ss_cfg.get("seed", cfg.get("seed", 42))))
    sizes = rng.randint(min_size, max_size + 1, size=n_slots).astype(int).tolist()
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sizes, f, ensure_ascii=False, indent=2)
    return sizes


def _slice_batches(questions: List[QAItem], sizes: List[int]) -> List[List[QAItem]]:
    out, off = [], 0
    for s in sizes:
        out.append(questions[off: off + int(s)])
        off += int(s)
    return out


def _state_to_dict(state) -> Dict[str, Any]:
    return {
        "source": state.source,
        "gpu_cache_usage_perc": state.gpu_cache_usage_perc,
        "num_requests_running": state.num_requests_running,
        "num_requests_waiting": state.num_requests_waiting,
        "num_batched_tokens": state.num_batched_tokens,
        "gpu_mem_used_mib": state.gpu_mem_used_mib,
        "gpu_mem_total_mib": state.gpu_mem_total_mib,
        "gpu_mem_free_mib": state.gpu_mem_free_mib,
        "gpu_util_percent": state.gpu_util_percent,
    }


def _run_one_request(
    *,
    qa: QAItem,
    local_idx: int,
    slot_id: int,
    slot_t0: float,
    profiler: OnlineAPIProfiler,
    retriever,
    state_observer: VLLMStateObserver,
    selector: ResourceAwareConfigSelector,
    synthesizer: MetisSynthesizer,
    top_k: int,
    save_docs: bool,
    save_raw_outputs: bool,
) -> Dict[str, Any]:
    worker_start_rel = time.perf_counter() - slot_t0

    profile = profiler.profile(qa.q)

    ret = retriever.retrieve(qa.q)
    docs = ret.docs[: int(top_k)]

    t_sel0 = time.perf_counter()
    state = state_observer.observe()
    choice, candidate_records = selector.select(question=qa.q, docs=docs, profile=profile, state=state)
    config_selection_time_s = time.perf_counter() - t_sel0

    gen = synthesizer.run(question=qa.q, docs=docs, choice=choice)

    component_latency = (
        float(profile.profiler_time_s)
        + float(ret.retrieval_time_s)
        + float(config_selection_time_s)
        + float(gen.generation_time_s)
    )
    slot_e2e_latency = time.perf_counter() - slot_t0

    scores = score_answer(gen.answer, qa.a)

    row: Dict[str, Any] = {
        "slot_id": int(slot_id),
        "local_idx": int(local_idx),
        "qid": qa.qid,
        "question": qa.q,
        "gold_answers": qa.a,
        "answer": gen.answer,
        "profiler_time_s": float(profile.profiler_time_s),
        "retrieval_time_s": float(ret.retrieval_time_s),
        "config_selection_time_s": float(config_selection_time_s),
        "generation_time_s": float(gen.generation_time_s),
        "component_total_latency_s": float(component_latency),
        "scheduler_queue_wait_s": float(worker_start_rel),
        "total_latency_s": float(worker_start_rel + component_latency),
        "slot_e2e_latency_s": float(slot_e2e_latency),
        "selected_method": choice.method,
        "selected_k": int(choice.k),
        "selected_intermediate_length": choice.intermediate_length,
        "fit_reason": choice.fit_reason,
        "estimated_input_tokens": int(choice.estimated_input_tokens),
        "estimated_output_tokens": int(choice.estimated_output_tokens),
        "estimated_token_cost": int(choice.estimated_token_cost),
        "profile": profile.__dict__,
        "vllm_state": _state_to_dict(state),
        "candidate_records": candidate_records,
        "exact": scores["exact"],
        "contains": scores["contains"],
        "token": scores["token"],
        "f1": scores["f1"],
    }
    if save_docs:
        row["retrieved_docs"] = [doc_to_record(d) for d in docs]
    if save_raw_outputs:
        row["raw_outputs"] = gen.raw_outputs
    return row


def _summarize(rows: List[Dict[str, Any]], acc_mode: str, quality_score_mode: str = "auto") -> Dict[str, Any]:
    lat = [float(r["total_latency_s"]) for r in rows]
    comp_lat = [float(r["component_total_latency_s"]) for r in rows]
    methods = Counter(str(r.get("selected_method")) for r in rows)
    ks = Counter(str(r.get("selected_k")) for r in rows)
    judge_vals = [float(r["judge_correct"]) for r in rows if r.get("judge_correct", None) is not None]
    judge_available = len(judge_vals) > 0
    quality_score_mode = str(quality_score_mode or "auto").lower()
    if quality_score_mode == "judge" and judge_available:
        q_vals = judge_vals
    elif quality_score_mode == "auto" and judge_available:
        q_vals = judge_vals
    else:
        q_vals = [float(r.get(acc_mode, r.get("token", 0.0))) for r in rows]

    out = {
        "request_count": len(rows),
        "mean_latency_s": float(np.mean(lat)) if lat else 0.0,
        "p50_latency_s": percentile(lat, 50),
        "p95_latency_s": percentile(lat, 95),
        "mean_component_total_latency_s": float(np.mean(comp_lat)) if comp_lat else 0.0,
        "mean_profiler_time_s": float(np.mean([float(r["profiler_time_s"]) for r in rows])) if rows else 0.0,
        "mean_retrieval_time_s": float(np.mean([float(r["retrieval_time_s"]) for r in rows])) if rows else 0.0,
        "mean_config_selection_time_s": float(np.mean([float(r["config_selection_time_s"]) for r in rows])) if rows else 0.0,
        "mean_generation_time_s": float(np.mean([float(r["generation_time_s"]) for r in rows])) if rows else 0.0,
        "exact": float(np.mean([float(r["exact"]) for r in rows])) if rows else 0.0,
        "contains": float(np.mean([float(r["contains"]) for r in rows])) if rows else 0.0,
        "token": float(np.mean([float(r["token"]) for r in rows])) if rows else 0.0,
        "f1": float(np.mean([float(r["f1"]) for r in rows])) if rows else 0.0,
        "judge_accuracy": float(np.mean(judge_vals)) if judge_available else None,
        "mean_judge_time_s": float(np.mean([float(r.get("judge_time_s", 0.0)) for r in rows])) if rows else 0.0,
        "quality_score": float(np.mean(q_vals)) if q_vals else 0.0,
        "quality_score_mode": ("judge" if ((quality_score_mode in ("auto", "judge")) and judge_available) else acc_mode),
        "method_counts_json": json.dumps(dict(methods), ensure_ascii=False),
        "k_counts_json": json.dumps(dict(ks), ensure_ascii=False),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, default=None, help="Override total number of queries for smoke tests.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_metis_config(args.config)
    project_root = os.path.abspath(os.path.expanduser(str(cfg.get("project_root", ROOT_DEFAULT))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    out_dir = os.path.abspath(os.path.expanduser(str(cfg.get("output_dir", "outputs_metis_hnsw_online"))))
    ensure_dir(out_dir)
    if args.overwrite:
        for name in ["metis_hnsw_details.jsonl", "metis_hnsw_summary.csv", "slot_metrics.csv", "run_config_used.yaml"]:
            p = os.path.join(out_dir, name)
            if os.path.exists(p):
                os.remove(p)
    save_yaml(os.path.join(out_dir, "run_config_used.yaml"), cfg)

    metis_cfg = cfg.get("metis", {}) or {}
    top_k = int(metis_cfg.get("top_k", 10))
    acc_mode = str(cfg.get("acc_mode", metis_cfg.get("acc_mode", "token")))
    eval_cfg = cfg.get("evaluation", {}) or {}
    quality_score_mode = str(eval_cfg.get("quality_score_mode", "auto"))

    slot_sizes = _make_slot_sizes(cfg, out_dir)
    total_needed = int(sum(slot_sizes))
    if args.limit is not None:
        total_needed = min(total_needed, int(args.limit))
        # Convert to enough slots preserving configured slot size order.
        new_sizes = []
        remain = total_needed
        for s in slot_sizes:
            if remain <= 0:
                break
            take = min(int(s), remain)
            new_sizes.append(take)
            remain -= take
        slot_sizes = new_sizes

    questions = _load_questions(cfg, total_needed)
    if len(questions) < total_needed:
        raise RuntimeError(f"Not enough questions: need {total_needed}, got {len(questions)}")
    batches = _slice_batches(questions[:total_needed], slot_sizes)

    print("========== METIS-HNSW Online Baseline ==========")
    print(f"[config] {args.config}")
    print(f"[out_dir] {out_dir}")
    print(f"[project_root] {project_root}")
    print(f"[top_k] {top_k}")
    print(f"[slot_sizes] n_slots={len(slot_sizes)} total={sum(slot_sizes)} min={min(slot_sizes)} max={max(slot_sizes)}")
    print("[latency] total_latency_s = queue_wait + profiler + HNSW retrieval + live-state selection + synthesis generation")
    print("================================================")

    print("[init] building HNSW light retriever...")
    retriever = build_hnsw_retriever_from_cfg(cfg, top_k=top_k)
    print("[init] HNSW ready.")

    print("[init] building online API profiler...")
    profiler = OnlineAPIProfiler(ProfilerConfig.from_cfg(cfg.get("profiler", {}) or {}))
    print("[init] profiler ready.")

    print("[init] building local vLLM client...")
    llm = VLLMChatClient(VLLMClientConfig.from_cfg(cfg.get("llm", {}) or {}))
    print("[init] vLLM client ready.")

    judge_cfg_dict = cfg.get("judge", cfg.get("judge_llm", {}) or {}) or {}
    if "judge_llm" in cfg and "judge" not in cfg:
        judge_cfg_dict = dict(judge_cfg_dict)
        judge_cfg_dict.setdefault("enabled", True)
    print(f"[init] building judge evaluator enabled={bool(judge_cfg_dict.get('enabled', False))}...")
    judge = StrictQAJudge(JudgeConfig.from_cfg(judge_cfg_dict))
    print(f"[init] judge ready enabled={judge.enabled}. Judge time is recorded but excluded from online latency.")

    state_observer = VLLMStateObserver(VLLMStateConfig.from_cfg(cfg.get("vllm_state", {}) or {}))
    tokenizer_path = (cfg.get("tokenizer_path") or (cfg.get("llm", {}) or {}).get("tokenizer") or (cfg.get("llm", {}) or {}).get("model"))
    token_counter = TokenCounter(tokenizer_path)

    # METIS-specific selector/synthesis configs; do not reuse Pareto normalization/optimizer settings.
    selector_cfg_dict = dict(metis_cfg.get("selector", {}) or {})
    selector_cfg_dict.setdefault("top_k", top_k)
    selector_cfg_dict.setdefault("max_model_len", int((cfg.get("llm", {}) or {}).get("max_model_len", metis_cfg.get("max_model_len", 8192))))
    selector_cfg_dict.setdefault("prompt_max_doc_chars", int(metis_cfg.get("prompt_max_doc_chars", (cfg.get("system", {}) or {}).get("prompt_max_doc_chars", 1600))))
    selector_cfg_dict.setdefault("max_answer_tokens", int((cfg.get("llm", {}) or {}).get("max_new_tokens", 16)))
    selector = ResourceAwareConfigSelector(SelectorConfig.from_cfg(selector_cfg_dict), token_counter=token_counter)

    synth_cfg_dict = dict(metis_cfg.get("synthesis", {}) or {})
    synth_cfg_dict.setdefault("prompt_max_doc_chars", selector.cfg.prompt_max_doc_chars)
    synth_cfg_dict.setdefault("max_answer_tokens", int((cfg.get("llm", {}) or {}).get("max_new_tokens", 16)))
    synthesizer = MetisSynthesizer(llm, SynthesisConfig.from_cfg(synth_cfg_dict))

    request_max_workers = int(metis_cfg.get("request_max_workers", (cfg.get("system", {}) or {}).get("llm_max_workers", 8)))
    save_docs = bool(metis_cfg.get("save_retrieved_docs", False))
    save_raw_outputs = bool(metis_cfg.get("save_raw_outputs", False))

    details_path = os.path.join(out_dir, "metis_hnsw_details.jsonl")
    summary_path = os.path.join(out_dir, "metis_hnsw_summary.csv")
    slot_metrics_path = os.path.join(out_dir, "slot_metrics.csv")

    all_rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []

    for slot_id, batch in tqdm(list(enumerate(batches)), desc="METIS-HNSW slots"):
        slot_t0 = time.perf_counter()
        rows: List[Dict[str, Any]] = []
        max_workers = max(1, min(request_max_workers, len(batch)))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for i, qa in enumerate(batch):
                futs.append(ex.submit(
                    _run_one_request,
                    qa=qa,
                    local_idx=i,
                    slot_id=slot_id,
                    slot_t0=slot_t0,
                    profiler=profiler,
                    retriever=retriever,
                    state_observer=state_observer,
                    selector=selector,
                    synthesizer=synthesizer,
                    top_k=top_k,
                    save_docs=save_docs,
                    save_raw_outputs=save_raw_outputs,
                ))
            for fut in cf.as_completed(futs):
                row = fut.result()
                rows.append(row)

        rows = sorted(rows, key=lambda r: int(r["local_idx"]))
        judge_wall_time_s = judge.judge_rows(rows)
        for row in rows:
            row["judge_wall_time_s_slot"] = float(judge_wall_time_s)
            append_jsonl(details_path, row)

        all_rows.extend(rows)
        slot_summary = _summarize(rows, acc_mode=acc_mode, quality_score_mode=quality_score_mode)
        slot_summary.update({"slot_id": int(slot_id), "slot_wall_time_s": float(time.perf_counter() - slot_t0), "slot_judge_wall_time_s": float(judge_wall_time_s)})
        slot_rows.append(slot_summary)
        pd.DataFrame(slot_rows).to_csv(slot_metrics_path, index=False)
        print(
            f"[slot {slot_id}] B={len(rows)} mean={slot_summary['mean_latency_s']:.4f}s "
            f"p95={slot_summary['p95_latency_s']:.4f}s q={slot_summary['quality_score']:.4f} "
            f"judge={slot_summary.get('judge_accuracy', None)} "
            f"methods={slot_summary['method_counts_json']}",
            flush=True,
        )

    summary = _summarize(all_rows, acc_mode=acc_mode, quality_score_mode=quality_score_mode)
    summary.update({
        "n_slots": len(slot_sizes),
        "slot_sizes_json": json.dumps(slot_sizes),
        "top_k": top_k,
        "acc_mode": acc_mode,
        "quality_score_mode_requested": quality_score_mode,
        "judge_enabled": bool(judge.enabled),
    })
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print("========== METIS-HNSW Summary ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"[details] {details_path}")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
