# /home/yy/adarag_repro/scripts/run_adarag_real_mp.py

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import shutil
from typing import Any, Dict, List, Sequence, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from adarag.utils import load_yaml
from adarag.data_hf import load_nq_open_stream
from adarag.data import QAItem

from paretoadarag.optimizer.adarag_bandit_optimizer_mp import AdaRAGBanditOptimizerMP
from paretoadarag.pipeline.adarag_system_oneprobe_mp import build_system_from_cfg


DEFAULT_CONFIG = "/home/yy/adarag_repro/scripts/config_adarag_real_mp.yaml"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _as_doc_choices(v: Optional[Sequence[int]], fallback_n: int) -> List[int]:
    if v is None:
        return list(range(1, int(fallback_n) + 1))
    out = [int(x) for x in v]
    if not out:
        raise ValueError("doc choices must not be empty.")
    if min(out) < 1:
        raise ValueError(f"doc choices must be >= 1, got {out}")
    return out


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
    qid = _get_field(x, ["qid", "id", "example_id"], default="") or ""
    return QAItem(q=str(q), a=_as_answers(a), qid=str(qid))


def _load_questions(cfg: dict, n_needed: int) -> List[QAItem]:
    ds_cfg = cfg["dataset"]
    it = load_nq_open_stream(
        split=ds_cfg.get("split", "validation"),
        seed=int(ds_cfg.get("seed", cfg.get("seed", 42))),
        local_path=ds_cfg["local_path"],
        max_examples=ds_cfg.get("max_examples", None),
    )

    out: List[QAItem] = []
    for x in it:
        out.append(_make_qa(x))
        if len(out) >= n_needed:
            break

    if not out:
        raise RuntimeError("Dataset stream is empty. Check dataset.local_path and max_examples.")
    return out


def _make_slot_sizes(cfg: dict, out_dir: str) -> List[int]:
    """
    Generate or load per-slot request counts.

    fixed mode:
        [slot_size] * n_slots

    random mode:
        load slot_sizing.path if it exists; otherwise generate by seed and save.

    Important:
        AdaRAG and ParetoAdaRAG must use the same slot_sizing.path.
    """
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
        raise ValueError(f"Invalid random slot size range: min={min_size}, max={max_size}")

    path = ss_cfg.get("path", None)
    if path:
        path = os.path.abspath(os.path.expanduser(str(path)))
    else:
        path = os.path.join(out_dir, "slot_sizes.json")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            sizes = json.load(f)
        sizes = [int(x) for x in sizes]
        if len(sizes) != n_slots:
            raise ValueError(f"slot_sizes length={len(sizes)} != n_slots={n_slots}: {path}")
        if min(sizes) < min_size or max(sizes) > max_size:
            raise ValueError(
                f"slot_sizes in {path} exceed configured range [{min_size}, {max_size}]"
            )
        return sizes

    seed = int(ss_cfg.get("seed", cfg.get("seed", 42)))
    rng = np.random.RandomState(seed)
    sizes = rng.randint(min_size, max_size + 1, size=n_slots).astype(int).tolist()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sizes, f, ensure_ascii=False, indent=2)

    return sizes


def _slice_batches_by_sizes(questions: List[QAItem], slot_sizes: List[int]) -> List[List[QAItem]]:
    batches: List[List[QAItem]] = []
    offset = 0
    for s in slot_sizes:
        s = int(s)
        batches.append(questions[offset: offset + s])
        offset += s
    return batches


def _write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _save_run_config_file(config_path: str, out_dir: str) -> str:
    _ensure_dir(out_dir)
    dst = os.path.join(out_dir, "run_config_used.yaml")
    src = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.exists(src):
        raise FileNotFoundError(f"Config file not found: {src}")
    if os.path.abspath(dst) != src:
        shutil.copy2(src, dst)
    print(f"[config_snapshot] saved: {dst}")
    return dst


def _append_dummy_nu(w: np.ndarray, nu: float = 0.0) -> np.ndarray:
    """
    Shared pipeline expects [x,y,p,nu].
    Original AdaRAG has [x,y,p], so append a dummy nu.
    The dummy nu does not affect routing/doc sampling/generation.
    """
    w = np.asarray(w, dtype=float)
    return np.concatenate([w, np.array([float(nu)], dtype=float)])


def _safe_float_list(xs: Any) -> List[float]:
    """
    Convert a sequence-like object to finite float list.

    Used to collect per-query end-to-end latencies from the AdaRAG probe.
    """
    out: List[float] = []
    if xs is None:
        return out

    try:
        iterator = list(xs)
    except Exception:
        return out

    for x in iterator:
        try:
            v = float(x)
            if np.isfinite(v):
                out.append(v)
        except Exception:
            continue
    return out


def _extract_tau_e2e_from_probe_result(res: Dict[str, Any]) -> List[float]:
    """
    Extract per-query end-to-end latency from one AdaRAG probe result.

    Preferred:
        res["tau_per_request"]

    Fallback:
        res["query_rows"] with common latency column names.

    This does not change save_query_metrics behavior.
    """
    vals = _safe_float_list(res.get("tau_per_request", None))
    if vals:
        return vals

    rows = res.get("query_rows", None)
    if not rows:
        return []

    candidates = [
        "tau_e2e_s",
        "e2e_latency_s",
        "latency_s",
        "total_latency_s",
        "rt_s",
    ]

    out: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for c in candidates:
            if c in r and r[c] is not None:
                try:
                    v = float(r[c])
                    if np.isfinite(v):
                        out.append(v)
                except Exception:
                    pass
                break
    return out


def _global_p95(xs: Sequence[float]) -> float:
    vals = _safe_float_list(xs)
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=float), 95))


def _plot_adarag_curves(
    slot_df: pd.DataFrame,
    out_dir: str,
    q_target: float,
    global_p95_latency_s: Optional[float] = None,
) -> Dict[str, str]:
    out_paths: Dict[str, str] = {}
    if slot_df is None or slot_df.empty:
        return out_paths

    required_cols = {"slot", "request_count", "q_feedback", "d_feedback", "p95_latency_s"}
    if not required_cols.issubset(set(slot_df.columns)):
        missing = sorted(required_cols - set(slot_df.columns))
        print(f"[plot] skipped: missing columns {missing}")
        return out_paths

    plot_df = slot_df.sort_values("slot").copy()
    x = plot_df["slot"].to_numpy()
    q = plot_df["q_feedback"].astype(float).to_numpy()
    d = plot_df["d_feedback"].astype(float).to_numpy()
    p95 = plot_df["p95_latency_s"].astype(float).to_numpy()
    weights = plot_df["request_count"].astype(float).to_numpy()

    fig, ax1 = plt.subplots(figsize=(11.5, 6.2))

    line_q, = ax1.plot(
        x, q, marker="o", linewidth=2.0, markersize=5,
        label="q_feedback (judge accuracy)"
    )
    line_target = ax1.axhline(
        y=q_target, linestyle="--", linewidth=1.8,
        label=f"q_target = {q_target:.2f}"
    )

    ax1.set_xlabel("Slot")
    ax1.set_ylabel("q_feedback / judge accuracy")

    if len(x) <= 40:
        ax1.set_xticks(x)
    else:
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.grid(True, linestyle="--", alpha=0.35)

    q_min = float(np.nanmin(q))
    q_max = float(np.nanmax(q))
    q_lower = max(0.0, min(q_min, q_target) - 0.05)
    q_upper = min(1.0, max(q_max, q_target) + 0.05)
    if q_upper <= q_lower:
        q_upper = q_lower + 0.1
    ax1.set_ylim(q_lower, q_upper)

    ax2 = ax1.twinx()
    line_d, = ax2.plot(
        x, d, marker="s", linewidth=2.0, markersize=5,
        label="d_feedback"
    )
    line_p95, = ax2.plot(
        x, p95, marker="^", linewidth=2.0, markersize=5,
        label="p95_latency_s"
    )
    ax2.set_ylabel("Latency feedback / p95 latency (s)")

    y2_min = float(np.nanmin([np.nanmin(d), np.nanmin(p95)]))
    y2_max = float(np.nanmax([np.nanmax(d), np.nanmax(p95)]))
    margin = max(0.1, 0.08 * (y2_max - y2_min))
    ax2.set_ylim(max(0.0, y2_min - margin), y2_max + margin)

    lines = [line_q, line_target, line_d, line_p95]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best", frameon=True)

    # Use simple slot-level means in the figure title.
    # This only affects the displayed title statistics, not optimization or saved slot metrics.
    avg_q_slot = float(np.nanmean(q))
    avg_d_slot = float(np.nanmean(d))
    avg_p95_slot = float(np.nanmean(p95))
    meet = avg_q_slot >= q_target

    if global_p95_latency_s is not None and np.isfinite(float(global_p95_latency_s)):
        global_p95_part = f", global_p95={float(global_p95_latency_s):.4f}"
    else:
        global_p95_part = ""

    ax1.set_title(
        "AdaRAG: judge accuracy, average latency, and p95 latency across slots\n"
        f"avg_q_slot={avg_q_slot:.4f}, avg_d_slot={avg_d_slot:.4f}, "
        f"avg_p95_slot={avg_p95_slot:.4f}{global_p95_part}, "
        f"meet_target={meet}"
    )

    plt.tight_layout()

    png_path = os.path.join(out_dir, "adarag_slot_curves.png")
    pdf_path = os.path.join(out_dir, "adarag_slot_curves.pdf")

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    out_paths["png"] = png_path
    out_paths["pdf"] = pdf_path
    return out_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = cfg.get("output_dir", "outputs_adarag_real_mp")
    _ensure_dir(out_dir)

    seed = int(cfg.get("seed", 42))
    slot_size = int(cfg.get("slot_size", 50))
    n_slots_cfg = int(cfg.get("n_slots", 10))

    doc_choices_light = _as_doc_choices(
        cfg.get("doc_choices_light", None),
        int(cfg.get("n_docs_light", 4)),
    )
    doc_choices_heavy = _as_doc_choices(
        cfg.get("doc_choices_heavy", None),
        int(cfg.get("n_docs_heavy", 6)),
    )

    sys_cfg = cfg.get("system", {}) or {}
    judge_enabled = bool(sys_cfg.get("judge_enabled_for_g", True))
    slot_preheat = bool(sys_cfg.get("slot_preheat", True))

    log_cfg = cfg.get("logging", {}) or {}
    debug_metrics = bool(log_cfg.get("debug_metrics", False))
    save_query_metrics = bool(log_cfg.get("save_query_metrics", False))
    save_probe_metrics = bool(log_cfg.get("save_probe_metrics", True))
    save_decisions = bool(log_cfg.get("save_decisions", False))

    slot_sizes = _make_slot_sizes(cfg, out_dir)
    total_needed = int(sum(slot_sizes))

    questions = _load_questions(cfg, total_needed)
    if len(questions) < total_needed:
        raise RuntimeError(
            f"Not enough questions: need {total_needed}, got {len(questions)}. "
            f"Please increase dataset.max_examples or reduce slot sizes."
        )

    questions = questions[:total_needed]
    batches = _slice_batches_by_sizes(questions, slot_sizes)
    n_slots = len(batches)

    if n_slots <= 0:
        raise RuntimeError("Not enough questions for one slot.")

    slot_sizes_used_path = os.path.join(out_dir, "slot_sizes_used.json")
    with open(slot_sizes_used_path, "w", encoding="utf-8") as f:
        json.dump([int(x) for x in slot_sizes], f, ensure_ascii=False, indent=2)

    ss_cfg = cfg.get("slot_sizing", {}) or {}

    print("========== AdaRAG Original Real vLLM ==========")
    print(f"[config] {args.config}")
    print(f"[out_dir] {out_dir}")
    print(f"[seed] {seed}")
    print(f"[slot_size_default] {slot_size}")
    print(f"[slot_size_mode] {ss_cfg.get('mode', 'fixed')}")
    print(f"[n_slots_config] {n_slots_cfg}")
    print(f"[n_slots_used] {n_slots}")
    print(f"[slot_sizes] min={min(slot_sizes)}, max={max(slot_sizes)}, total={sum(slot_sizes)}")
    print(f"[slot_sizes_used] {slot_sizes_used_path}")
    print(f"[doc_choices_light] {doc_choices_light}")
    print(f"[doc_choices_heavy] {doc_choices_heavy}")
    print(f"[dim_w] {len(doc_choices_light) + len(doc_choices_heavy) + 1}")
    print(f"[judge_enabled] {judge_enabled}")
    print(f"[slot_preheat] {slot_preheat}")
    print(f"[heavy_max_workers] {int(sys_cfg.get('heavy_max_workers', 2))}")
    print(f"[llm_max_workers] {int(sys_cfg.get('llm_max_workers', 8))}")
    print(f"[debug_metrics] {debug_metrics}")
    print(f"[save_query_metrics] {save_query_metrics}")
    print(f"[save_probe_metrics] {save_probe_metrics}")
    print(f"[save_decisions] {save_decisions}")
    print("================================================")

    opt_cfg = cfg.get("optimizer", {}) or {}
    norm_cfg = cfg.get("normalization", {}) or {}
    q_target = float(opt_cfg.get("q_target", cfg.get("quality_target", 0.60)))

    optimizer = AdaRAGBanditOptimizerMP(
        doc_choices_light=doc_choices_light,
        doc_choices_heavy=doc_choices_heavy,
        init_doc_policy=str(opt_cfg.get("init_doc_policy", "increasing")),
        init_top0_prob=float(opt_cfg.get("init_top0_prob", 0.0)),
        p_init=float(opt_cfg.get("p_init", 0.35)),
        p_min=float(opt_cfg.get("p_min", 0.0)),
        p_max=float(opt_cfg.get("p_max", 1.0)),
        q_target=q_target,
        alpha=float(opt_cfg.get("alpha", 0.0005)),
        mu=float(opt_cfg.get("mu", 0.005)),
        delta=float(opt_cfg.get("delta", 0.05)),
        gamma=float(opt_cfg.get("gamma", 0.05)),
        seed=seed,
        normalize_enabled=bool(norm_cfg.get("enabled", False)),
        d_ref=float(norm_cfg.get("d_ref", 1.0)),
        g_ref=float(norm_cfg.get("g_ref", 1.0)),
    )

    print("[normalization]")
    print(f"  enabled = {bool(norm_cfg.get('enabled', False))}")
    print(f"  d_ref   = {float(norm_cfg.get('d_ref', 1.0))}")
    print(f"  g_ref   = {float(norm_cfg.get('g_ref', 1.0))}")

    print("[init] building system...")
    system = build_system_from_cfg(cfg, judge_enabled=judge_enabled)
    print("[init] system ready.")

    slot_metrics_path = os.path.join(out_dir, "slot_metrics.csv")
    probe_metrics_path = os.path.join(out_dir, "probe_metrics.csv")
    query_metrics_path = os.path.join(out_dir, "query_metrics.csv")
    decisions_path = os.path.join(out_dir, "decisions.jsonl")
    summary_path = os.path.join(out_dir, "run_summary.json")

    if args.overwrite:
        for p in [
            slot_metrics_path,
            probe_metrics_path,
            query_metrics_path,
            decisions_path,
            summary_path,
        ]:
            if os.path.exists(p):
                os.remove(p)

    config_snapshot_path = _save_run_config_file(args.config, out_dir)

    print("[warmup] running one warmup probe...")
    warm_batch = questions[: min(2, len(questions))]
    _ = system.run_probe(
        batch=warm_batch,
        w=optimizer.state.w_hat.copy(),
        slot_id=-1,
        probe_type="warmup",
        judge_enabled=False,
        verbose=False,
    )
    if judge_enabled:
        _ = system.run_probe(
            batch=warm_batch,
            w=optimizer.state.w_hat.copy(),
            slot_id=-2,
            probe_type="warmup_g",
            judge_enabled=True,
            verbose=False,
        )
    print("[warmup] done.")

    slot_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []

    # Collect all per-query end-to-end latencies from AdaRAG probes.
    # Used only for global_p95_latency_s.
    # This does not depend on save_query_metrics.
    global_tau_e2e_s: List[float] = []

    for t, batch in tqdm(list(enumerate(batches)), desc="AdaRAG slots"):
        w_exec, u_dir, w_hat = optimizer.propose()

        if slot_preheat:
            _ = system.run_probe(
                batch=batch,
                w=w_hat.copy(),
                slot_id=t,
                probe_type="preheat",
                judge_enabled=False,
                verbose=False,
            )

        res = system.run_probe(
            batch=batch,
            w=w_exec,
            slot_id=t,
            probe_type="adarag",
            judge_enabled=judge_enabled,
            verbose=False,
        )

        # Collect per-query latency from the formal AdaRAG probe for global p95.
        global_tau_e2e_s.extend(_extract_tau_e2e_from_probe_result(res))

        B = int(len(batch))
        if int(res.get("request_count", B)) != B:
            print(f"[WARN] res request_count={res.get('request_count')} != len(batch)={B}; use len(batch).")

        d_feedback = float(res["d_value"])
        p95_latency_s = float(res["p95_latency_s"])

        if judge_enabled and res.get("judge_accuracy", None) is not None:
            q_feedback = float(res["judge_accuracy"])
        else:
            q_feedback = float(res["q_value"])

        g_feedback = float(q_target - q_feedback)

        dbg = optimizer.update(
            u_dir=u_dir,
            d_feedback=d_feedback,
            q_feedback=q_feedback,
        )

        x_hat, y_hat, p_hat = optimizer.unpack_w(optimizer.state.w_hat)

        if save_probe_metrics:
            probe_row = {
                "slot": int(t),
                "probe_type": "adarag",
                "request_count": B,

                "d_value": d_feedback,
                "q_value": float(res["q_value"]),
                "accuracy": float(res["accuracy"]),
                "judge_accuracy": res.get("judge_accuracy", None),

                "p95_latency_s": p95_latency_s,
                "light_batch_wall_time_s": float(res.get("light_batch_wall_time_s", np.nan)),
                "batch_wall_time_s": float(res["batch_wall_time_s"]),
                "judge_wall_time_s": float(res.get("judge_wall_time_s", 0.0)),
            }

            if debug_metrics:
                probe_row.update({
                    "p_exec": float(res["p_exec"]),
                    "heavy_frac_real": float(res["heavy_frac_real"]),
                    "avg_docs_light": float(res["avg_docs_light"]),
                    "avg_docs_heavy": float(res["avg_docs_heavy"]),
                    "mean_light_retrieve_s": float(res["mean_light_retrieve_s"]),
                    "mean_heavy_retrieve_s": float(res["mean_heavy_retrieve_s"]),
                    "mean_prompt_build_s": float(res["mean_prompt_build_s"]),
                    "mean_llm_client_wait_s": float(res["mean_llm_client_wait_s"]),
                    "mean_llm_generate_s": float(res["mean_llm_generate_s"]),
                    "oracle_recall_light_full": float(res["oracle_recall_light_full"]),
                    "oracle_recall_heavy_full": float(res["oracle_recall_heavy_full"]),
                    "prompt_gold_rate": float(res["prompt_gold_rate"]),
                })

            probe_rows.append(probe_row)

        if save_query_metrics:
            query_rows.extend(res["query_rows"])

        slot_row = {
            "slot": int(t),
            "request_count": B,

            "d_feedback": d_feedback,
            "q_feedback": q_feedback,
            "g_feedback": g_feedback,
            "p95_latency_s": p95_latency_s,
            "light_batch_wall_time_s": float(res.get("light_batch_wall_time_s", np.nan)),

            "batch_wall_time_s": float(res["batch_wall_time_s"]),
            "judge_wall_time_s": float(res.get("judge_wall_time_s", 0.0)),

            "lambda": float(dbg["lambda"]),
            "update_norm": float(dbg["update_norm"]),

            "p_hat_after": float(p_hat),
            "x_sum_after": float(np.sum(x_hat)),
            "y_sum_after": float(np.sum(y_hat)),
            "x_top0_after": float(max(0.0, 1.0 - np.sum(x_hat))),
            "y_top0_after": float(max(0.0, 1.0 - np.sum(y_hat))),
        }

        if debug_metrics:
            slot_row.update({
                "v_norm": float(dbg["v_norm"]),
                "grad_d_norm": float(dbg["grad_d_norm"]),
                "grad_q_norm": float(dbg["grad_q_norm"]),
                "grad_cosine_d_q": float(dbg["grad_cosine_d_q"]),

                "E_light_docs_after": float(np.dot(x_hat, np.asarray(doc_choices_light, dtype=float))),
                "E_heavy_docs_after": float(np.dot(y_hat, np.asarray(doc_choices_heavy, dtype=float))),

                "p_exec": float(res["p_exec"]),
                "heavy_frac_real": float(res["heavy_frac_real"]),
                "avg_docs_light": float(res["avg_docs_light"]),
                "avg_docs_heavy": float(res["avg_docs_heavy"]),
                "mean_light_retrieve_s": float(res["mean_light_retrieve_s"]),
                "mean_heavy_retrieve_s": float(res["mean_heavy_retrieve_s"]),
                "mean_prompt_build_s": float(res["mean_prompt_build_s"]),
                "mean_llm_client_wait_s": float(res["mean_llm_client_wait_s"]),
                "mean_llm_generate_s": float(res["mean_llm_generate_s"]),
                "prompt_gold_rate": float(res["prompt_gold_rate"]),
            })

        slot_rows.append(slot_row)

        if save_decisions:
            _write_jsonl(decisions_path, {
                "slot": int(t),
                "request_count": B,
                "doc_choices_light": doc_choices_light,
                "doc_choices_heavy": doc_choices_heavy,
                "w_hat_before": w_hat.tolist(),
                "w_exec": w_exec.tolist(),
                "u_dir": u_dir.tolist(),
                "optimizer_debug": dbg,
                "config_snapshot": config_snapshot_path,
                "slot_preheat": {
                    "enabled": bool(slot_preheat),
                    "policy": "w_hat",
                    "judge_enabled": False,
                    "logged_to_metrics": False,
                },
            })

        pd.DataFrame(slot_rows).to_csv(slot_metrics_path, index=False)

        if save_probe_metrics:
            pd.DataFrame(probe_rows).to_csv(probe_metrics_path, index=False)

        if save_query_metrics:
            pd.DataFrame(query_rows).to_csv(query_metrics_path, index=False)

        print(
            f"\n[slot {t}] "
            f"B={B}, d={d_feedback:.4f}, p95={p95_latency_s:.4f}, q={q_feedback:.4f}, "
            f"lambda={dbg['lambda']:.4f}, p_hat={p_hat:.3f}, "
            f"x0={slot_row['x_top0_after']:.3f}, y0={slot_row['y_top0_after']:.3f}"
        )

        if debug_metrics:
            print(
                f"  [debug] "
                f"E_light={slot_row['E_light_docs_after']:.3f}, "
                f"E_heavy={slot_row['E_heavy_docs_after']:.3f}, "
                f"grad_d={slot_row['grad_d_norm']:.3f}, "
                f"grad_q={slot_row['grad_q_norm']:.3f}"
            )

    print("\n=== Done ===")
    print(f"Saved: {slot_metrics_path}")
    if save_probe_metrics:
        print(f"Saved: {probe_metrics_path}")
    if save_query_metrics:
        print(f"Saved: {query_metrics_path}")
    if save_decisions:
        print(f"Saved: {decisions_path}")
    print(f"Saved config: {config_snapshot_path}")
    print(f"Saved slot sizes: {slot_sizes_used_path}")

    df = pd.DataFrame(slot_rows)
    if not df.empty:
        total_requests = int(df["request_count"].sum())

        avg_q_slot = float(df["q_feedback"].mean())
        avg_d_slot = float(df["d_feedback"].mean())
        avg_p95_slot = float(df["p95_latency_s"].mean())
        avg_g_slot = float(q_target - avg_q_slot)

        avg_q_weighted = float(np.average(df["q_feedback"], weights=df["request_count"]))
        avg_d_weighted = float(np.average(df["d_feedback"], weights=df["request_count"]))

        global_p95_latency_s = _global_p95(global_tau_e2e_s)

        meet_weighted = bool(avg_q_weighted >= q_target)
        below_count = int((df["q_feedback"] < q_target).sum())

        summary = {
            "n_slots": int(len(df)),
            "total_requests": total_requests,
            "slot_sizes": [int(x) for x in df["request_count"].tolist()],
            "slot_sizing": cfg.get("slot_sizing", {}) or {},

            "q_target": q_target,

            "avg_q_feedback_slot_mean": avg_q_slot,
            "avg_q_feedback_weighted": avg_q_weighted,

            "avg_d_feedback_slot_mean": avg_d_slot,
            "avg_d_feedback_weighted": avg_d_weighted,

            "avg_p95_latency_slot_mean": avg_p95_slot,
            "global_p95_latency_s": global_p95_latency_s,
            "avg_g_feedback_slot_mean": avg_g_slot,

            "meet_target_by_weighted_average": meet_weighted,
            "below_target_slot_count": below_count,
            "below_target_slot_ratio": float(below_count / max(1, len(df))),

            "last_q_feedback": float(df["q_feedback"].iloc[-1]),
            "last_d_feedback": float(df["d_feedback"].iloc[-1]),
            "last_p95_latency_s": float(df["p95_latency_s"].iloc[-1]),
            "last_p_hat_after": float(df["p_hat_after"].iloc[-1]),

            "slot_preheat_enabled": bool(slot_preheat),
            "debug_metrics": debug_metrics,
            "save_probe_metrics": save_probe_metrics,
            "save_query_metrics": save_query_metrics,
            "save_decisions": save_decisions,
            "slot_sizes_used_path": slot_sizes_used_path,
            "config_snapshot": config_snapshot_path,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n[summary]")
        print(f"total_requests                = {total_requests}")
        print(f"avg_q_feedback_slot_mean      = {avg_q_slot:.6f}")
        print(f"avg_q_feedback_weighted       = {avg_q_weighted:.6f}")
        print(f"q_target                      = {q_target:.6f}")
        print(f"meet_target_by_weighted_avg   = {meet_weighted}")
        print(f"below_target_slot_count       = {below_count}/{len(df)}")
        print(f"avg_d_feedback_slot_mean      = {avg_d_slot:.6f}")
        print(f"avg_d_feedback_weighted       = {avg_d_weighted:.6f}")
        print(f"avg_p95_latency_slot_mean     = {avg_p95_slot:.6f}")
        print(f"global_p95_latency_s          = {global_p95_latency_s:.6f}")
        print(f"avg_g_feedback_slot_mean      = {avg_g_slot:.6f}")
        print(f"Saved summary: {summary_path}")

        plot_paths = _plot_adarag_curves(
            df,
            out_dir,
            q_target,
            global_p95_latency_s=global_p95_latency_s,
        )
        for k, p in plot_paths.items():
            print(f"Saved plot [{k}]: {p}")

        print("\n[tail slot metrics]")
        print(df.tail().round(4).to_string(index=False))


if __name__ == "__main__":
    main()