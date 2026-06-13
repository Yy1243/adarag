# -*- coding: utf-8 -*-
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adarag.utils import load_yaml
from adarag.data_hf import load_nq_open_stream
from adarag.data import QAItem
from paretoadarag.pipeline.pareto_adarag_system_triprobe_mp import build_system_from_cfg


DEFAULT_CONFIG = "/T20050013/adarag_repro/scripts/config_pareto_adarag_3p_vllm.yaml"


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


def load_questions(cfg: dict, max_questions: int) -> List[QAItem]:
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
        if len(out) >= max_questions:
            break

    if not out:
        raise RuntimeError("No questions loaded.")
    return out


def fixed_w_light_topk(k: int, n_docs_light: int, n_docs_heavy: int, nu: float = 1.3) -> np.ndarray:
    """
    固定 light_topk：
    x[k-1] = 1, p = 0，不走重检索，不采样 top0。
    """
    if not (1 <= k <= n_docs_light):
        raise ValueError(f"k={k} invalid for n_docs_light={n_docs_light}")

    x = np.zeros(n_docs_light, dtype=float)
    y = np.zeros(n_docs_heavy, dtype=float)
    x[k - 1] = 1.0
    p = 0.0
    return np.concatenate([x, y, np.array([p, nu], dtype=float)])


def fixed_w_heavy_topk(k: int, n_docs_light: int, n_docs_heavy: int, nu: float = 1.3) -> np.ndarray:
    """
    固定 heavy_topk：
    y[k-1] = 1, p = 1，所有问题走重检索，不采样 top0。
    """
    if not (1 <= k <= n_docs_heavy):
        raise ValueError(f"k={k} invalid for n_docs_heavy={n_docs_heavy}")

    x = np.zeros(n_docs_light, dtype=float)
    y = np.zeros(n_docs_heavy, dtype=float)
    y[k - 1] = 1.0
    p = 1.0
    return np.concatenate([x, y, np.array([p, nu], dtype=float)])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--max-questions", type=int, default=300)
    ap.add_argument("--modes", type=str, default="light_top1,light_top2,light_top3")
    ap.add_argument("--output-dir", type=str, default="outputs_sanity_pareto_fixed_light123")
    ap.add_argument("--nu", type=float, default=1.3)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_docs_light = int(cfg.get("n_docs_light", cfg.get("n_docs_x", 3)))
    n_docs_heavy = int(cfg.get("n_docs_heavy", cfg.get("n_docs_y", cfg.get("n_docs", 10))))

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]

    print("========== Pareto Fixed-Policy Sanity Check ==========")
    print(f"[config] {args.config}")
    print(f"[max_questions] {args.max_questions}")
    print(f"[modes] {modes}")
    print(f"[n_docs_light] {n_docs_light}")
    print(f"[n_docs_heavy] {n_docs_heavy}")
    print(f"[output_dir] {args.output_dir}")
    print("======================================================")

    questions = load_questions(cfg, args.max_questions)

    print("[init] building Pareto pipeline system with judge...")
    system = build_system_from_cfg(cfg, judge_enabled=True)
    print("[init] system ready.")

    # Warmup：先预热 vLLM，再预热 judge。预热结果不计入最终结果。
    warm_batch = questions[: min(2, len(questions))]
    w_warm = fixed_w_light_topk(1, n_docs_light, n_docs_heavy, args.nu)

    _ = system.run_probe(
        batch=warm_batch,
        w=w_warm,
        slot_id=-1,
        probe_type="warmup_nojudge",
        judge_enabled=False,
        verbose=False,
    )

    _ = system.run_probe(
        batch=warm_batch,
        w=w_warm,
        slot_id=-2,
        probe_type="warmup_judge",
        judge_enabled=True,
        verbose=False,
    )

    print("[warmup] done.")

    summary_rows = []
    query_rows = []

    for idx, mode in enumerate(modes):
        mode_l = mode.lower()

        if mode_l.startswith("light_top"):
            k = int(mode_l.replace("light_top", ""))
            w = fixed_w_light_topk(k, n_docs_light, n_docs_heavy, args.nu)
        elif mode_l.startswith("heavy_top"):
            k = int(mode_l.replace("heavy_top", ""))
            w = fixed_w_heavy_topk(k, n_docs_light, n_docs_heavy, args.nu)
        else:
            raise ValueError(f"Unknown mode={mode}")

        print(f"\n[run] {mode}")

        res = system.run_probe(
            batch=questions,
            w=w,
            slot_id=idx,
            probe_type=mode,
            judge_enabled=True,
            verbose=True,
        )

        summary_rows.append({
            "mode": mode,
            "n_questions": int(res["request_count"]),
            "judge_accuracy": res.get("judge_accuracy", None),
            "token_accuracy": res.get("accuracy", None),
            "q_value": res.get("q_value", None),
            "prompt_gold_rate": res.get("prompt_gold_rate", None),

            "d_value": res.get("d_value", None),
            "p95_latency_s": res.get("p95_latency_s", None),
            "psi_value": res.get("psi_value", None),
            "batch_wall_time_s": res.get("batch_wall_time_s", None),
            "judge_wall_time_s": res.get("judge_wall_time_s", None),

            "heavy_frac_real": res.get("heavy_frac_real", None),
            "avg_docs_light": res.get("avg_docs_light", None),
            "avg_docs_heavy": res.get("avg_docs_heavy", None),

            "mean_light_retrieve_s": res.get("mean_light_retrieve_s", None),
            "mean_heavy_retrieve_s": res.get("mean_heavy_retrieve_s", None),
            "mean_llm_client_wait_s": res.get("mean_llm_client_wait_s", None),
            "mean_llm_generate_s": res.get("mean_llm_generate_s", None),
        })

        for row in res["query_rows"]:
            row = dict(row)
            row["mode"] = mode
            query_rows.append(row)

        pd.DataFrame(summary_rows).to_csv(out_dir / "sanity_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(query_rows).to_csv(out_dir / "sanity_query_metrics.csv", index=False, encoding="utf-8-sig")

        print(pd.DataFrame(summary_rows).round(4).to_string(index=False))

    print("\n=== Done ===")
    print(f"Saved summary: {out_dir / 'sanity_summary.csv'}")
    print(f"Saved query  : {out_dir / 'sanity_query_metrics.csv'}")


if __name__ == "__main__":
    main()