# -*- coding: utf-8 -*-
"""
run_adarag_edge.py (SYNC, B-scheme)

Same loop/optimizer/logging as run_adarag_real.py, but offload run_slot to edge servers:
  res = POST {edge}/run_slot

Edge servers must be started first.
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Any, List

import numpy as np
import pandas as pd
import httpx
from tqdm import tqdm

from adarag.data import QAItem
from adarag.utils import load_yaml
from adarag.optimizer.bandit_optimizer import AdaRAGBanditOptimizer
from adarag.data_hf import load_nq_open_stream


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--edges", type=str, default="http://127.0.0.1:3300,http://127.0.0.1:3301,http://127.0.0.1:3302,http://127.0.0.1:3303,http://127.0.0.1:3304")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    out_dir = cfg.get("output_dir", "outputs_real_min")
    out_dir = out_dir + "_edge"  # avoid clobbering local baseline
    _ensure_dir(out_dir)

    seed = int(cfg.get("seed", 42))

    # --------------------------
    # data stream (same as baseline)
    # --------------------------
    ds_cfg = cfg["dataset"]
    it = load_nq_open_stream(
        split=ds_cfg.get("split", "validation"),
        seed=int(ds_cfg.get("seed", seed)),
        local_path=ds_cfg["local_path"],
        max_examples=ds_cfg.get("max_examples", None),
    )
    raw_items = list(it)
    if not raw_items:
        raise RuntimeError("Dataset stream is empty. Check dataset.local_path and max_examples.")

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
    # optimizer (same as baseline)
    # --------------------------
    n_docs = int(cfg.get("n_docs", 10))
    opt_cfg = cfg.get("optimizer", {}) or {}
    optimizer = AdaRAGBanditOptimizer(
        n_docs=n_docs,
        Q_target=float(cfg.get("quality_target", 0.0)),
        alpha=float(opt_cfg.get("alpha", 0.08)),
        mu=float(opt_cfg.get("mu", 0.3)),
        delta=float(opt_cfg.get("delta", 0.05)),
        gamma=float(opt_cfg.get("gamma", 0.05)),
        seed=seed,
    )

    latency_target_s = float(cfg.get("latency_target_s", 0.05))
    decisions_path = os.path.join(out_dir, "decisions.jsonl")
    metrics_path = os.path.join(out_dir, "slot_metrics.csv")
    examples_path = os.path.join(out_dir, "slot_examples.jsonl")

    if args.overwrite:
        open(decisions_path, "w", encoding="utf-8").close()
        open(examples_path, "w", encoding="utf-8").close()

    edges = [e.strip().rstrip("/") for e in args.edges.split(",") if e.strip()]
    if not edges:
        raise ValueError("No edges provided.")
    print(f"[EdgeMode] edges={edges}")

    rows = []
    lam = float(optimizer.state.lambda_)

    with httpx.Client(timeout=600) as client:
        for t in tqdm(range(n_slots), desc="Slots(edge)"):
            raw_batch = qa_used[t * slot_size: (t + 1) * slot_size]
            batch = [_make_qa(x) for x in raw_batch]

            z_perturbed, u, z_hat = optimizer.propose()

            edge = edges[t % len(edges)]  # round-robin (replace with your scheduler later)
            body = {
                "items": [{"qid": qa.qid, "q": qa.q, "a": qa.a} for qa in batch],
                "z": z_perturbed.tolist(),
                "latency_target_s": 1e9,  # keep same as run_adarag_real.py
            }
            res = client.post(edge + "/run_slot", json=body).json()

            judge_acc = res.get("judge_accuracy", None)
            if judge_acc is not None:
                q_feedback = float(judge_acc)
            else:
                q_feedback = float(res.get("accuracy", 0.0))
            d_feedback = float(res.get("latency_s", 0.0))

            dbg = optimizer.update(
                z_perturbed=np.asarray(z_perturbed, dtype=float),
                u=np.asarray(u, dtype=float),
                d_feedback=d_feedback,
                q_feedback=q_feedback,
            )
            lam = float(optimizer.state.lambda_)

            # dump examples (same as baseline)
            with open(examples_path, "a", encoding="utf-8") as fex:
                for ex in res.get("examples", []):
                    ex = dict(ex)
                    ex["slot"] = int(t)
                    ex["lambda"] = float(lam)
                    ex["p"] = float(res.get("p", 0.0))
                    ex["edge"] = edge
                    fex.write(json.dumps(ex, ensure_ascii=False) + "\n")

            row = {
                "slot": t,
                "latency_s": d_feedback,
                "accuracy": q_feedback,
                "p": float(res.get("p", 0.0)),
                "heavy_frac_real": float(res.get("heavy_frac_real", 0.0)),
                "avg_docs_light": float(res.get("avg_docs_light", 0.0)),
                "avg_docs_heavy": float(res.get("avg_docs_heavy", 0.0)),
                "oracle_recall_any": float(res.get("oracle_recall_any", 0.0)),
                "prompt_gold_rate": float(res.get("prompt_gold_rate", 0.0)),
                "lambda": float(lam),
                "accuracy_contains": float(res.get("accuracy", 0.0)),
                "oracle_recall_any_full": float(res.get("oracle_recall_any_full", 0.0)),
                "oracle_recall_light_full": float(res.get("oracle_recall_light_full", 0.0)),
                "oracle_recall_heavy_full": float(res.get("oracle_recall_heavy_full", 0.0)),
                "edge": edge,
            }
            rows.append(row)

            with open(decisions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "t": t + 1,
                    "q_feedback": q_feedback,
                    "d_feedback": d_feedback,
                    "z_hat": dbg.get("z_hat", None),
                    "z_perturbed": dbg.get("z_perturbed", None),
                    "lambda": float(lam),
                    "edge": edge,
                }, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False)

    print("\n=== Done (EDGE) ===")
    print(df.tail())
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {decisions_path}")
    print(f"Saved: {examples_path}")


if __name__ == "__main__":
    main()