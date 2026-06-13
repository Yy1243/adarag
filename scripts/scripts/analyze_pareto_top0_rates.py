# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


def load_decision_stats(decisions_path: Path, n_docs_light: int, n_docs_heavy: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if not decisions_path.exists():
        print(f"[WARN] decisions.jsonl not found: {decisions_path}")
        return pd.DataFrame()

    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        slot = int(obj["slot"])

        for probe_type, key in [
            ("d", "w_exec_d"),
            ("psi", "w_exec_psi"),
            ("g", "w_exec_g"),
        ]:
            if key not in obj:
                continue

            w = np.asarray(obj[key], dtype=float)

            x = w[:n_docs_light]
            y = w[n_docs_light:n_docs_light + n_docs_heavy]
            p = float(w[-2])
            nu = float(w[-1])

            x_top0 = max(0.0, 1.0 - float(np.sum(x)))
            y_top0 = max(0.0, 1.0 - float(np.sum(y)))

            e_light_docs = float(np.dot(x, np.arange(1, n_docs_light + 1)))
            e_heavy_docs = float(np.dot(y, np.arange(1, n_docs_heavy + 1)))

            row = {
                "slot": slot,
                "probe_type": probe_type,
                "p_exec_from_decision": p,
                "nu_exec_from_decision": nu,
                "x_sum": float(np.sum(x)),
                "y_sum": float(np.sum(y)),
                "x_top0_expected": x_top0,
                "y_top0_expected": y_top0,
                "E_light_docs_expected": e_light_docs,
                "E_heavy_docs_expected": e_heavy_docs,
            }

            for i, val in enumerate(x, start=1):
                row[f"x_top{i}_prob"] = float(val)
            for i, val in enumerate(y, start=1):
                row[f"y_top{i}_prob"] = float(val)

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=str,
        default="outputs_pareto_adarag_3p_vllm",
        help="ParetoAdaRAG-3P output directory.",
    )
    ap.add_argument("--n-docs-light", type=int, default=3)
    ap.add_argument("--n-docs-heavy", type=int, default=10)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output csv path. Default: <output-dir>/top0_rate_analysis.csv",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    query_path = out_dir / "query_metrics.csv"
    probe_path = out_dir / "probe_metrics.csv"
    slot_path = out_dir / "slot_metrics.csv"
    decisions_path = out_dir / "decisions.jsonl"

    if not query_path.exists():
        raise FileNotFoundError(f"query_metrics.csv not found: {query_path}")

    q = pd.read_csv(query_path)

    # Aggregate actual query-level sampling outcomes.
    route_agg = (
        q.groupby(["slot", "probe_type", "route"])
        .agg(
            n_queries=("qid", "count"),
            actual_zero_doc_rate=("n_take_docs", lambda s: float((s == 0).mean())),
            actual_mean_docs=("n_take_docs", "mean"),
            actual_top1_rate=("n_take_docs", lambda s: float((s == 1).mean())),
            actual_top2_rate=("n_take_docs", lambda s: float((s == 2).mean())),
            actual_top3plus_rate=("n_take_docs", lambda s: float((s >= 3).mean())),
            prompt_gold_rate=("prompt_has_gold", "mean"),
            contains_acc=("correct_contains", "mean"),
            mean_tau_s=("tau_e2e_s", "mean"),
            p95_tau_s=("tau_e2e_s", lambda s: float(np.quantile(s, 0.95))),
        )
        .reset_index()
    )

    # Pivot route-level actual stats into one row per slot/probe.
    rows = []
    for (slot, probe), df in route_agg.groupby(["slot", "probe_type"]):
        row: Dict[str, Any] = {
            "slot": int(slot),
            "probe_type": probe,
        }

        total_n = int(df["n_queries"].sum())
        row["n_queries_total"] = total_n

        for _, r in df.iterrows():
            route = str(r["route"])
            prefix = f"{route}_"

            row[prefix + "n_queries"] = int(r["n_queries"])
            row[prefix + "zero_doc_rate_actual"] = float(r["actual_zero_doc_rate"])
            row[prefix + "mean_docs_actual"] = float(r["actual_mean_docs"])
            row[prefix + "top1_rate_actual"] = float(r["actual_top1_rate"])
            row[prefix + "top2_rate_actual"] = float(r["actual_top2_rate"])
            row[prefix + "top3plus_rate_actual"] = float(r["actual_top3plus_rate"])
            row[prefix + "prompt_gold_rate"] = float(r["prompt_gold_rate"])
            row[prefix + "contains_acc"] = float(r["contains_acc"])
            row[prefix + "mean_tau_s"] = float(r["mean_tau_s"])
            row[prefix + "p95_tau_s"] = float(r["p95_tau_s"])

        # Overall actual zero-doc rate across all routes.
        sub = q[(q["slot"] == slot) & (q["probe_type"] == probe)]
        row["overall_zero_doc_rate_actual"] = float((sub["n_take_docs"] == 0).mean())
        row["overall_mean_docs_actual"] = float(sub["n_take_docs"].mean())
        row["overall_prompt_gold_rate"] = float(sub["prompt_has_gold"].mean())
        row["overall_contains_acc"] = float(sub["correct_contains"].mean())

        rows.append(row)

    actual_df = pd.DataFrame(rows)

    # Merge probe-level quality/latency feedback.
    if probe_path.exists():
        probe_df = pd.read_csv(probe_path)
        keep = [
            "slot", "probe_type",
            "d_value", "psi_value", "g_value", "q_value",
            "accuracy", "judge_accuracy", "p95_latency_s",
            "batch_wall_time_s", "judge_wall_time_s",
            "p_exec", "nu_exec", "heavy_frac_real",
            "avg_docs_light", "avg_docs_heavy",
            "prompt_gold_rate",
        ]
        keep = [c for c in keep if c in probe_df.columns]
        actual_df = actual_df.merge(probe_df[keep], on=["slot", "probe_type"], how="left")

    # Merge slot-level optimizer q feedback, especially for g probe comparison.
    if slot_path.exists():
        slot_df = pd.read_csv(slot_path)
        keep = [
            "slot", "q_feedback", "g_feedback",
            "theta1", "theta2", "lambda",
            "p_hat_after", "nu_hat_after",
            "x_sum_after", "y_sum_after",
        ]
        keep = [c for c in keep if c in slot_df.columns]
        actual_df = actual_df.merge(slot_df[keep], on="slot", how="left")

    # Merge expected top0 probabilities from decisions.
    dec_df = load_decision_stats(decisions_path, args.n_docs_light, args.n_docs_heavy)
    if not dec_df.empty:
        actual_df = actual_df.merge(dec_df, on=["slot", "probe_type"], how="left")

    # Add a helpful flag: only g-probe has judge q feedback.
    actual_df["is_g_probe"] = actual_df["probe_type"].eq("g").astype(int)

    out_path = Path(args.out) if args.out else (out_dir / "top0_rate_analysis.csv")
    actual_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved: {out_path}")
    print("\n[g-probe summary]")
    g = actual_df[actual_df["probe_type"] == "g"].copy()
    show_cols = [
        "slot",
        "q_feedback", "judge_accuracy", "q_value",
        "overall_zero_doc_rate_actual", "overall_mean_docs_actual",
        "light_zero_doc_rate_actual", "light_mean_docs_actual",
        "heavy_zero_doc_rate_actual", "heavy_mean_docs_actual",
        "x_top0_expected", "E_light_docs_expected",
        "y_top0_expected", "E_heavy_docs_expected",
        "heavy_frac_real", "p_exec",
        "overall_prompt_gold_rate",
    ]
    show_cols = [c for c in show_cols if c in g.columns]
    if len(g):
        print(g[show_cols].round(4).to_string(index=False))
    else:
        print("[WARN] no g-probe rows found.")

    print("\n[Correlation on g-probe]")
    corr_cols = [
        "q_feedback",
        "judge_accuracy",
        "overall_zero_doc_rate_actual",
        "overall_mean_docs_actual",
        "light_zero_doc_rate_actual",
        "light_mean_docs_actual",
        "heavy_zero_doc_rate_actual",
        "heavy_mean_docs_actual",
        "overall_prompt_gold_rate",
        "x_top0_expected",
        "y_top0_expected",
        "E_light_docs_expected",
        "E_heavy_docs_expected",
        "heavy_frac_real",
    ]
    corr_cols = [c for c in corr_cols if c in g.columns]
    if len(g) >= 2:
        print(g[corr_cols].corr(numeric_only=True).round(4).to_string())
    else:
        print("[WARN] not enough g-probe rows for correlation.")


if __name__ == "__main__":
    main()