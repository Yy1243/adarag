# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def atomic_write_json(path: Path, obj: dict):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_write_faiss(index, path: Path):
    tmp = Path(str(path) + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def load_model(model_path: str, device: str, trust_remote_code: bool):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        return SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except TypeError:
        try:
            return SentenceTransformer(
                model_path,
                device=device,
                trust_remote_code=trust_remote_code,
            )
        except TypeError:
            return SentenceTransformer(model_path, device=device)


def make_index(dim: int, m: int, ef_construction: int, ef_search: int, metric: str):
    metric = metric.lower().strip()

    if metric == "ip":
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    elif metric == "l2":
        index = faiss.IndexHNSWFlat(dim, m)
    else:
        raise ValueError(f"metric must be 'ip' or 'l2', got {metric}")

    index.hnsw.efConstruction = int(ef_construction)
    index.hnsw.efSearch = int(ef_search)
    return index


def read_jsonl_batch(f, batch_size: int) -> Tuple[List[str], int, int]:
    """
    return:
      texts: 用于编码的文本列表
      raw_lines: 本批读取的原始行数
      byte_offset: 当前文件偏移量，用于快速断点恢复
    """
    texts: List[str] = []
    raw_lines = 0

    for _ in range(batch_size):
        line = f.readline()
        if not line:
            break

        raw_lines += 1
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        title = str(obj.get("title", "") or "")
        text = str(obj.get("text", obj.get("contents", obj.get("passage", ""))) or "")

        # 与当前检索器 Doc.text 格式保持一致
        if title:
            full_text = title + "\n" + text
        else:
            full_text = text

        # 关键：只截断异常长文本，避免 Jina 长上下文 attention 直接 OOM
        # 4000 字符通常已远大于 w100 passage 的正常长度，只影响异常长行
        MAX_TEXT_CHARS = 4000
        if len(full_text) > MAX_TEXT_CHARS:
            full_text = full_text[:MAX_TEXT_CHARS]

        texts.append(full_text)

    return texts, raw_lines, f.tell()


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _is_cuda_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return (
        isinstance(err, getattr(torch, "OutOfMemoryError", RuntimeError))
        or "cuda out of memory" in msg
        or "outofmemoryerror" in msg
    )


def safe_encode_texts(
    model,
    texts,
    batch_size: int,
    normalize: bool,
    min_batch_size: int = 8,
):
    """
    正常情况下使用 batch_size 编码。
    如果遇到 CUDA OOM，则只把当前 batch 递归拆小重试。
    由于 read_jsonl_batch 已经做了异常长文本截断，
    若拆到 min_batch_size 仍失败，则说明当前 batch_size 仍过大或显存未释放，直接抛错。
    """
    try:
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=bool(normalize),
            show_progress_bar=False,
        )

    except Exception as e:
        if not _is_cuda_oom(e):
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n = len(texts)

        if n <= min_batch_size:
            print(
                f"[OOM] batch n={n} still failed after text truncation. "
                f"Please reduce --batch-size or lower MAX_TEXT_CHARS.",
                flush=True,
            )
            raise

        mid = n // 2

        print(
            f"[OOM] batch n={n}, split into {mid} + {n - mid} and retry...",
            flush=True,
        )

        emb1 = safe_encode_texts(
            model=model,
            texts=texts[:mid],
            batch_size=max(1, min(batch_size // 2, mid)),
            normalize=normalize,
            min_batch_size=min_batch_size,
        )

        emb2 = safe_encode_texts(
            model=model,
            texts=texts[mid:],
            batch_size=max(1, min(batch_size // 2, n - mid)),
            normalize=normalize,
            min_batch_size=min_batch_size,
        )

        return np.concatenate([emb1, emb2], axis=0)
        
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--corpus", required=True)
    ap.add_argument("--index-out", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)

    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=128)

    ap.add_argument("--metric", choices=["ip", "l2"], default="ip")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")

    ap.add_argument("--save-every", type=int, default=200000)
    ap.add_argument("--print-every", type=int, default=10000)

    # 如果已知总行数，传入后可显示百分比和 ETA
    ap.add_argument("--total-lines", type=int, default=21015324)

    # 强制重建；慎用，会删除 partial/meta/final
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    corpus = Path(args.corpus)
    final_index_path = Path(args.index_out)
    partial_index_path = Path(str(final_index_path) + ".partial")
    meta_path = Path(str(final_index_path) + ".meta.json")

    if final_index_path.exists() and not args.force:
        print(f"[OK] final index already exists: {final_index_path}")
        print("Use --force only if you really want to rebuild it.")
        return

    if args.force:
        for p in [final_index_path, partial_index_path, meta_path]:
            if p.exists():
                print(f"[force] remove {p}")
                p.unlink()

    print("=" * 100)
    print("[CONFIG]")
    print("corpus            =", corpus)
    print("index_out         =", final_index_path)
    print("partial_index     =", partial_index_path)
    print("meta              =", meta_path)
    print("model             =", args.model)
    print("device            =", args.device)
    print("batch_size        =", args.batch_size)
    print("M                 =", args.m)
    print("efConstruction    =", args.ef_construction)
    print("efSearch          =", args.ef_search)
    print("metric            =", args.metric)
    print("normalize         =", args.normalize)
    print("trust_remote_code =", args.trust_remote_code)
    print("save_every        =", args.save_every)
    print("print_every       =", args.print_every)
    print("total_lines       =", args.total_lines)
    print("=" * 100, flush=True)

    model = load_model(
        model_path=args.model,
        device=args.device,
        trust_remote_code=bool(args.trust_remote_code),
    )

    index = None
    dim = None

    indexed = 0
    lines_seen = 0
    byte_offset = 0

    # resume
    if partial_index_path.exists() and meta_path.exists() and not args.force:
        meta = json.load(meta_path.open("r", encoding="utf-8"))

        indexed = int(meta.get("indexed", 0))
        lines_seen = int(meta.get("lines_seen", indexed))
        byte_offset = int(meta.get("byte_offset", 0))
        dim = int(meta["dim"])

        old_m = int(meta.get("m", args.m))
        old_metric = str(meta.get("metric", args.metric))
        old_norm = bool(meta.get("normalize", args.normalize))

        if old_m != args.m:
            raise RuntimeError(f"M mismatch: meta has {old_m}, current arg has {args.m}. Cannot resume.")
        if old_metric != args.metric:
            raise RuntimeError(f"metric mismatch: meta has {old_metric}, current arg has {args.metric}. Cannot resume.")
        if old_norm != bool(args.normalize):
            raise RuntimeError(f"normalize mismatch: meta has {old_norm}, current arg has {args.normalize}. Cannot resume.")

        print(f"[resume] loading partial index: {partial_index_path}", flush=True)
        index = faiss.read_index(str(partial_index_path))

        if int(index.ntotal) != indexed:
            raise RuntimeError(
                f"Resume mismatch: index.ntotal={index.ntotal}, meta.indexed={indexed}."
            )

        print(
            f"[resume] indexed={indexed}, lines_seen={lines_seen}, byte_offset={byte_offset}, dim={dim}",
            flush=True,
        )

    t0 = time.time()
    last_print_indexed = indexed
    last_print_t = t0
    last_save_indexed = indexed

    with corpus.open("r", encoding="utf-8") as f:
        if byte_offset > 0:
            f.seek(byte_offset)
            print(f"[resume] seek to byte_offset={byte_offset}", flush=True)

        while True:
            texts, raw_lines, new_offset = read_jsonl_batch(f, args.batch_size)
            if raw_lines == 0:
                break

            lines_seen += raw_lines
            byte_offset = new_offset

            if not texts:
                continue

            emb = safe_encode_texts(
                model=model,
                texts=texts,
                batch_size=args.batch_size,
                normalize=bool(args.normalize),
                min_batch_size=8,
            )   

            emb = np.asarray(emb, dtype="float32")
            if emb.ndim != 2:
                raise RuntimeError(f"Bad embedding shape: {emb.shape}")

            if index is None:
                dim = int(emb.shape[1])
                index = make_index(
                    dim=dim,
                    m=args.m,
                    ef_construction=args.ef_construction,
                    ef_search=args.ef_search,
                    metric=args.metric,
                )
                print(
                    f"[init] HNSW index created: dim={dim}, M={args.m}, "
                    f"efConstruction={args.ef_construction}, efSearch={args.ef_search}, metric={args.metric}",
                    flush=True,
                )

            index.add(emb)
            indexed = int(index.ntotal)

            # print progress
            if indexed - last_print_indexed >= args.print_every:
                now = time.time()
                dt = now - last_print_t
                total_elapsed = now - t0

                step_vecs = indexed - last_print_indexed
                step_speed = step_vecs / max(dt, 1e-9)
                avg_speed = max(indexed, 1) / max(total_elapsed, 1e-9)

                if args.total_lines and args.total_lines > 0:
                    pct = 100.0 * lines_seen / args.total_lines
                    remain = max(args.total_lines - lines_seen, 0)
                    eta = remain / max(avg_speed, 1e-9)
                    eta_s = fmt_time(eta)
                    progress_s = f"{pct:.2f}% ETA={eta_s}"
                else:
                    progress_s = "ETA=N/A"

                print(
                    f"[progress] indexed={indexed:,} lines_seen={lines_seen:,} "
                    f"{progress_s} "
                    f"step_speed={step_speed:.2f} vec/s avg_speed={avg_speed:.2f} vec/s "
                    f"elapsed={fmt_time(total_elapsed)}",
                    flush=True,
                )

                last_print_indexed = indexed
                last_print_t = now

            # save checkpoint
            if indexed - last_save_indexed >= args.save_every:
                now = time.time()
                print(
                    f"[checkpoint] saving partial index... indexed={indexed:,}, "
                    f"lines_seen={lines_seen:,}, byte_offset={byte_offset}",
                    flush=True,
                )

                atomic_write_faiss(index, partial_index_path)
                atomic_write_json(
                    meta_path,
                    {
                        "corpus": str(corpus),
                        "index_out": str(final_index_path),
                        "partial_index": str(partial_index_path),
                        "model": str(args.model),
                        "device": str(args.device),
                        "indexed": indexed,
                        "lines_seen": lines_seen,
                        "byte_offset": byte_offset,
                        "dim": int(dim),
                        "m": int(args.m),
                        "ef_construction": int(args.ef_construction),
                        "ef_search": int(args.ef_search),
                        "metric": str(args.metric),
                        "normalize": bool(args.normalize),
                        "trust_remote_code": bool(args.trust_remote_code),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )

                print(
                    f"[checkpoint] done. elapsed={fmt_time(now - t0)}",
                    flush=True,
                )
                last_save_indexed = indexed

    if index is None:
        raise RuntimeError("No vectors indexed. Check corpus file.")

    print("[final] saving final index...", flush=True)
    atomic_write_faiss(index, final_index_path)

    atomic_write_json(
        meta_path,
        {
            "corpus": str(corpus),
            "index_out": str(final_index_path),
            "model": str(args.model),
            "device": str(args.device),
            "indexed": int(index.ntotal),
            "lines_seen": int(lines_seen),
            "byte_offset": int(byte_offset),
            "dim": int(dim),
            "m": int(args.m),
            "ef_construction": int(args.ef_construction),
            "ef_search": int(args.ef_search),
            "metric": str(args.metric),
            "normalize": bool(args.normalize),
            "trust_remote_code": bool(args.trust_remote_code),
            "finished": True,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    total_elapsed = time.time() - t0
    print("=" * 100)
    print("[DONE]")
    print("final_index =", final_index_path)
    print("indexed     =", index.ntotal)
    print("lines_seen  =", lines_seen)
    print("dim         =", dim)
    print("metric_type =", index.metric_type)
    print("efConstruction =", index.hnsw.efConstruction)
    print("efSearch =", index.hnsw.efSearch)
    print("nb_neighbors(layer 0) =", index.hnsw.nb_neighbors(0))
    print("nb_neighbors(layer 1) =", index.hnsw.nb_neighbors(1))
    print("elapsed     =", fmt_time(total_elapsed))
    print("=" * 100)


if __name__ == "__main__":
    main()
