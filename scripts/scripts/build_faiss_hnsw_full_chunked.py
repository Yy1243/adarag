# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CORPUS_PATH = "/T20050013/adarag_repro/adarag_assets/data/kb/wiki_passages_w100_1m.jsonl"
INDEX_PATH = "/T20050013/adarag_repro/adarag_assets/data/kb/faiss_hnsw_full_bge.index"
STATE_PATH = INDEX_PATH + ".state.json"

EMBED_MODEL = "/T20050013/adarag_repro/adarag_assets/bge-base-en-v1.5/BAAI/bge-base-en-v1.5"
DEVICE = "cuda"   # 不稳就改 cpu

CHUNK_SIZE = 200_000
BATCH_SIZE = 128
SAVE_EVERY = 10   # 每 10 个 chunk 保存一次整索引和状态

HNSW_M = 64
EF_CONSTRUCTION = 200
EF_SEARCH = 128


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"next_line": 0, "chunks_done": 0, "ntotal": 0}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(next_line: int, chunks_done: int, ntotal: int) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"next_line": next_line, "chunks_done": chunks_done, "ntotal": ntotal},
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(tmp, STATE_PATH)


def iter_chunk(path: str, start_line: int, chunk_size: int) -> Tuple[List[str], int]:
    texts: List[str] = []
    next_line = start_line
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            obj = json.loads(line)
            title = str(obj.get("title", "") or "").strip()
            body = str(obj.get("text", obj.get("contents", "")) or "").strip()
            text = f"{title}\n{body}".strip() if title else body
            texts.append(text)
            next_line = i + 1
            if len(texts) >= chunk_size:
                break
    return texts, next_line


def persist_index_and_state(index, next_line: int, chunks_done: int) -> None:
    t0 = time.time()
    tmp_index = INDEX_PATH + ".tmp"
    faiss.write_index(index, tmp_index)
    os.replace(tmp_index, INDEX_PATH)
    save_state(
        next_line=next_line,
        chunks_done=chunks_done,
        ntotal=index.ntotal,
    )
    write_s = time.time() - t0
    print(
        f"[saved] ntotal={index.ntotal}, next_line={next_line}, "
        f"chunks_done={chunks_done}, write={write_s:.1f}s",
        flush=True,
    )


def main():
    state = load_state()
    start_line = int(state["next_line"])
    chunks_done = int(state["chunks_done"])

    print(f"[resume] next_line={start_line}, chunks_done={chunks_done}", flush=True)

    encoder = SentenceTransformer(
        EMBED_MODEL,
        device=DEVICE,
        trust_remote_code=False,
        model_kwargs={"local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
    )

    index = None
    if os.path.exists(INDEX_PATH):
        print(f"[load] existing index: {INDEX_PATH}", flush=True)
        index = faiss.read_index(INDEX_PATH)
        try:
            index.hnsw.efSearch = EF_SEARCH
        except Exception:
            pass

    while True:
        t_read0 = time.time()
        texts, next_line = iter_chunk(CORPUS_PATH, start_line=start_line, chunk_size=CHUNK_SIZE)
        read_s = time.time() - t_read0

        if not texts:
            print("[done] no more texts", flush=True)
            break

        print(
            f"[chunk {chunks_done+1}] lines {start_line} -> {next_line-1}, n={len(texts)}",
            flush=True,
        )

        t_enc0 = time.time()
        emb = encoder.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        encode_s = time.time() - t_enc0

        if index is None:
            dim = emb.shape[1]
            index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = EF_CONSTRUCTION
            index.hnsw.efSearch = EF_SEARCH

        t_add0 = time.time()
        index.add(emb)
        add_s = time.time() - t_add0

        chunks_done += 1
        start_line = next_line

        print(
            f"[timing] read={read_s:.1f}s encode={encode_s:.1f}s "
            f"add={add_s:.1f}s ntotal={index.ntotal}",
            flush=True,
        )

        if chunks_done % SAVE_EVERY == 0:
            persist_index_and_state(index, next_line=next_line, chunks_done=chunks_done)
        else:
            print(
                f"[buffered] ntotal={index.ntotal}, next_line={next_line}, "
                f"chunks_done={chunks_done}",
                flush=True,
            )

    if index is not None:
        # 最终保存一次，避免最后不足 SAVE_EVERY 的 chunk 丢失
        persist_index_and_state(index, next_line=start_line, chunks_done=chunks_done)
        print(f"[final] index_ntotal={index.ntotal}", flush=True)


if __name__ == "__main__":
    main()