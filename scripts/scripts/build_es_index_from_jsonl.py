# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def bulk_send(es_url: str, index_name: str, actions: list[str], timeout: int = 120):
    if not actions:
        return
    url = f"{es_url.rstrip('/')}/_bulk"
    data = "\n".join(actions) + "\n"
    r = requests.post(
        url,
        data=data.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
        timeout=timeout,
    )
    if r.status_code >= 400:
        print(r.text[:3000])
        r.raise_for_status()
    resp = r.json()
    if resp.get("errors"):
        bad = []
        for item in resp.get("items", []):
            err = item.get("index", {}).get("error")
            if err:
                bad.append(err)
            if len(bad) >= 3:
                break
        raise RuntimeError(f"Bulk indexing errors: {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--es-url", default="http://127.0.0.1:9200")
    ap.add_argument("--index-name", required=True)
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--batch-size", type=int, default=2000)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    actions = []
    n = 0
    t0 = time.time()

    with corpus.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            doc_id = str(obj.get("id", obj.get("doc_id", n)))
            title = str(obj.get("title", "") or "")
            text = str(obj.get("text", obj.get("contents", obj.get("passage", ""))) or "")

            actions.append(json.dumps({"index": {"_index": args.index_name, "_id": doc_id}}, ensure_ascii=False))
            actions.append(json.dumps({"doc_id": doc_id, "title": title, "text": text}, ensure_ascii=False))

            n += 1

            if len(actions) >= args.batch_size * 2:
                bulk_send(args.es_url, args.index_name, actions)
                actions = []
                if n % 20000 == 0:
                    print(f"indexed {n}, elapsed={time.time() - t0:.1f}s", flush=True)

            if args.limit and n >= args.limit:
                break

    if actions:
        bulk_send(args.es_url, args.index_name, actions)

    requests.post(f"{args.es_url.rstrip('/')}/{args.index_name}/_refresh", timeout=120)
    requests.put(
        f"{args.es_url.rstrip('/')}/{args.index_name}/_settings",
        json={"index": {"refresh_interval": "1s"}},
        timeout=120,
    )

    print(f"Done. indexed={n}, elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
