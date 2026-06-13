# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


def norm_basic(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_loose(s: str) -> str:
    s = norm_basic(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_answer(doc_text: str, answer: str) -> Tuple[bool, bool]:
    """
    return:
      strict_hit: 基本大小写/空白/破折号归一后的包含
      loose_hit:  去标点后的宽松包含
    """
    db = norm_basic(doc_text)
    ab = norm_basic(answer)

    strict_hit = bool(ab) and ab in db

    dl = norm_loose(doc_text)
    al = norm_loose(answer)

    if not al:
        loose_hit = False
    elif " " in al:
        loose_hit = al in dl
    else:
        loose_hit = re.search(rf"\b{re.escape(al)}\b", dl) is not None

    return strict_hit, loose_hit


def load_dataset(path: str, max_questions: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question", obj.get("q", ""))
            ans = obj.get("answer", obj.get("answers", []))
            if isinstance(ans, str):
                ans = [ans]
            ans = [str(a) for a in ans if str(a).strip()]
            rows.append({
                "qid": obj.get("qid", obj.get("id", i)),
                "question": str(q),
                "answers": ans,
            })
            if max_questions is not None and len(rows) >= max_questions:
                break
    return rows


def es_query_for_answer(answer: str, size: int) -> Dict[str, Any]:
    """
    用答案本身去 ES 里搜。
    match_phrase 负责短语匹配；
    cross_fields 负责 title/text 联合字段兜底。
    """
    return {
        "size": size,
        "_source": ["doc_id", "title", "text"],
        "query": {
            "bool": {
                "should": [
                    {
                        "match_phrase": {
                            "title": {
                                "query": answer,
                                "boost": 4.0,
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "text": {
                                "query": answer,
                                "boost": 2.0,
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": answer,
                            "type": "cross_fields",
                            "fields": ["title^3", "text"],
                            "operator": "and",
                            "boost": 0.5,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
    }


def search_answer(
    es_url: str,
    index_name: str,
    answer: str,
    hit_size: int,
    timeout: int,
) -> Tuple[bool, bool, str, str, str, float]:
    """
    return:
      strict_hit, loose_hit, matched_doc_id, matched_title, snippet, elapsed
    """
    url = f"{es_url.rstrip('/')}/{index_name}/_search"
    body = es_query_for_answer(answer, hit_size)

    t0 = time.time()
    r = requests.post(url, json=body, timeout=timeout)
    elapsed = time.time() - t0
    r.raise_for_status()

    hits = r.json().get("hits", {}).get("hits", [])

    best_loose = None

    for h in hits:
        src = h.get("_source", {}) or {}
        doc_id = str(src.get("doc_id", h.get("_id", "")))
        title = str(src.get("title", "") or "")
        text = str(src.get("text", "") or "")
        full = (title + "\n" + text).strip()

        strict_hit, loose_hit = contains_answer(full, answer)

        if strict_hit:
            snippet = full[:300].replace("\n", " ")
            return True, True, doc_id, title, snippet, elapsed

        if loose_hit and best_loose is None:
            snippet = full[:300].replace("\n", " ")
            best_loose = (False, True, doc_id, title, snippet, elapsed)

    if best_loose is not None:
        return best_loose

    return False, False, "", "", "", elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--es-url", default="http://127.0.0.1:9200")
    ap.add_argument("--index-name", default="adarag_kb_v2_1m")
    ap.add_argument("--max-questions", type=int, default=3000)
    ap.add_argument("--hit-size", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default="outputs_answer_coverage_es.csv")
    args = ap.parse_args()

    data = load_dataset(args.dataset, args.max_questions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    strict_q = 0
    loose_q = 0
    any_es_q = 0

    rows = []

    t_all = time.time()

    for idx, item in enumerate(data, 1):
        qid = item["qid"]
        q = item["question"]
        answers = item["answers"]

        q_strict = False
        q_loose = False
        q_any_es = False

        matched_answer = ""
        matched_doc_id = ""
        matched_title = ""
        matched_snippet = ""
        total_es_time = 0.0

        for ans in answers:
            try:
                strict_hit, loose_hit, doc_id, title, snippet, elapsed = search_answer(
                    es_url=args.es_url,
                    index_name=args.index_name,
                    answer=ans,
                    hit_size=args.hit_size,
                    timeout=args.timeout,
                )
            except Exception as e:
                print(f"[WARN] qid={qid} answer={ans!r} ES failed: {repr(e)}", flush=True)
                strict_hit = loose_hit = False
                doc_id = title = snippet = ""
                elapsed = 0.0

            total_es_time += elapsed

            if doc_id or title or snippet:
                q_any_es = True

            if strict_hit or loose_hit:
                matched_answer = ans
                matched_doc_id = doc_id
                matched_title = title
                matched_snippet = snippet

            q_strict = q_strict or strict_hit
            q_loose = q_loose or loose_hit

            if q_strict:
                break

        n += 1
        strict_q += int(q_strict)
        loose_q += int(q_loose)
        any_es_q += int(q_any_es)

        rows.append({
            "qid": qid,
            "question": q,
            "answers_json": json.dumps(answers, ensure_ascii=False),
            "strict_covered": int(q_strict),
            "loose_covered": int(q_loose),
            "es_returned_any_candidate": int(q_any_es),
            "matched_answer": matched_answer,
            "matched_doc_id": matched_doc_id,
            "matched_title": matched_title,
            "matched_snippet": matched_snippet,
            "es_time_s": total_es_time,
        })

        if idx % 100 == 0:
            print(
                f"[progress] {idx}/{len(data)} "
                f"strict={strict_q / n:.4f} loose={loose_q / n:.4f} "
                f"elapsed={time.time() - t_all:.1f}s",
                flush=True,
            )

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_questions": n,
        "strict_coverage": strict_q / max(n, 1),
        "loose_coverage": loose_q / max(n, 1),
        "es_returned_any_candidate_rate": any_es_q / max(n, 1),
        "hit_size_per_answer": args.hit_size,
        "dataset": args.dataset,
        "es_index": args.index_name,
        "out_csv": str(out_path),
        "elapsed_s": time.time() - t_all,
    }

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved CSV:", out_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
